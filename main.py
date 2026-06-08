from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from typing import Optional
from pathlib import Path
import base64
import os
import re
import tempfile
import io
import json

import cv2
import numpy as np
from PIL import Image, ImageFilter, ExifTags

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader


app = FastAPI()

API_KEY = os.getenv("API_KEY", "")

# ── Physical document dimensions (true card size, used for aspect ratio only) ─

ID_CARD_WIDTH_MM  = 85.60
ID_CARD_HEIGHT_MM = 53.98
ID_CARD_ASPECT    = ID_CARD_WIDTH_MM / ID_CARD_HEIGHT_MM  # 1.5857

PASSPORT_SCAN_WIDTH_MM  = 176.0
PASSPORT_SCAN_HEIGHT_MM = 125.0
PASSPORT_ASPECT         = PASSPORT_SCAN_WIDTH_MM / PASSPORT_SCAN_HEIGHT_MM

# ── Layout: sizes on the A4 page (measured from reference PDFs) ───────────────
# These match the Word "Centre Shadow Rectangle" output you provided.

# ID card: rendered at ~107×74mm including shadow padding
ID_RENDER_WIDTH_MM  = 107.0
ID_RENDER_HEIGHT_MM = 74.0

# Positions on A4 page (top-left of image including shadow)
ID_FRONT_X_MM = 17.0
ID_FRONT_Y_MM = 15.0   # from top of page
ID_BACK_Y_MM  = 93.0   # front bottom + gap

# Passport: rendered at ~114×160mm including shadow padding
PASSPORT_RENDER_WIDTH_MM  = 114.0
PASSPORT_RENDER_HEIGHT_MM = 160.0
PASSPORT_X_MM = 6.0
PASSPORT_Y_MM = 13.0

DPI = 300

# ── Shadow settings (replicates Word Centre Shadow Rectangle) ─────────────────

SHADOW_OFFSET_MM = 2.5
SHADOW_BLUR_MM   = 3.5
SHADOW_OPACITY   = 140
SHADOW_COLOR     = (100, 100, 100)

# ── Dark edge trim ────────────────────────────────────────────────────────────
# After perspective correction, trim this many pixels from each edge
# to remove any residual dark background from the photo.
EDGE_TRIM_PX = 8


# ── Health endpoints ──────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"status": "ok", "message": "ID Clean Document API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}


# ── General utilities ─────────────────────────────────────────────────────────

def safe_file_part(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = value.strip("_")
    return value or "clean_document"

def mm_to_px(mm_value: float, dpi: int = DPI) -> int:
    return int(round(mm_value / 25.4 * dpi))

def mm_to_pt(mm_value: float) -> float:
    return mm_value / 25.4 * 72.0


# ── EXIF-aware image reading ──────────────────────────────────────────────────

def read_image_exif_aware(path: Path) -> np.ndarray:
    pil_img = Image.open(path)
    try:
        exif = pil_img._getexif()
        if exif:
            orientation_key = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
            )
            if orientation_key and orientation_key in exif:
                orientation = exif[orientation_key]
                rotations = {3: 180, 6: -90, 8: 90}
                if orientation in rotations:
                    pil_img = pil_img.rotate(rotations[orientation], expand=True)
                elif orientation in {2, 4, 5, 7}:
                    pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                    if orientation in {5, 7}:
                        pil_img = pil_img.rotate(90 if orientation == 5 else -90, expand=True)
    except Exception:
        pass
    pil_img = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_BGR2RGB)

def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

def rgb_to_pil(image_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(image_rgb)


# ── Perspective correction ────────────────────────────────────────────────────

def parse_corners(corners_json: Optional[str]) -> Optional[np.ndarray]:
    if not corners_json:
        return None
    try:
        data = json.loads(corners_json)
        pts  = np.array([
            data["top_left"],
            data["top_right"],
            data["bottom_right"],
            data["bottom_left"],
        ], dtype="float32")
        return pts
    except Exception:
        return None

def parse_rotation(corners_json: Optional[str]) -> int:
    if not corners_json:
        return 0
    try:
        data = json.loads(corners_json)
        return int(data.get("rotation_needed", 0))
    except Exception:
        return 0

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s    = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff    = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect       = order_points(pts)
    tl, tr, br, bl = rect
    max_width  = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    max_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if max_width < 50 or max_height < 50:
        return image
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype="float32")
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(
        image, matrix, (max_width, max_height),
        borderValue=(255, 255, 255),
    )

def apply_rotation(image: np.ndarray, rotation_needed: int) -> np.ndarray:
    if rotation_needed == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_needed in (-90, 270):
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_needed == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    return image

def trim_dark_edges(image: np.ndarray, trim_px: int = EDGE_TRIM_PX) -> np.ndarray:
    """
    Trim a fixed number of pixels from all edges after perspective correction.
    This removes any residual dark background that bleeds in from the photo edges.
    """
    h, w = image.shape[:2]
    t = trim_px
    if 2 * t >= h or 2 * t >= w:
        return image
    return image[t:h-t, t:w-t]


# ── Orientation helpers ───────────────────────────────────────────────────────

def force_landscape(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h > w:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image

def force_passport_orientation(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h > w:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


# ── Card renderer ─────────────────────────────────────────────────────────────

def render_card(
    card_rgb: np.ndarray,
    target_aspect: float,
    doc_type: str = "id",
) -> Image.Image:
    """
    Render the card as a PIL RGB image with drop shadow.
    The card is scaled to fit inside the target render dimensions
    while preserving aspect ratio. White padding fills any gap.
    Shadow replicates Word Centre Shadow Rectangle style.
    """
    # Normalise orientation
    if doc_type == "passport":
        card_rgb = force_passport_orientation(card_rgb)
    else:
        card_rgb = force_landscape(card_rgb)

    # Determine target pixel dimensions from layout constants
    if doc_type == "passport":
        render_w_mm = PASSPORT_RENDER_WIDTH_MM
        render_h_mm = PASSPORT_RENDER_HEIGHT_MM
    else:
        render_w_mm = ID_RENDER_WIDTH_MM
        render_h_mm = ID_RENDER_HEIGHT_MM

    shadow_offset_px = mm_to_px(SHADOW_OFFSET_MM)
    shadow_blur_px   = mm_to_px(SHADOW_BLUR_MM)
    shadow_pad       = shadow_blur_px * 2 + shadow_offset_px

    # Card area inside the render box (subtract shadow padding)
    card_area_w_px = mm_to_px(render_w_mm) - shadow_pad
    card_area_h_px = mm_to_px(render_h_mm) - shadow_pad

    # Scale card to fit inside card area, preserve aspect ratio
    h, w = card_rgb.shape[:2]
    scale = min(card_area_w_px / w, card_area_h_px / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(card_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    card_pil = rgb_to_pil(resized)

    # Total canvas size
    canvas_w = card_area_w_px + shadow_pad
    canvas_h = card_area_h_px + shadow_pad

    # Card position (top-left of card area, centred within card_area)
    card_x = (card_area_w_px - new_w) // 2
    card_y = (card_area_h_px - new_h) // 2

    # Shadow position (offset from card)
    shadow_x = card_x + shadow_offset_px
    shadow_y = card_y + shadow_offset_px

    # White base canvas
    base = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    # Shadow layer
    shadow_layer = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
    shadow_rect  = Image.new(
        "RGBA", (new_w, new_h),
        (SHADOW_COLOR[0], SHADOW_COLOR[1], SHADOW_COLOR[2], SHADOW_OPACITY),
    )
    shadow_layer.paste(shadow_rect, (shadow_x, shadow_y))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_blur_px))

    # Composite: base → shadow → card
    base = Image.alpha_composite(base, shadow_layer)
    base.paste(card_pil, (card_x, card_y))

    return base.convert("RGB")


# ── PIL → ReportLab ImageReader ───────────────────────────────────────────────

def pil_to_image_reader(pil_img: Image.Image) -> ImageReader:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", dpi=(DPI, DPI))
    buf.seek(0)
    return ImageReader(buf)


# ── Full processing pipeline ──────────────────────────────────────────────────

def process_image(
    input_path: Path,
    output_path: Path,
    doc_type: str = "id",
    corners_json: Optional[str] = None,
) -> dict:
    """
    1. EXIF-aware read
    2. Perspective correction using Claude Vision corners
    3. Rotation correction
    4. Dark edge trim (removes residual background)
    5. Render with shadow at layout dimensions
    6. Save PNG at 300 DPI
    """
    image = read_image_exif_aware(input_path)

    pts = parse_corners(corners_json)
    if pts is not None:
        image  = four_point_transform(image, pts)
        method = "claude_corners_perspective_corrected"
    else:
        method = "no_corners_fallback"

    rotation = parse_rotation(corners_json)
    if rotation != 0:
        image  = apply_rotation(image, rotation)
        method += f"_rotated_{rotation}"

    # Trim dark edges after perspective correction
    image = trim_dark_edges(image, trim_px=EDGE_TRIM_PX)

    target_aspect = (
        PASSPORT_ASPECT if doc_type == "passport" else ID_CARD_ASPECT
    )

    final_pil = render_card(image, target_aspect, doc_type=doc_type)
    final_pil.save(output_path, dpi=(DPI, DPI))

    return {
        "input_file":  input_path.name,
        "output_file": output_path.name,
        "method":      method,
    }


# ── DOCX helpers ──────────────────────────────────────────────────────────────

def add_image_at_position(
    doc: Document,
    image_path: Path,
    width_cm: float,
):
    """Add image paragraph left-aligned with standard indent."""
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after  = Pt(0)
    run = paragraph.add_run()
    run.add_picture(str(image_path), width=Cm(width_cm))
    return paragraph


def create_docx(
    docx_path: Path,
    doc_type: str,
    front_image: Path,
    back_image: Optional[Path],
):
    doc     = Document()
    section = doc.sections[0]

    # A4 page with margins matching reference layout
    section.page_width    = Cm(21.0)
    section.page_height   = Cm(29.7)
    section.top_margin    = Cm(1.5)
    section.bottom_margin = Cm(1.5)
    section.left_margin   = Cm(1.5)
    section.right_margin  = Cm(1.5)

    if doc_type == "id":
        w_cm = ID_RENDER_WIDTH_MM / 10.0
        add_image_at_position(doc, front_image, w_cm)
        doc.add_paragraph("")
        add_image_at_position(doc, back_image, w_cm)
    elif doc_type == "passport":
        w_cm = PASSPORT_RENDER_WIDTH_MM / 10.0
        add_image_at_position(doc, front_image, w_cm)
    else:
        raise ValueError(f"Unsupported document type: {doc_type}")

    doc.save(docx_path)


# ── PDF helpers ───────────────────────────────────────────────────────────────

def create_pdf(
    pdf_path: Path,
    doc_type: str,
    front_image: Path,
    back_image: Optional[Path],
):
    """
    Place card images on A4 at exact positions measured from reference PDFs.
    """
    page_w, page_h = A4   # 595.3 x 841.9 pt
    c = rl_canvas.Canvas(str(pdf_path), pagesize=A4)

    def draw(img_path: Path, x_mm: float, y_mm: float, w_mm: float):
        pil_img = Image.open(img_path)
        # Actual image height in mm at DPI
        h_mm = pil_img.height / (DPI / 25.4)
        # ReportLab y=0 is bottom of page
        y_pt = page_h - mm_to_pt(y_mm) - mm_to_pt(h_mm)
        reader = pil_to_image_reader(pil_img)
        c.drawImage(
            reader,
            mm_to_pt(x_mm), y_pt,
            width=mm_to_pt(w_mm),
            height=mm_to_pt(h_mm),
            mask="auto",
        )

    if doc_type == "id":
        draw(front_image, ID_FRONT_X_MM, ID_FRONT_Y_MM, ID_RENDER_WIDTH_MM)
        if back_image:
            draw(back_image, ID_FRONT_X_MM, ID_BACK_Y_MM, ID_RENDER_WIDTH_MM)
    elif doc_type == "passport":
        draw(front_image, PASSPORT_X_MM, PASSPORT_Y_MM, PASSPORT_RENDER_WIDTH_MM)
    else:
        raise ValueError(f"Unsupported document type: {doc_type}")

    c.showPage()
    c.save()


# ── Upload / base64 utils ─────────────────────────────────────────────────────

async def save_upload(upload_file: UploadFile, destination: Path):
    destination.write_bytes(await upload_file.read())

def file_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/process-document")
async def process_document(
    first_name:       str                  = Form(...),
    last_name:        str                  = Form(...),
    doc_type:         str                  = Form(...),
    output_base_name: str                  = Form(...),
    front_image:      UploadFile           = File(...),
    back_image:       Optional[UploadFile] = File(None),
    front_corners:    Optional[str]        = Form(None),
    back_corners:     Optional[str]        = Form(None),
    x_api_key:        Optional[str]        = Header(None),
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    doc_type = str(doc_type or "").lower().strip()
    if doc_type not in ["id", "passport"]:
        raise HTTPException(status_code=400, detail="doc_type must be 'id' or 'passport'")
    if doc_type == "id" and back_image is None:
        raise HTTPException(status_code=400, detail="Back image is required for ID documents")

    output_base_name = safe_file_part(output_base_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp         = Path(tmpdir)
        front_input = tmp / "front_input"
        back_input  = tmp / "back_input"

        await save_upload(front_image, front_input)
        if back_image:
            await save_upload(back_image, back_input)

        processed_info = []

        if doc_type == "id":
            front_processed = tmp / "front_processed.png"
            back_processed  = tmp / "back_processed.png"
            processed_info.append(
                process_image(front_input, front_processed, doc_type="id", corners_json=front_corners)
            )
            processed_info.append(
                process_image(back_input, back_processed, doc_type="id", corners_json=back_corners)
            )
        else:
            front_processed = tmp / "passport_processed.png"
            back_processed  = None
            processed_info.append(
                process_image(front_input, front_processed, doc_type="passport", corners_json=front_corners)
            )

        docx_filename = f"{output_base_name}.docx"
        pdf_filename  = f"{output_base_name}.pdf"
        docx_path     = tmp / docx_filename
        pdf_path      = tmp / pdf_filename

        create_docx(
            docx_path=docx_path, doc_type=doc_type,
            front_image=front_processed, back_image=back_processed,
        )
        create_pdf(
            pdf_path=pdf_path, doc_type=doc_type,
            front_image=front_processed, back_image=back_processed,
        )

        return {
            "status":           "success",
            "doc_type":         doc_type,
            "first_name":       first_name,
            "last_name":        last_name,
            "output_base_name": output_base_name,
            "docx_filename":    docx_filename,
            "pdf_filename":     pdf_filename,
            "docx_base64":      file_to_base64(docx_path),
            "pdf_base64":       file_to_base64(pdf_path),
            "processed_images": processed_info,
        }


@app.post("/process-document-test")
async def process_document_test(
    first_name:       str                  = Form(...),
    last_name:        str                  = Form(...),
    doc_type:         str                  = Form(...),
    output_base_name: str                  = Form(...),
    front_image:      UploadFile           = File(...),
    back_image:       Optional[UploadFile] = File(None),
    front_corners:    Optional[str]        = Form(None),
    back_corners:     Optional[str]        = Form(None),
    x_api_key:        Optional[str]        = Header(None),
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if doc_type == "id" and back_image is None:
        raise HTTPException(status_code=400, detail="Back image is required for ID documents")

    return {
        "status":           "received",
        "first_name":       first_name,
        "last_name":        last_name,
        "doc_type":         doc_type,
        "output_base_name": output_base_name,
        "front_filename":   front_image.filename,
        "back_filename":    back_image.filename if back_image else None,
        "front_corners":    front_corners,
        "back_corners":     back_corners,
    }


VERSION = "2025-06-08-v13"

@app.get("/version")
def version():
    return {"version": VERSION}
