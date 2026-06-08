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

# ── DPI ───────────────────────────────────────────────────────────────────────

DPI = 300

# ── Layout constants (measured from reference PDFs) ───────────────────────────

# ID card render size on page (includes shadow padding)
ID_RENDER_WIDTH_MM  = 107.0
ID_RENDER_HEIGHT_MM =  74.0

# ID card positions on A4 page
ID_FRONT_X_MM =  17.0
ID_FRONT_Y_MM =  15.0
ID_BACK_Y_MM  =  93.0

# Passport render size on page
PASSPORT_RENDER_WIDTH_MM  = 114.0
PASSPORT_RENDER_HEIGHT_MM = 160.0

# Passport position on A4 page
PASSPORT_X_MM =  6.0
PASSPORT_Y_MM = 13.0

# ── Shadow (replicates Word Centre Shadow Rectangle) ──────────────────────────

SHADOW_OFFSET_MM = 2.5
SHADOW_BLUR_MM   = 3.5
SHADOW_OPACITY   = 150
SHADOW_COLOR     = (90, 90, 90)

# ── Passport physical aspect (landscape scan: 176 × 125 mm) ──────────────────

PASSPORT_SCAN_ASPECT = 176.0 / 125.0


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"status": "ok", "message": "ID Clean Document API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}


# ── Utilities ─────────────────────────────────────────────────────────────────

def safe_file_part(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = value.strip("_")
    return value or "clean_document"

def mm_to_px(mm_value: float, dpi: int = DPI) -> int:
    return int(round(mm_value / 25.4 * dpi))

def mm_to_pt(mm_value: float) -> float:
    return mm_value / 25.4 * 72.0


# ── EXIF-aware image read → RGB ndarray ──────────────────────────────────────

def read_image_rgb(path: Path) -> np.ndarray:
    """
    Read image honouring EXIF orientation.
    Returns an RGB uint8 ndarray — colours are correct throughout the pipeline.
    """
    pil_img = Image.open(path)
    try:
        exif = pil_img._getexif()
        if exif:
            key = next((k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None)
            if key and key in exif:
                orientation = exif[key]
                if orientation == 3:
                    pil_img = pil_img.rotate(180, expand=True)
                elif orientation == 6:
                    pil_img = pil_img.rotate(-90, expand=True)
                elif orientation == 8:
                    pil_img = pil_img.rotate(90, expand=True)
    except Exception:
        pass
    return np.array(pil_img.convert("RGB"))


# ── Corner parsing ────────────────────────────────────────────────────────────

def parse_corners(corners_json: Optional[str]) -> Optional[np.ndarray]:
    """
    Parse corners JSON from Claude Vision (via n8n form field).
    Returns float32 (4,2) array [TL, TR, BR, BL] or None.
    """
    if not corners_json:
        return None
    try:
        d = json.loads(corners_json)
        return np.array([
            d["top_left"], d["top_right"],
            d["bottom_right"], d["bottom_left"],
        ], dtype="float32")
    except Exception:
        return None

def parse_rotation(corners_json: Optional[str]) -> int:
    if not corners_json:
        return 0
    try:
        return int(json.loads(corners_json).get("rotation_needed", 0))
    except Exception:
        return 0


# ── Perspective correction ────────────────────────────────────────────────────

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s    = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff    = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image_rgb: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Warp image to flat rectangle. Operates on RGB ndarray, returns RGB ndarray."""
    rect       = order_points(pts)
    tl, tr, br, bl = rect
    max_width  = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    max_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if max_width < 50 or max_height < 50:
        return image_rgb
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(
        image_rgb, M, (max_width, max_height),
        flags=cv2.INTER_LANCZOS4,
        borderValue=(255, 255, 255),
    )

def apply_rotation(image_rgb: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
    elif degrees in (-90, 270):
        return cv2.rotate(image_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif degrees == 180:
        return cv2.rotate(image_rgb, cv2.ROTATE_180)
    return image_rgb

def force_landscape(image_rgb: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if h > w:
        return cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
    return image_rgb

def force_passport_orientation(image_rgb: np.ndarray) -> np.ndarray:
    """Normalise passport to portrait (taller than wide), text reading L→R."""
    h, w = image_rgb.shape[:2]
    if h > w:
        image_rgb = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
    return cv2.rotate(image_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)


# ── Card renderer with shadow ─────────────────────────────────────────────────

def render_card_with_shadow(
    card_rgb: np.ndarray,
    render_w_mm: float,
    render_h_mm: float,
) -> Image.Image:
    """
    Fit card_rgb inside a canvas of (render_w_mm × render_h_mm) at DPI,
    preserving aspect ratio, centred, with a Gaussian drop shadow that
    replicates the Word Centre Shadow Rectangle style.
    Returns a PIL RGB image ready to save as PNG.
    """
    shadow_offset_px = mm_to_px(SHADOW_OFFSET_MM)
    shadow_blur_px   = mm_to_px(SHADOW_BLUR_MM)
    shadow_pad       = shadow_blur_px * 2 + shadow_offset_px

    # Available card area (canvas minus shadow padding)
    card_area_w = mm_to_px(render_w_mm) - shadow_pad
    card_area_h = mm_to_px(render_h_mm) - shadow_pad

    # Scale to fit, preserving aspect ratio
    h, w  = card_rgb.shape[:2]
    scale = min(card_area_w / w, card_area_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized  = cv2.resize(card_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    card_pil = Image.fromarray(resized)   # RGB ndarray → PIL RGB (correct colours)

    # Canvas dimensions
    canvas_w = card_area_w + shadow_pad
    canvas_h = card_area_h + shadow_pad

    # Card top-left (centred inside card area)
    card_x = (card_area_w - new_w) // 2
    card_y = (card_area_h - new_h) // 2

    # Shadow top-left (offset from card)
    shadow_x = card_x + shadow_offset_px
    shadow_y = card_y + shadow_offset_px

    # White base
    base = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    # Shadow layer: solid rect → Gaussian blur
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


# ── PIL → ReportLab reader ────────────────────────────────────────────────────

def pil_to_reader(pil_img: Image.Image) -> ImageReader:
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
    1. Read image as RGB (EXIF-aware)
    2. Perspective-correct using Claude Vision corners
    3. Apply rotation if needed
    4. Normalise orientation
    5. Render with drop shadow at target layout size
    6. Save PNG at 300 DPI
    """
    img = read_image_rgb(input_path)

    # Perspective correction
    pts = parse_corners(corners_json)
    if pts is not None:
        img    = four_point_transform(img, pts)
        method = "claude_corners_corrected"
    else:
        method = "no_corners_fallback"

    # Rotation
    rot = parse_rotation(corners_json)
    if rot != 0:
        img    = apply_rotation(img, rot)
        method += f"_rot{rot}"

    # Orientation normalisation
    if doc_type == "passport":
        img         = force_passport_orientation(img)
        render_w_mm = PASSPORT_RENDER_WIDTH_MM
        render_h_mm = PASSPORT_RENDER_HEIGHT_MM
    else:
        img         = force_landscape(img)
        render_w_mm = ID_RENDER_WIDTH_MM
        render_h_mm = ID_RENDER_HEIGHT_MM

    # Render with shadow
    final_pil = render_card_with_shadow(img, render_w_mm, render_h_mm)
    final_pil.save(output_path, dpi=(DPI, DPI))

    return {
        "input_file":  input_path.name,
        "output_file": output_path.name,
        "method":      method,
    }


# ── DOCX ──────────────────────────────────────────────────────────────────────

def create_docx(
    docx_path: Path,
    doc_type: str,
    front_image: Path,
    back_image: Optional[Path],
):
    doc     = Document()
    section = doc.sections[0]
    section.page_width    = Cm(21.0)
    section.page_height   = Cm(29.7)
    section.top_margin    = Cm(1.5)
    section.bottom_margin = Cm(1.5)
    section.left_margin   = Cm(1.5)
    section.right_margin  = Cm(1.5)

    def add_img(path, w_mm):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after  = Pt(0)
        p.add_run().add_picture(str(path), width=Cm(w_mm / 10.0))

    if doc_type == "id":
        add_img(front_image, ID_RENDER_WIDTH_MM)
        doc.add_paragraph("")
        add_img(back_image,  ID_RENDER_WIDTH_MM)
    elif doc_type == "passport":
        add_img(front_image, PASSPORT_RENDER_WIDTH_MM)
    else:
        raise ValueError(f"Unsupported doc_type: {doc_type}")

    doc.save(docx_path)


# ── PDF ───────────────────────────────────────────────────────────────────────

def create_pdf(
    pdf_path: Path,
    doc_type: str,
    front_image: Path,
    back_image: Optional[Path],
):
    page_w, page_h = A4
    c = rl_canvas.Canvas(str(pdf_path), pagesize=A4)

    def draw(img_path, x_mm, y_mm, w_mm):
        pil_img = Image.open(img_path)
        h_mm    = pil_img.height / (DPI / 25.4)
        y_pt    = page_h - mm_to_pt(y_mm) - mm_to_pt(h_mm)
        c.drawImage(
            pil_to_reader(pil_img),
            mm_to_pt(x_mm), y_pt,
            width=mm_to_pt(w_mm), height=mm_to_pt(h_mm),
            mask="auto",
        )

    if doc_type == "id":
        draw(front_image, ID_FRONT_X_MM, ID_FRONT_Y_MM, ID_RENDER_WIDTH_MM)
        if back_image:
            draw(back_image, ID_FRONT_X_MM, ID_BACK_Y_MM, ID_RENDER_WIDTH_MM)
    elif doc_type == "passport":
        draw(front_image, PASSPORT_X_MM, PASSPORT_Y_MM, PASSPORT_RENDER_WIDTH_MM)
    else:
        raise ValueError(f"Unsupported doc_type: {doc_type}")

    c.showPage()
    c.save()


# ── Upload / base64 ───────────────────────────────────────────────────────────

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
            processed_info.append(process_image(front_input, front_processed, "id", front_corners))
            processed_info.append(process_image(back_input,  back_processed,  "id", back_corners))
        else:
            front_processed = tmp / "passport_processed.png"
            back_processed  = None
            processed_info.append(process_image(front_input, front_processed, "passport", front_corners))

        docx_filename = f"{output_base_name}.docx"
        pdf_filename  = f"{output_base_name}.pdf"
        docx_path     = tmp / docx_filename
        pdf_path      = tmp / pdf_filename

        create_docx(docx_path, doc_type, front_processed, back_processed)
        create_pdf(pdf_path,  doc_type, front_processed, back_processed)

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


VERSION = "2025-06-08-v14"

@app.get("/version")
def version():
    return {"version": VERSION}
