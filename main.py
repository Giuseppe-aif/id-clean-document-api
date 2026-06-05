from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from typing import Optional
from pathlib import Path
import base64
import os
import re
import tempfile
import io

import cv2
import numpy as np
from PIL import Image, ImageFilter, ExifTags
from rembg import remove, new_session

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader


app = FastAPI()

API_KEY = os.getenv("API_KEY", "")

# ── Physical document dimensions ──────────────────────────────────────────────

ID_CARD_WIDTH_MM  = 85.60
ID_CARD_HEIGHT_MM = 53.98

# Passport: raw scan is a landscape two-page spread (176 × 125 mm).
# After 90° CCW rotation → portrait on A4: width=125 mm, height=176 mm.
PASSPORT_SCAN_WIDTH_MM  = 176.0
PASSPORT_SCAN_HEIGHT_MM = 125.0

PASSPORT_WIDTH_MM  = PASSPORT_SCAN_HEIGHT_MM   # 125
PASSPORT_HEIGHT_MM = PASSPORT_SCAN_WIDTH_MM    # 176

DPI = 300

# ── Page / layout constants ───────────────────────────────────────────────────

DOCX_PAGE_LEFT_MARGIN_MM   = 20.0
DOCX_PAGE_RIGHT_MARGIN_MM  = 20.0
DOCX_PAGE_TOP_MARGIN_MM    = 20.0
DOCX_PAGE_BOTTOM_MARGIN_MM = 20.0

PAGE_IMAGE_LEFT_MARGIN_MM = 40.0
PDF_CARD_MARGIN_MM        = 8.0

# ── Shadow settings ───────────────────────────────────────────────────────────

SHADOW_OFFSET_MM = 2.0
SHADOW_BLUR_MM   = 3.0
SHADOW_OPACITY   = 160
SHADOW_COLOR     = (80, 80, 80)

# ── rembg / isolation tunables ────────────────────────────────────────────────

# Alpha threshold for non-transparent pixel detection
ALPHA_THRESHOLD = 10

# Alpha matting parameters (from rembg official docs)
# These reduce the white halo around the card edges
ALPHA_MATTING_FG_THRESHOLD  = 270
ALPHA_MATTING_BG_THRESHOLD  = 20
ALPHA_MATTING_ERODE_SIZE    = 11

# Perspective correction from alpha mask
# Aspect ratio tolerance when validating detected quad
ASPECT_TOLERANCE = 0.35

# Padding around the final crop
TIGHT_CROP_PADDING_PX = 4


# ── rembg session (loaded once at startup) ────────────────────────────────────

_REMBG_SESSION = None

def get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        _REMBG_SESSION = new_session("isnet-general-use")
    return _REMBG_SESSION


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


# ── EXIF-aware image reading ──────────────────────────────────────────────────

def read_image_exif_aware(path: Path) -> np.ndarray:
    """
    Read an image respecting EXIF orientation so phone photos taken sideways
    arrive correctly oriented before any processing begins.
    Returns a BGR ndarray (OpenCV convention).
    """
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
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


# ── Perspective correction helpers ────────────────────────────────────────────

def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Warp image to a flat rectangle defined by 4 corner points."""
    rect = order_points(pts)
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
        image, matrix, (max_width, max_height), borderValue=(255, 255, 255)
    )

def find_quad_from_alpha(alpha: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the 4-corner quad of the card from the rembg alpha mask.

    Using the alpha mask (not the original image) is much more reliable:
    rembg has already cleanly separated the card from any background, so
    the contour we find here is the actual card outline — no background
    confusion, no contrast dependency.

    Returns a float32 (4,2) array of corner points, or None if no clean
    quad is found.
    """
    # Threshold the alpha mask to binary
    _, binary = cv2.threshold(alpha, ALPHA_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Light morphological close to fill tiny gaps at card edges
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Take the largest contour — that's the card
    contour  = max(contours, key=cv2.contourArea)
    hull     = cv2.convexHull(contour)
    perim    = cv2.arcLength(hull, True)

    # Try progressively looser approximations until we get 4 points
    for eps in [0.02, 0.03, 0.05, 0.08, 0.10]:
        approx = cv2.approxPolyDP(hull, eps * perim, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")

    # Fallback: use minimum area rectangle
    rect = cv2.minAreaRect(hull)
    return cv2.boxPoints(rect).astype("float32")


# ── Core isolation pipeline ───────────────────────────────────────────────────

def isolate_and_correct(image_bgr: np.ndarray, target_aspect: float) -> tuple[np.ndarray, str]:
    """
    Full isolation pipeline:

    1. rembg (ISNet) removes the background with alpha matting for clean edges.
       Alpha matting is documented in rembg official docs and specifically
       reduces the white halo artifact around object edges.

    2. Find the card's 4 corners from the clean alpha mask.
       Using the alpha mask instead of the original image is far more reliable —
       rembg has already solved the background separation problem, so the
       contour we find is the actual card outline with no background interference.

    3. Apply perspective correction (four_point_transform) to flatten any angle.
       This corrects photos taken at slight angles — the card need not be
       photographed perfectly perpendicular to work correctly.

    4. Flatten onto white, resize to exact physical dimensions.

    Returns (card_bgr, method_string).
    """
    pil_rgb = bgr_to_pil(image_bgr)

    # Step 1 — rembg with alpha matting (reduces halo)
    try:
        rgba_out = remove(
            pil_rgb,
            session=get_rembg_session(),
            alpha_matting=True,
            alpha_matting_foreground_threshold=ALPHA_MATTING_FG_THRESHOLD,
            alpha_matting_background_threshold=ALPHA_MATTING_BG_THRESHOLD,
            alpha_matting_erode_size=ALPHA_MATTING_ERODE_SIZE,
            post_process_mask=True,
        )
    except Exception:
        # Alpha matting can fail on some images — fall back to standard removal
        rgba_out = remove(pil_rgb, session=get_rembg_session(), post_process_mask=True)

    alpha = np.array(rgba_out.split()[3])

    # Step 2 — find quad from alpha mask
    quad = find_quad_from_alpha(alpha)

    if quad is not None:
        # Validate aspect ratio of detected quad
        rect      = order_points(quad)
        tl, tr, br, bl = rect
        card_w    = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
        card_h    = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
        if card_h > 0:
            aspect = card_w / card_h
            aspect_ok = (
                abs(aspect - target_aspect)         <= ASPECT_TOLERANCE or
                abs((1.0 / aspect) - target_aspect) <= ASPECT_TOLERANCE
            )
        else:
            aspect_ok = False

        if aspect_ok:
            # Step 3 — perspective correct using the RGBA image
            rgba_arr      = np.array(rgba_out)
            corrected_arr = cv2.warpPerspective(
                rgba_arr, cv2.getPerspectiveTransform(
                    order_points(quad),
                    np.array([
                        [0, 0],
                        [int(max(np.linalg.norm(rect[1]-rect[0]), np.linalg.norm(rect[2]-rect[3])))-1, 0],
                        [int(max(np.linalg.norm(rect[1]-rect[0]), np.linalg.norm(rect[2]-rect[3])))-1,
                         int(max(np.linalg.norm(rect[3]-rect[0]), np.linalg.norm(rect[2]-rect[1])))-1],
                        [0, int(max(np.linalg.norm(rect[3]-rect[0]), np.linalg.norm(rect[2]-rect[1])))-1],
                    ], dtype="float32")
                ),
                (
                    int(max(np.linalg.norm(rect[1]-rect[0]), np.linalg.norm(rect[2]-rect[3]))),
                    int(max(np.linalg.norm(rect[3]-rect[0]), np.linalg.norm(rect[2]-rect[1]))),
                ),
                borderValue=(255, 255, 255, 0),
            )
            rgba_corrected = Image.fromarray(corrected_arr, "RGBA")
            method = "rembg_alpha_matting_perspective_corrected"
        else:
            # Quad found but wrong aspect — skip perspective correction
            rgba_corrected = rgba_out
            method = "rembg_alpha_matting_no_perspective"
    else:
        rgba_corrected = rgba_out
        method = "rembg_alpha_matting_no_quad"

    # Step 4 — tight crop from alpha bounding box
    alpha_c = np.array(rgba_corrected.split()[3])
    coords  = np.argwhere(alpha_c > ALPHA_THRESHOLD)

    if coords.size == 0:
        # rembg removed everything — return original
        white = Image.new("RGB", rgba_out.size, (255, 255, 255))
        white.paste(rgba_out, mask=rgba_out.split()[3])
        return cv2.cvtColor(np.array(white), cv2.COLOR_RGB2BGR), "fallback_original"

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    h, w   = alpha_c.shape
    y0 = max(y0 - TIGHT_CROP_PADDING_PX, 0)
    x0 = max(x0 - TIGHT_CROP_PADDING_PX, 0)
    y1 = min(y1 + TIGHT_CROP_PADDING_PX, h - 1)
    x1 = min(x1 + TIGHT_CROP_PADDING_PX, w - 1)

    rgba_cropped = rgba_corrected.crop((x0, y0, x1 + 1, y1 + 1))

    # Flatten onto white
    white = Image.new("RGB", rgba_cropped.size, (255, 255, 255))
    white.paste(rgba_cropped, mask=rgba_cropped.split()[3])

    return cv2.cvtColor(np.array(white), cv2.COLOR_RGB2BGR), method


# ── Orientation helpers ───────────────────────────────────────────────────────

def force_landscape(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if h > w:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    return image_bgr

def force_passport_orientation(image_bgr: np.ndarray) -> np.ndarray:
    """
    Passport spreads must appear portrait (taller than wide) with text
    reading left-to-right. Always rotate 90° CCW from landscape.
    """
    h, w = image_bgr.shape[:2]
    if h > w:
        image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)


# ── Exact-size renderer ───────────────────────────────────────────────────────

def render_card_exact(
    card_bgr: np.ndarray,
    target_width_mm: float,
    target_height_mm: float,
    doc_type: str = "id",
) -> np.ndarray:
    if doc_type == "passport":
        card_bgr = force_passport_orientation(card_bgr)
    else:
        card_bgr = force_landscape(card_bgr)

    target_w_px = mm_to_px(target_width_mm)
    target_h_px = mm_to_px(target_height_mm)
    return cv2.resize(card_bgr, (target_w_px, target_h_px), interpolation=cv2.INTER_LANCZOS4)


# ── Drop shadow compositor ────────────────────────────────────────────────────

def add_drop_shadow(card_bgr: np.ndarray) -> Image.Image:
    """
    Composite the card onto a white canvas with a soft Gaussian drop shadow,
    replicating the Word 'Centre Shadow Rectangle' style.
    """
    card_pil = bgr_to_pil(card_bgr)
    cw, ch   = card_pil.size

    shadow_offset_px = mm_to_px(SHADOW_OFFSET_MM)
    shadow_blur_px   = mm_to_px(SHADOW_BLUR_MM)

    pad      = shadow_blur_px * 2 + shadow_offset_px
    canvas_w = cw + pad
    canvas_h = ch + pad
    card_x   = pad // 2 - shadow_offset_px // 2
    card_y   = pad // 2 - shadow_offset_px // 2
    shadow_x = card_x + shadow_offset_px
    shadow_y = card_y + shadow_offset_px

    base = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    shadow_layer = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
    shadow_rect  = Image.new(
        "RGBA", (cw, ch),
        (SHADOW_COLOR[0], SHADOW_COLOR[1], SHADOW_COLOR[2], SHADOW_OPACITY),
    )
    shadow_layer.paste(shadow_rect, (shadow_x, shadow_y))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_blur_px))

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
    target_width_mm: float,
    target_height_mm: float,
    doc_type: str = "id",
) -> dict:
    """
    Pipeline:
    1. EXIF-aware read           — correct phone photo orientation
    2. rembg + alpha matting     — clean isolation, no halo
    3. Quad detection from alpha — find card corners from clean mask
    4. Perspective correction    — flatten any photo angle
    5. Exact physical resize     — consistent size, correct mm at 300 DPI
    6. Drop shadow composite     — scanned appearance
    7. Save PNG at 300 DPI
    """
    image = read_image_exif_aware(input_path)

    target_aspect = (
        PASSPORT_SCAN_WIDTH_MM / PASSPORT_SCAN_HEIGHT_MM
        if doc_type == "passport"
        else target_width_mm / target_height_mm
    )

    card_bgr, method = isolate_and_correct(image, target_aspect)
    card_exact       = render_card_exact(card_bgr, target_width_mm, target_height_mm, doc_type=doc_type)
    final_pil        = add_drop_shadow(card_exact)
    final_pil.save(output_path, dpi=(DPI, DPI))

    return {
        "input_file":  input_path.name,
        "output_file": output_path.name,
        "method":      method,
    }


# ── DOCX helpers ──────────────────────────────────────────────────────────────

def _image_display_width_mm(base_mm: float) -> float:
    pad_px = mm_to_px(SHADOW_BLUR_MM) * 2 + mm_to_px(SHADOW_OFFSET_MM)
    pad_mm = pad_px / (DPI / 25.4)
    return base_mm + pad_mm


def add_image_paragraph(doc: Document, image_path: Path, width_mm: float):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    extra_indent_mm = max(PAGE_IMAGE_LEFT_MARGIN_MM - DOCX_PAGE_LEFT_MARGIN_MM, 0)
    paragraph.paragraph_format.left_indent = Cm(extra_indent_mm / 10.0)
    run = paragraph.add_run()
    run.add_picture(str(image_path), width=Cm(width_mm / 10.0))


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
    section.top_margin    = Cm(DOCX_PAGE_TOP_MARGIN_MM    / 10.0)
    section.bottom_margin = Cm(DOCX_PAGE_BOTTOM_MARGIN_MM / 10.0)
    section.left_margin   = Cm(DOCX_PAGE_LEFT_MARGIN_MM   / 10.0)
    section.right_margin  = Cm(DOCX_PAGE_RIGHT_MARGIN_MM  / 10.0)

    if doc_type == "id":
        w_mm = _image_display_width_mm(ID_CARD_WIDTH_MM)
        add_image_paragraph(doc, front_image, w_mm)
        doc.add_paragraph("")
        add_image_paragraph(doc, back_image, w_mm)
    elif doc_type == "passport":
        w_mm = _image_display_width_mm(PASSPORT_WIDTH_MM)
        add_image_paragraph(doc, front_image, w_mm)
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
    page_w, page_h = A4
    c = rl_canvas.Canvas(str(pdf_path), pagesize=A4)
    x = PAGE_IMAGE_LEFT_MARGIN_MM * mm

    def draw(img_path: Path, y_top_mm: float) -> float:
        pil_img = Image.open(img_path)
        w_mm    = pil_img.width  / (DPI / 25.4)
        h_mm    = pil_img.height / (DPI / 25.4)
        y_pt    = page_h - y_top_mm * mm - h_mm * mm
        reader  = pil_to_image_reader(pil_img)
        c.drawImage(reader, x, y_pt, width=w_mm * mm, height=h_mm * mm, mask="auto")
        return h_mm

    if doc_type == "id":
        y = DOCX_PAGE_TOP_MARGIN_MM + PDF_CARD_MARGIN_MM
        h = draw(front_image, y)
        draw(back_image, y + h + PDF_CARD_MARGIN_MM)
    elif doc_type == "passport":
        draw(front_image, DOCX_PAGE_TOP_MARGIN_MM + PDF_CARD_MARGIN_MM)
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
                process_image(front_input, front_processed, ID_CARD_WIDTH_MM, ID_CARD_HEIGHT_MM, doc_type="id")
            )
            processed_info.append(
                process_image(back_input, back_processed, ID_CARD_WIDTH_MM, ID_CARD_HEIGHT_MM, doc_type="id")
            )
        else:
            front_processed = tmp / "passport_processed.png"
            back_processed  = None
            processed_info.append(
                process_image(front_input, front_processed, PASSPORT_WIDTH_MM, PASSPORT_HEIGHT_MM, doc_type="passport")
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
    }


VERSION = "2025-06-05-v6"

@app.get("/version")
def version():
    return {"version": VERSION}
