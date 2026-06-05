from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from typing import Optional
from pathlib import Path
import base64
import os
import re
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ExifTags

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader

import io


app = FastAPI()

API_KEY = os.getenv("API_KEY", "")

# ── Physical document dimensions ──────────────────────────────────────────────

ID_CARD_WIDTH_MM  = 85.60
ID_CARD_HEIGHT_MM = 53.98
ID_CARD_ASPECT    = ID_CARD_WIDTH_MM / ID_CARD_HEIGHT_MM   # ≈ 1.5857

# The raw scan is a landscape two-page spread (176 × 125 mm).
# We rotate it 90° CCW so text reads left-to-right and the image
# sits portrait (tall) on the A4 page: width=125 mm, height=176 mm.
PASSPORT_SCAN_WIDTH_MM  = 176.0
PASSPORT_SCAN_HEIGHT_MM = 125.0
PASSPORT_SCAN_ASPECT    = PASSPORT_SCAN_WIDTH_MM / PASSPORT_SCAN_HEIGHT_MM

# After rotation the canvas dimensions are swapped.
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

SHADOW_OFFSET_MM  = 2.0    # how far the shadow is offset (right + down)
SHADOW_BLUR_MM    = 3.0    # blur radius of the shadow
SHADOW_OPACITY    = 160    # 0-255; 160 ≈ 63% opacity (matches Word "Centre Shadow Rectangle")
SHADOW_COLOR      = (80, 80, 80)  # dark grey, not pure black

# ── Detection tunables ────────────────────────────────────────────────────────

ASPECT_TOLERANCE    = 0.30
MIN_CARD_AREA_RATIO = 0.10


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
# FIX: OpenCV's imdecode ignores EXIF orientation. We use Pillow first so that
# phone photos taken in landscape (stored as portrait + EXIF rotate) are
# correctly oriented before any processing begins.

def read_image_exif_aware(path: Path) -> np.ndarray:
    """
    Read an image file respecting EXIF orientation metadata.
    Returns a BGR ndarray (OpenCV convention).
    """
    pil_img = Image.open(path)

    # Apply EXIF orientation if present
    try:
        exif = pil_img._getexif()
        if exif:
            orientation_key = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
            )
            if orientation_key and orientation_key in exif:
                orientation = exif[orientation_key]
                rotations = {
                    3: 180,
                    6: -90,   # 270 CCW == 90 CW
                    8:  90,
                }
                flip_horizontal = {2, 4, 5, 7}
                if orientation in rotations:
                    pil_img = pil_img.rotate(rotations[orientation], expand=True)
                elif orientation in flip_horizontal:
                    pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                    if orientation in {5, 7}:
                        pil_img = pil_img.rotate(90 if orientation == 5 else -90, expand=True)
    except Exception:
        pass  # No EXIF or unreadable — proceed with raw pixels

    pil_img = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

def save_png(path: Path, image_bgr: np.ndarray):
    pil_img = bgr_to_pil(image_bgr)
    pil_img.save(path, dpi=(DPI, DPI))


# ── Perspective correction ────────────────────────────────────────────────────

def resize_for_detection(image: np.ndarray, max_dim: int = 1500):
    h, w    = image.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return image.copy(), 1.0
    scale   = max_dim / float(largest)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    return resized, scale

def order_points(pts: np.ndarray) -> np.ndarray:
    rect    = np.zeros((4, 2), dtype="float32")
    s       = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff    = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    tl, tr, br, bl = rect

    width_a   = np.linalg.norm(br - bl)
    width_b   = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a   = np.linalg.norm(tr - br)
    height_b   = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    if max_width < 50 or max_height < 50:
        return image

    dst = np.array([
        [0,             0             ],
        [max_width - 1, 0             ],
        [max_width - 1, max_height - 1],
        [0,             max_height - 1],
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(
        image, matrix, (max_width, max_height),
        borderValue=(255, 255, 255),
    )

def build_card_mask(gray: np.ndarray) -> np.ndarray:
    """
    Morphological mask that merges the card body into one solid region
    so we can find its bounding quad reliably.
    """
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

    close_size = max(gray.shape) // 20
    if close_size % 2 == 0:
        close_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    flood = closed.copy()
    fmask = np.zeros((closed.shape[0] + 2, closed.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, fmask, (0, 0), 255)
    filled = cv2.bitwise_not(flood)
    return cv2.bitwise_or(closed, filled)

def contour_to_quad(contour: np.ndarray) -> np.ndarray:
    hull      = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for eps in [0.02, 0.03, 0.05, 0.08]:
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")
    rect = cv2.minAreaRect(hull)
    return cv2.boxPoints(rect).astype("float32")

def find_card_contour(image: np.ndarray, target_aspect: float) -> Optional[np.ndarray]:
    resized, scale = resize_for_detection(image, max_dim=1500)
    img_area       = resized.shape[0] * resized.shape[1]

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    solid_mask  = build_card_mask(gray)
    contours, _ = cv2.findContours(solid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_pts  = None
    best_area = 0

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = cv2.contourArea(contour)
        if area < img_area * MIN_CARD_AREA_RATIO:
            break

        pts  = contour_to_quad(contour)
        rect = order_points(pts)
        tl, tr, br, bl = rect

        card_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
        card_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0

        if card_h < 1:
            continue

        aspect = card_w / card_h
        if (abs(aspect - target_aspect)         > ASPECT_TOLERANCE and
                abs((1.0 / aspect) - target_aspect) > ASPECT_TOLERANCE):
            continue

        if area > best_area:
            best_area = area
            best_pts  = pts / scale

    return best_pts


# ── Hard rectangular extraction (replaces rembg) ─────────────────────────────
# FIX: Instead of using rembg (which produces soft edges and rounded corners),
# we rely entirely on the perspective-corrected rectangle. The output is already
# a clean rectangle after four_point_transform — we just need to place it on a
# white canvas at the exact target physical size.
#
# rembg is removed entirely. Sharp corners are guaranteed because the card
# boundary IS the rectangle produced by the perspective transform.

def force_landscape(image_bgr: np.ndarray) -> np.ndarray:
    """Ensure the card image is landscape (wider than tall)."""
    h, w = image_bgr.shape[:2]
    if h > w:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    return image_bgr

def force_portrait(image_bgr: np.ndarray) -> np.ndarray:
    """Ensure the passport spread is portrait (taller than wide)."""
    h, w = image_bgr.shape[:2]
    if w > h:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image_bgr

def render_card_exact(
    card_bgr: np.ndarray,
    target_width_mm: float,
    target_height_mm: float,
    doc_type: str = "id",
) -> np.ndarray:
    """
    FIX: Force the card to exact physical pixel dimensions.
    Previous version used min(scale) fitting which caused size inconsistency
    between front and back. Now we resize directly to target pixels.
    Orientation is corrected first (landscape for ID, portrait for passport).
    """
    if doc_type == "passport":
        card_bgr = force_portrait(card_bgr)
    else:
        card_bgr = force_landscape(card_bgr)

    target_w_px = mm_to_px(target_width_mm)
    target_h_px = mm_to_px(target_height_mm)

    # Direct resize to exact physical dimensions — no aspect-ratio fitting.
    # The perspective transform already produced the correct rectangle;
    # any remaining distortion is negligible and legally acceptable.
    resized = cv2.resize(card_bgr, (target_w_px, target_h_px), interpolation=cv2.INTER_LANCZOS4)
    return resized


# ── Drop shadow compositor ────────────────────────────────────────────────────
# Replicates the Word "Centre Shadow Rectangle" effect as a PIL RGBA image.

def add_drop_shadow(card_bgr: np.ndarray) -> Image.Image:
    """
    Composite the card image onto a white RGBA canvas with a soft drop shadow,
    replicating the Word 'Centre Shadow Rectangle' style.

    Returns a PIL RGB image (white background, card + shadow on top).
    """
    card_pil = bgr_to_pil(card_bgr)
    cw, ch   = card_pil.size

    shadow_offset_px = mm_to_px(SHADOW_OFFSET_MM)
    shadow_blur_px   = mm_to_px(SHADOW_BLUR_MM)

    # Canvas is larger than card to accommodate shadow
    pad        = shadow_blur_px * 2 + shadow_offset_px
    canvas_w   = cw + pad
    canvas_h   = ch + pad
    card_x     = pad // 2 - shadow_offset_px // 2
    card_y     = pad // 2 - shadow_offset_px // 2
    shadow_x   = card_x + shadow_offset_px
    shadow_y   = card_y + shadow_offset_px

    # White base canvas (RGB)
    base = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    # Shadow layer: solid dark rectangle, same size as card, then blurred
    shadow_layer = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
    shadow_rect  = Image.new(
        "RGBA", (cw, ch),
        (SHADOW_COLOR[0], SHADOW_COLOR[1], SHADOW_COLOR[2], SHADOW_OPACITY)
    )
    shadow_layer.paste(shadow_rect, (shadow_x, shadow_y))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_blur_px))

    # Composite: white base → shadow → card
    base_rgba = base.convert("RGBA")
    base_rgba = Image.alpha_composite(base_rgba, shadow_layer)
    base_rgba.paste(card_pil, (card_x, card_y))

    return base_rgba.convert("RGB")


# ── Full extraction pipeline ──────────────────────────────────────────────────

def process_image(
    input_path: Path,
    output_path: Path,
    target_width_mm: float,
    target_height_mm: float,
    doc_type: str = "id",
) -> dict:
    """
    Full pipeline:
    1. EXIF-aware read (handles phone photo orientations)
    2. Perspective correction (find card quad + four_point_transform)
    3. Hard rectangular crop to exact physical pixels (sharp corners, consistent size)
    4. Add drop shadow
    5. Save PNG at 300 DPI
    """
    image = read_image_exif_aware(input_path)

    # Detection aspect: for passport use the raw landscape scan ratio
    if doc_type == "passport":
        detection_aspect = PASSPORT_SCAN_WIDTH_MM / PASSPORT_SCAN_HEIGHT_MM
    else:
        detection_aspect = target_width_mm / target_height_mm

    pts = find_card_contour(image, detection_aspect)

    if pts is not None:
        corrected = four_point_transform(image, pts)
        method    = "perspective_corrected"
    else:
        corrected = image.copy()
        method    = "fallback_no_perspective"

    # Exact physical size, sharp corners
    card_exact = render_card_exact(corrected, target_width_mm, target_height_mm, doc_type=doc_type)

    # Drop shadow composite
    final_pil = add_drop_shadow(card_exact)
    final_pil.save(output_path, dpi=(DPI, DPI))

    return {
        "input_file":  input_path.name,
        "output_file": output_path.name,
        "method":      method,
    }


# ── DOCX helpers ──────────────────────────────────────────────────────────────

def _docx_image_width_with_shadow(base_mm: float) -> float:
    """
    The saved PNG includes shadow padding. We pass the full image width
    to python-docx so the shadow is visible. The card itself is base_mm wide
    inside the PNG; shadow adds ~2× SHADOW_BLUR_MM + SHADOW_OFFSET_MM on each side.
    We keep it simple: let python-docx render the PNG at its natural DPI size.
    """
    shadow_pad_mm = (mm_to_px(SHADOW_BLUR_MM) * 2 + mm_to_px(SHADOW_OFFSET_MM)) / (DPI / 25.4)
    return base_mm + shadow_pad_mm


def add_image_paragraph(
    doc: Document,
    image_path: Path,
    width_mm: float,
):
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
        w_mm = _docx_image_width_with_shadow(ID_CARD_WIDTH_MM)
        add_image_paragraph(doc, front_image, w_mm)
        doc.add_paragraph("")
        add_image_paragraph(doc, back_image, w_mm)
    elif doc_type == "passport":
        w_mm = _docx_image_width_with_shadow(PASSPORT_WIDTH_MM)
        add_image_paragraph(doc, front_image, w_mm)
    else:
        raise ValueError(f"Unsupported document type: {doc_type}")

    doc.save(docx_path)


# ── PDF helpers ───────────────────────────────────────────────────────────────

def _pil_to_image_reader(pil_img: Image.Image) -> ImageReader:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)


def create_pdf(
    pdf_path: Path,
    doc_type: str,
    front_image: Path,
    back_image: Optional[Path],
):
    """
    Render card PNGs (which already contain the drop shadow) into A4 PDF.
    The shadow padding is included in the PNG dimensions, so we read the
    actual pixel size and scale to physical mm at 300 DPI.
    """
    page_w, page_h = A4
    c      = rl_canvas.Canvas(str(pdf_path), pagesize=A4)
    x      = PAGE_IMAGE_LEFT_MARGIN_MM * mm

    def draw_image_mm(img_path: Path, y_top_mm: float):
        """Draw image at x, positioned so its top is at y_top_mm from page top."""
        pil_img = Image.open(img_path)
        img_w_mm = pil_img.width  / (DPI / 25.4)
        img_h_mm = pil_img.height / (DPI / 25.4)
        y_pt = page_h - y_top_mm * mm - img_h_mm * mm
        c.drawImage(
            ImageReader(str(img_path)),
            x, y_pt,
            width=img_w_mm * mm,
            height=img_h_mm * mm,
        )
        return img_h_mm

    if doc_type == "id":
        y_cursor = DOCX_PAGE_TOP_MARGIN_MM + PDF_CARD_MARGIN_MM
        h = draw_image_mm(front_image, y_cursor)
        y_cursor += h + PDF_CARD_MARGIN_MM
        draw_image_mm(back_image, y_cursor)
    elif doc_type == "passport":
        y_cursor = DOCX_PAGE_TOP_MARGIN_MM + PDF_CARD_MARGIN_MM
        draw_image_mm(front_image, y_cursor)
    else:
        raise ValueError(f"Unsupported document type: {doc_type}")

    c.showPage()
    c.save()


# ── Upload / base64 utils ─────────────────────────────────────────────────────

async def save_upload(upload_file: UploadFile, destination: Path):
    content = await upload_file.read()
    destination.write_bytes(content)

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


VERSION = "2025-06-05-v2"

@app.get("/version")
def version():
    return {"version": VERSION}
