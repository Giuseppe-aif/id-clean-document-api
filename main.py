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
# We rotate it 90° CCW → portrait on A4: width=125 mm, height=176 mm.
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

SHADOW_OFFSET_MM = 2.0    # right + down offset in mm
SHADOW_BLUR_MM   = 3.0    # Gaussian blur radius in mm
SHADOW_OPACITY   = 160    # 0–255; 160 ≈ 63 % (matches Word "Centre Shadow Rectangle")
SHADOW_COLOR     = (80, 80, 80)

# ── Detection tunables ────────────────────────────────────────────────────────

ASPECT_TOLERANCE    = 0.35   # widened slightly to handle more real-world photos
MIN_CARD_AREA_RATIO = 0.05   # lowered: light-coloured cards with small dark area still qualify


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
    Read an image respecting EXIF orientation so phone photos arrive
    correctly oriented regardless of how the camera stored them.
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


# ── Perspective correction ────────────────────────────────────────────────────

def resize_for_detection(image: np.ndarray, max_dim: int = 1500):
    h, w = image.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return image.copy(), 1.0
    scale = max_dim / float(largest)
    return cv2.resize(image, (int(w * scale), int(h * scale))), scale

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
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

def build_card_mask_morphological(gray: np.ndarray) -> np.ndarray:
    """
    Primary detection strategy: threshold + morphological closing + flood fill.
    Works well when the card has clear contrast against the background.
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

def build_card_mask_edges(gray: np.ndarray) -> np.ndarray:
    """
    Fallback detection strategy: Canny edge detection + dilation + flood fill.
    Works better when the card is light-coloured and blends with the background
    (e.g. white/grey card on white table) — the morphological threshold fails
    in that case because there is not enough pixel-level contrast, but the card
    edges are still detectable via gradient.
    """
    # CLAHE boosts local contrast so even faint edges become detectable
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100)

    # Dilate edges to close gaps, then flood-fill to get a solid mask
    dil_size = max(gray.shape) // 40
    if dil_size % 2 == 0:
        dil_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dil_size, dil_size))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    flood = dilated.copy()
    fmask = np.zeros((dilated.shape[0] + 2, dilated.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, fmask, (0, 0), 255)
    filled = cv2.bitwise_not(flood)
    return cv2.bitwise_or(dilated, filled)

def contour_to_quad(contour: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for eps in [0.02, 0.03, 0.05, 0.08]:
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")
    rect = cv2.minAreaRect(hull)
    return cv2.boxPoints(rect).astype("float32")

def _search_contours(
    mask: np.ndarray,
    img_area: int,
    target_aspect: float,
    scale: float,
) -> tuple[Optional[np.ndarray], float]:
    """
    Search a binary mask for the best card-shaped contour.
    Returns (pts_in_original_coords, best_area) or (None, 0).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_pts  = None
    best_area = 0.0

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

    return best_pts, best_area

def _is_plausibly_corrected(corrected: np.ndarray, target_aspect: float) -> bool:
    """
    After four_point_transform, check that the resulting image aspect ratio
    is close to what we expect. If not, the quad detection found the wrong shape.
    """
    h, w = corrected.shape[:2]
    if h < 1:
        return False
    aspect = w / h
    return (abs(aspect - target_aspect)         <= ASPECT_TOLERANCE or
            abs((1.0 / aspect) - target_aspect) <= ASPECT_TOLERANCE)

def find_card_contour(image: np.ndarray, target_aspect: float) -> Optional[np.ndarray]:
    """
    Two-pass card detector:

    Pass 1 — morphological threshold mask (fast, works on most photos)
    Pass 2 — CLAHE + Canny edge mask (robust fallback for light-coloured cards
              on light backgrounds, e.g. the back of many European ID cards)

    If Pass 1 finds a plausible quad AND the perspective-corrected result has
    a sensible aspect ratio, it is used.  Otherwise Pass 2 is tried.
    If both fail, None is returned and the pipeline falls back to no correction.
    """
    resized, scale = resize_for_detection(image, max_dim=1500)
    img_area       = resized.shape[0] * resized.shape[1]
    gray           = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray_blurred   = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── Pass 1: morphological ──────────────────────────────────────────────
    mask1 = build_card_mask_morphological(gray_blurred)
    pts1, area1 = _search_contours(mask1, img_area, target_aspect, scale)

    if pts1 is not None:
        candidate = four_point_transform(image, pts1)
        if _is_plausibly_corrected(candidate, target_aspect):
            return pts1
        # Found something but aspect ratio wrong — don't give up yet

    # ── Pass 2: edge-based ─────────────────────────────────────────────────
    mask2 = build_card_mask_edges(gray_blurred)
    pts2, area2 = _search_contours(mask2, img_area, target_aspect, scale)

    if pts2 is not None:
        candidate2 = four_point_transform(image, pts2)
        if _is_plausibly_corrected(candidate2, target_aspect):
            return pts2

    # ── Last resort: return whichever pass had the larger hit ─────────────
    if pts1 is not None and pts2 is None:
        return pts1
    if pts2 is not None and pts1 is None:
        return pts2
    if pts1 is not None and pts2 is not None:
        return pts1 if area1 >= area2 else pts2

    return None


# ── Orientation helpers ───────────────────────────────────────────────────────

def force_landscape(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if h > w:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    return image_bgr

def force_portrait(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if w > h:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image_bgr


# ── Exact-size renderer ───────────────────────────────────────────────────────

def render_card_exact(
    card_bgr: np.ndarray,
    target_width_mm: float,
    target_height_mm: float,
    doc_type: str = "id",
) -> np.ndarray:
    """
    Resize the perspective-corrected card to exact physical pixel dimensions.
    Orientation is normalised first (landscape for ID, portrait for passport).
    Both front and back will always produce identical pixel dimensions.
    """
    if doc_type == "passport":
        card_bgr = force_portrait(card_bgr)
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
    Returns a PIL RGB image ready to save as PNG.
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

    # White base
    base = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    # Shadow: solid rect → blur
    shadow_layer = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
    shadow_rect  = Image.new(
        "RGBA", (cw, ch),
        (SHADOW_COLOR[0], SHADOW_COLOR[1], SHADOW_COLOR[2], SHADOW_OPACITY),
    )
    shadow_layer.paste(shadow_rect, (shadow_x, shadow_y))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_blur_px))

    # Composite: base → shadow → card
    base = Image.alpha_composite(base, shadow_layer)
    base.paste(card_pil, (card_x, card_y))

    return base.convert("RGB")


# ── PIL → ReportLab ImageReader (preserves full quality) ─────────────────────

def pil_to_image_reader(pil_img: Image.Image) -> ImageReader:
    """
    Convert a PIL image to a ReportLab ImageReader via an in-memory PNG buffer.
    Using this instead of ImageReader(filepath) ensures the shadow (which is
    stored in PNG metadata) is rendered correctly at the right DPI in the PDF.
    """
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
    1. EXIF-aware read
    2. Two-pass perspective detection + correction
    3. Exact physical resize (sharp corners, consistent size)
    4. Drop shadow composite
    5. Save PNG at 300 DPI
    """
    image = read_image_exif_aware(input_path)

    detection_aspect = (
        PASSPORT_SCAN_WIDTH_MM / PASSPORT_SCAN_HEIGHT_MM
        if doc_type == "passport"
        else target_width_mm / target_height_mm
    )

    pts = find_card_contour(image, detection_aspect)

    if pts is not None:
        corrected = four_point_transform(image, pts)
        method    = "perspective_corrected"
    else:
        corrected = image.copy()
        method    = "fallback_no_perspective"

    card_exact = render_card_exact(corrected, target_width_mm, target_height_mm, doc_type=doc_type)
    final_pil  = add_drop_shadow(card_exact)
    final_pil.save(output_path, dpi=(DPI, DPI))

    return {
        "input_file":  input_path.name,
        "output_file": output_path.name,
        "method":      method,
    }


# ── DOCX helpers ──────────────────────────────────────────────────────────────

def _image_display_width_mm(base_mm: float) -> float:
    """
    The PNG includes shadow padding around the card.
    Compute the total width in mm so python-docx renders it at natural size.
    """
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
    """
    Place card PNGs (shadow already baked in) onto an A4 page.
    PNG pixel dimensions are read and converted to mm at 300 DPI so the card
    renders at exactly the correct physical size.
    Images are loaded via pil_to_image_reader so shadow quality is preserved.
    """
    page_w, page_h = A4
    c = rl_canvas.Canvas(str(pdf_path), pagesize=A4)
    x = PAGE_IMAGE_LEFT_MARGIN_MM * mm

    def draw(img_path: Path, y_top_mm: float) -> float:
        """Draw one card image; return its height in mm."""
        pil_img  = Image.open(img_path)
        w_mm     = pil_img.width  / (DPI / 25.4)
        h_mm     = pil_img.height / (DPI / 25.4)
        y_pt     = page_h - y_top_mm * mm - h_mm * mm
        reader   = pil_to_image_reader(pil_img)   # FIX: was ImageReader(filepath) — shadow lost
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


VERSION = "2025-06-05-v3"

@app.get("/version")
def version():
    return {"version": VERSION}
