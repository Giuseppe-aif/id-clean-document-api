from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from typing import Optional
from pathlib import Path
import base64
import os
import re
import tempfile

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


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

# ── Detection tunables ────────────────────────────────────────────────────────

ASPECT_TOLERANCE    = 0.30
MIN_CARD_AREA_RATIO = 0.10

# ── rembg session (loaded once at startup) ────────────────────────────────────
# "isnet-general-use" is the most accurate general model.
# Swap for "u2net" if you want faster inference with slightly lower quality.
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

def read_image(path: Path) -> np.ndarray:
    """Read any image OpenCV can handle, regardless of filename extension."""
    data  = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path.name}")
    return image

def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

def pil_to_bgr(image_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

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

    # Fill internal holes by flood-filling from the border.
    flood = closed.copy()
    fmask = np.zeros((closed.shape[0] + 2, closed.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, fmask, (0, 0), 255)
    filled = cv2.bitwise_not(flood)
    return cv2.bitwise_or(closed, filled)

def contour_to_quad(contour: np.ndarray) -> np.ndarray:
    """Reduce a contour to the best 4-point quad via convex hull."""
    hull      = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for eps in [0.02, 0.03, 0.05, 0.08]:
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")
    rect = cv2.minAreaRect(hull)
    return cv2.boxPoints(rect).astype("float32")

def find_card_contour(image: np.ndarray, target_aspect: float) -> Optional[np.ndarray]:
    """
    Detect the card boundary using morphological mask + aspect-ratio filtering.
    Returns 4-point float32 array in original-image coordinates, or None.
    """
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


# ── Background removal (rembg) ────────────────────────────────────────────────

def remove_background(image_bgr: np.ndarray) -> Image.Image:
    """
    Remove the background from a BGR OpenCV image using rembg (deep-learning).
    Returns a PIL RGBA image: document pixels keep their colour, background
    pixels are fully transparent.
    """
    pil_rgb  = bgr_to_pil(image_bgr)
    rgba_out = remove(pil_rgb, session=get_rembg_session())
    return rgba_out  # PIL RGBA

def rgba_on_white(rgba: Image.Image) -> np.ndarray:
    """Flatten a PIL RGBA image onto a pure white BGR canvas."""
    rgb   = Image.new("RGB", rgba.size, (255, 255, 255))
    rgb.paste(rgba, mask=rgba.split()[3])          # alpha-composite
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

def tight_crop_rgba(rgba: Image.Image, padding_px: int = 4) -> Image.Image:
    """
    Crop an RGBA image tightly to all non-transparent pixels, with a small
    padding so no document edge is shaved off.
    """
    alpha  = np.array(rgba.split()[3])
    coords = np.argwhere(alpha > 10)           # non-transparent
    if coords.size == 0:
        return rgba

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    h, w   = alpha.shape

    y0 = max(y0 - padding_px, 0)
    x0 = max(x0 - padding_px, 0)
    y1 = min(y1 + padding_px, h - 1)
    x1 = min(x1 + padding_px, w - 1)

    cropped = rgba.crop((x0, y0, x1 + 1, y1 + 1))
    if cropped.width < 50 or cropped.height < 50:
        return rgba
    return cropped


# ── Full extraction pipeline ──────────────────────────────────────────────────

def extract_card(image: np.ndarray, target_aspect: float):
    """
    1. Perspective-correct the card.
    2. rembg → clean RGBA mask (no background).
    3. Tight-crop to document content only.

    Returns (cropped_rgba: PIL.Image, method: str)
    """
    pts = find_card_contour(image, target_aspect)

    if pts is not None:
        corrected = four_point_transform(image, pts)
        method    = "perspective_corrected"
    else:
        corrected = image.copy()
        method    = "fallback_no_perspective"

    rgba    = remove_background(corrected)
    cropped = tight_crop_rgba(rgba, padding_px=4)

    return cropped, method


# ── Rendering at exact physical size ─────────────────────────────────────────

def rotate_ccw_90(rgba: Image.Image):
    """Rotate 90 degrees counter-clockwise so landscape spread becomes portrait."""
    return rgba.rotate(90, expand=True), "rotated_ccw_90"

def rotate_id_to_landscape(rgba: Image.Image):
    """For ID cards: ensure the card is landscape (wider than tall)."""
    if rgba.height > rgba.width:
        return rgba.rotate(-90, expand=True), "rotated_landscape"
    return rgba, "no_rotation"

def render_card_png(
    rgba: Image.Image,
    target_width_mm: float,
    target_height_mm: float,
    doc_type: str = "id",
) -> tuple[np.ndarray, str]:
    """
    Render the RGBA card onto an exact physical-size white canvas at DPI
    resolution.  Returns a BGR ndarray ready to save.

    ID cards : rotate to landscape (wider than tall) if needed.
    Passport : rotate 90 CCW so the landscape two-page spread becomes
               portrait with text reading left-to-right.
    """
    if doc_type == "passport":
        rgba, rotation_status = rotate_ccw_90(rgba)
    else:
        rgba, rotation_status = rotate_id_to_landscape(rgba)

    target_w_px = mm_to_px(target_width_mm)
    target_h_px = mm_to_px(target_height_mm)

    # Scale to fit inside target canvas, preserving aspect ratio.
    scale = min(target_w_px / rgba.width, target_h_px / rgba.height)
    new_w = int(round(rgba.width  * scale))
    new_h = int(round(rgba.height * scale))

    resized = rgba.resize((new_w, new_h), Image.LANCZOS)

    # White canvas, then alpha-composite the card centred on it.
    canvas_pil = Image.new("RGBA", (target_w_px, target_h_px), (255, 255, 255, 255))
    x = (target_w_px - new_w) // 2
    y = (target_h_px - new_h) // 2
    canvas_pil.paste(resized, (x, y), mask=resized.split()[3])

    final_rgb = canvas_pil.convert("RGB")
    final_bgr = cv2.cvtColor(np.array(final_rgb), cv2.COLOR_RGB2BGR)
    return final_bgr, rotation_status


# ── Top-level image processor ─────────────────────────────────────────────────

def process_image(
    input_path: Path,
    output_path: Path,
    target_width_mm: float,
    target_height_mm: float,
    doc_type: str = "id",
) -> dict:
    image = read_image(input_path)

    # For passport the raw scan is landscape (wider than tall), so use the
    # scan aspect ratio for contour detection, not the post-rotation one.
    if doc_type == "passport":
        detection_aspect = PASSPORT_SCAN_WIDTH_MM / PASSPORT_SCAN_HEIGHT_MM
    else:
        detection_aspect = target_width_mm / target_height_mm

    rgba, method = extract_card(image, detection_aspect)

    final_bgr, rotation_status = render_card_png(
        rgba, target_width_mm, target_height_mm, doc_type=doc_type
    )

    save_png(output_path, final_bgr)

    return {
        "input_file":  input_path.name,
        "output_file": output_path.name,
        "method":      method,
        "rotation":    rotation_status,
    }


# ── DOCX helpers ──────────────────────────────────────────────────────────────

def add_left_positioned_image_to_docx(
    doc: Document,
    image_path: Path,
    width_mm: float,
    height_mm: float,
):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    extra_indent_mm = max(PAGE_IMAGE_LEFT_MARGIN_MM - DOCX_PAGE_LEFT_MARGIN_MM, 0)
    paragraph.paragraph_format.left_indent = Cm(extra_indent_mm / 10.0)
    run = paragraph.add_run()
    run.add_picture(str(image_path), width=Cm(width_mm / 10.0), height=Cm(height_mm / 10.0))


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
        add_left_positioned_image_to_docx(doc, front_image, ID_CARD_WIDTH_MM, ID_CARD_HEIGHT_MM)
        doc.add_paragraph("")
        add_left_positioned_image_to_docx(doc, back_image,  ID_CARD_WIDTH_MM, ID_CARD_HEIGHT_MM)
    elif doc_type == "passport":
        add_left_positioned_image_to_docx(doc, front_image, PASSPORT_WIDTH_MM, PASSPORT_HEIGHT_MM)
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
    c      = canvas.Canvas(str(pdf_path), pagesize=A4)
    margin = PDF_CARD_MARGIN_MM * mm
    x      = PAGE_IMAGE_LEFT_MARGIN_MM * mm

    if doc_type == "id":
        img_w   = ID_CARD_WIDTH_MM  * mm
        img_h   = ID_CARD_HEIGHT_MM * mm
        y_front = page_h - DOCX_PAGE_TOP_MARGIN_MM * mm - margin - img_h
        y_back  = y_front - margin - img_h
        c.drawImage(ImageReader(str(front_image)), x, y_front, width=img_w, height=img_h)
        c.drawImage(ImageReader(str(back_image)),  x, y_back,  width=img_w, height=img_h)
    elif doc_type == "passport":
        img_w = PASSPORT_WIDTH_MM  * mm
        img_h = PASSPORT_HEIGHT_MM * mm
        y     = page_h - DOCX_PAGE_TOP_MARGIN_MM * mm - margin - img_h
        c.drawImage(ImageReader(str(front_image)), x, y, width=img_w, height=img_h)
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
                process_image(front_input, front_processed, ID_CARD_WIDTH_MM,  ID_CARD_HEIGHT_MM,  doc_type="id")
            )
            processed_info.append(
                process_image(back_input,  back_processed,  ID_CARD_WIDTH_MM,  ID_CARD_HEIGHT_MM,  doc_type="id")
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
