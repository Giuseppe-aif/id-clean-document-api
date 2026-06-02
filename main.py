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

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


app = FastAPI()

API_KEY = os.getenv("API_KEY", "")

ID_CARD_WIDTH_MM  = 85.60
ID_CARD_HEIGHT_MM = 53.98
ID_CARD_ASPECT    = ID_CARD_WIDTH_MM / ID_CARD_HEIGHT_MM   # ≈ 1.5857

PASSPORT_WIDTH_MM  = 125.0
PASSPORT_HEIGHT_MM = 88.0
PASSPORT_ASPECT    = PASSPORT_WIDTH_MM / PASSPORT_HEIGHT_MM  # ≈ 1.4205

DPI = 300

# A4 page/layout settings.
DOCX_PAGE_LEFT_MARGIN_MM   = 20.0
DOCX_PAGE_RIGHT_MARGIN_MM  = 20.0
DOCX_PAGE_TOP_MARGIN_MM    = 20.0
DOCX_PAGE_BOTTOM_MARGIN_MM = 20.0

# Desired image start position from the physical left edge of the A4 page.
PAGE_IMAGE_LEFT_MARGIN_MM = 40.0

# White space added in PDF/DOCX around each card (not baked into the PNG).
PDF_CARD_MARGIN_MM = 8.0

# Border styling.
BORDER_THICKNESS_PX = 2
BORDER_COLOR = (170, 170, 170)   # light grey, BGR

# How much the detected contour aspect ratio may differ from true card ratio.
ASPECT_TOLERANCE = 0.25

# Minimum fraction of image area the card contour must occupy.
MIN_CARD_AREA_RATIO = 0.10

# Flood-fill background replacement: pixels darker than this are considered background.
DARK_BG_THRESHOLD = 60


@app.get("/")
def home():
    return {"status": "ok", "message": "ID Clean Document API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


def safe_file_part(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = value.strip("_")
    return value or "clean_document"


def mm_to_px(mm_value: float, dpi: int = DPI) -> int:
    return int(round(mm_value / 25.4 * dpi))


def read_image(path: Path):
    data  = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path.name}")
    return image


def save_png(path: Path, image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(image_rgb)
    pil_img.save(path, dpi=(DPI, DPI))


def resize_for_detection(image, max_dim=1500):
    h, w    = image.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return image.copy(), 1.0
    scale   = max_dim / float(largest)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    return resized, scale


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s       = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff    = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
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
        [0,             0            ],
        [max_width - 1, 0            ],
        [max_width - 1, max_height - 1],
        [0,             max_height - 1],
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped  = cv2.warpPerspective(
        image, matrix, (max_width, max_height),
        borderValue=(255, 255, 255),
    )
    return warped


def replace_dark_background_with_white(image):
    """
    Flood-fills dark background pixels from all four corners with white.
    Preserves dark content inside the card (text, photo, barcodes).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    dark_mask  = (gray < DARK_BG_THRESHOLD).astype(np.uint8)
    flood_input = dark_mask.copy()
    fill_mask   = np.zeros((h + 2, w + 2), dtype=np.uint8)

    for (fy, fx) in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        if dark_mask[fy, fx] == 1:
            cv2.floodFill(flood_input, fill_mask, (fx, fy), 2)

    result = image.copy()
    result[flood_input == 2] = [255, 255, 255]
    return result


def tight_crop_to_content(image):
    """
    After background replacement, find the bounding box of all non-white pixels
    and crop tightly to it, leaving a tiny 2 px margin.
    """
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask   = gray < 250          # anything not pure white
    coords = np.argwhere(mask)

    if coords.size == 0:
        return image

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    pad = 2
    h, w = image.shape[:2]
    y0 = max(y0 - pad, 0)
    x0 = max(x0 - pad, 0)
    y1 = min(y1 + pad, h - 1)
    x1 = min(x1 + pad, w - 1)

    cropped = image[y0:y1 + 1, x0:x1 + 1]

    if cropped.shape[0] < 50 or cropped.shape[1] < 50:
        return image

    return cropped


def find_card_contour(image, target_aspect, detection_scale=1500):
    """
    Scans contours from largest to smallest and returns the best-matching
    quadrilateral whose aspect ratio is close to target_aspect.
    Returns (4×2 float32 pts in original-image coords) or None.
    """
    resized, scale = resize_for_detection(image, max_dim=detection_scale)
    img_area       = resized.shape[0] * resized.shape[1]

    # --- edge map ---
    gray  = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 120)
    edges = cv2.dilate(edges, None, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_pts  = None
    best_area = 0

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        area = cv2.contourArea(contour)
        if area < img_area * MIN_CARD_AREA_RATIO:
            break                          # remaining contours are too small

        perimeter = cv2.arcLength(contour, True)
        approx    = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) != 4:
            continue

        pts    = approx.reshape(4, 2).astype("float32")
        rect   = order_points(pts)
        tl, tr, br, bl = rect

        w_top = np.linalg.norm(tr - tl)
        w_bot = np.linalg.norm(br - bl)
        h_lft = np.linalg.norm(bl - tl)
        h_rgt = np.linalg.norm(br - tr)

        card_w  = (w_top + w_bot) / 2.0
        card_h  = (h_lft + h_rgt) / 2.0

        if card_h < 1:
            continue

        aspect = card_w / card_h

        if abs(aspect - target_aspect) > ASPECT_TOLERANCE:
            # Try rotated aspect (portrait photo of landscape card).
            if abs((1.0 / aspect) - target_aspect) > ASPECT_TOLERANCE:
                continue

        if area > best_area:
            best_area = area
            best_pts  = pts / scale     # back to original-image coordinates

    return best_pts


def extract_card(image, target_aspect):
    """
    Full extraction pipeline for one card image:
      1. Try contour-based perspective correction matched to target aspect ratio.
      2. Replace dark background with white.
      3. Tight-crop to remaining content.
    Returns the cleaned card image.
    """
    pts = find_card_contour(image, target_aspect)

    if pts is not None:
        corrected = four_point_transform(image, pts)
        method    = "perspective_corrected"
    else:
        # Fallback: replace background first, then tight-crop.
        corrected = image.copy()
        method    = "fallback_crop"

    corrected = replace_dark_background_with_white(corrected)
    corrected = tight_crop_to_content(corrected)

    return corrected, method


def rotate_to_landscape(image):
    h, w = image.shape[:2]
    if h > w:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), "rotated_landscape"
    return image, "no_rotation"


def render_card_png(image, target_width_mm, target_height_mm):
    """
    Produces a PNG sized exactly to the real card dimensions at DPI resolution.
    The card fills the canvas as large as possible; a thin grey border marks edges.
    No margin is baked in — margins are handled by PDF/DOCX layout.
    """
    image, rotation_status = rotate_to_landscape(image)

    target_w_px = mm_to_px(target_width_mm)
    target_h_px = mm_to_px(target_height_mm)

    h, w  = image.shape[:2]
    scale = min(target_w_px / w, target_h_px / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized    = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas_img = np.full((target_h_px, target_w_px, 3), 255, dtype=np.uint8)

    x = (target_w_px - new_w) // 2
    y = (target_h_px - new_h) // 2
    canvas_img[y:y + new_h, x:x + new_w] = resized

    # Thin grey border so all four edges are clearly defined.
    cv2.rectangle(
        canvas_img,
        (x, y),
        (x + new_w - 1, y + new_h - 1),
        BORDER_COLOR,
        thickness=BORDER_THICKNESS_PX,
    )

    return canvas_img, rotation_status


def process_image(input_path: Path, output_path: Path, target_width_mm, target_height_mm):
    image          = read_image(input_path)
    target_aspect  = target_width_mm / target_height_mm

    corrected, method = extract_card(image, target_aspect)

    final_img, rotation_status = render_card_png(
        corrected, target_width_mm, target_height_mm,
    )

    save_png(output_path, final_img)

    return {
        "input_file":  input_path.name,
        "output_file": output_path.name,
        "method":      method,
        "rotation":    rotation_status,
    }


# ── DOCX helpers ──────────────────────────────────────────────────────────────

def add_left_positioned_image_to_docx(doc, image_path: Path, width_mm: float, height_mm: float):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    extra_indent_mm = max(PAGE_IMAGE_LEFT_MARGIN_MM - DOCX_PAGE_LEFT_MARGIN_MM, 0)
    paragraph.paragraph_format.left_indent = Cm(extra_indent_mm / 10.0)
    run = paragraph.add_run()
    run.add_picture(str(image_path), width=Cm(width_mm / 10.0), height=Cm(height_mm / 10.0))


def create_docx(docx_path: Path, doc_type: str, front_image: Path, back_image: Optional[Path]):
    doc     = Document()
    section = doc.sections[0]
    section.page_width     = Cm(21.0)
    section.page_height    = Cm(29.7)
    section.top_margin     = Cm(DOCX_PAGE_TOP_MARGIN_MM    / 10.0)
    section.bottom_margin  = Cm(DOCX_PAGE_BOTTOM_MARGIN_MM / 10.0)
    section.left_margin    = Cm(DOCX_PAGE_LEFT_MARGIN_MM   / 10.0)
    section.right_margin   = Cm(DOCX_PAGE_RIGHT_MARGIN_MM  / 10.0)

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

def create_pdf(pdf_path: Path, doc_type: str, front_image: Path, back_image: Optional[Path]):
    """
    Places card images at their real physical size on the A4 page.
    PDF_CARD_MARGIN_MM of white space surrounds each card on the page.
    """
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
    first_name:       str            = Form(...),
    last_name:        str            = Form(...),
    doc_type:         str            = Form(...),
    output_base_name: str            = Form(...),
    front_image:      UploadFile     = File(...),
    back_image:       Optional[UploadFile] = File(None),
    x_api_key:        Optional[str]  = Header(None),
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
        tmp        = Path(tmpdir)
        front_input = tmp / "front_input"
        back_input  = tmp / "back_input"

        await save_upload(front_image, front_input)
        if back_image:
            await save_upload(back_image, back_input)

        processed_info = []

        if doc_type == "id":
            front_processed = tmp / "front_processed.png"
            back_processed  = tmp / "back_processed.png"
            processed_info.append(process_image(front_input, front_processed, ID_CARD_WIDTH_MM,  ID_CARD_HEIGHT_MM))
            processed_info.append(process_image(back_input,  back_processed,  ID_CARD_WIDTH_MM,  ID_CARD_HEIGHT_MM))
        else:
            front_processed = tmp / "passport_processed.png"
            back_processed  = None
            processed_info.append(process_image(front_input, front_processed, PASSPORT_WIDTH_MM, PASSPORT_HEIGHT_MM))

        docx_filename = f"{output_base_name}.docx"
        pdf_filename  = f"{output_base_name}.pdf"
        docx_path     = tmp / docx_filename
        pdf_path      = tmp / pdf_filename

        create_docx(docx_path=docx_path, doc_type=doc_type, front_image=front_processed, back_image=back_processed)
        create_pdf( pdf_path=pdf_path,   doc_type=doc_type, front_image=front_processed, back_image=back_processed)

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
    first_name:       str            = Form(...),
    last_name:        str            = Form(...),
    doc_type:         str            = Form(...),
    output_base_name: str            = Form(...),
    front_image:      UploadFile     = File(...),
    back_image:       Optional[UploadFile] = File(None),
    x_api_key:        Optional[str]  = Header(None),
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if doc_type == "id" and back_image is None:
        raise HTTPException(status_code=400, detail="Back image is required for ID documents")

    return {
        "status":          "received",
        "first_name":      first_name,
        "last_name":       last_name,
        "doc_type":        doc_type,
        "output_base_name": output_base_name,
        "front_filename":  front_image.filename,
        "back_filename":   back_image.filename if back_image else None,
    }
