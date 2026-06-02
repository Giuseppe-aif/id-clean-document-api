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

ID_CARD_WIDTH_MM = 85.60
ID_CARD_HEIGHT_MM = 53.98

PASSPORT_WIDTH_MM = 125.0
PASSPORT_HEIGHT_MM = 88.0

DPI = 300

DOCUMENT_SAFE_MARGIN_MM = 4.0

@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "ID Clean Document API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


def safe_file_part(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = value.strip("_")
    return value or "clean_document"


def mm_to_px(mm_value: float, dpi: int = DPI) -> int:
    return int(round(mm_value / 25.4 * dpi))


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {path.name}")

    return image


def save_png(path: Path, image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    pil_img.save(path, dpi=(DPI, DPI))


def resize_for_detection(image, max_dim=1000):
    h, w = image.shape[:2]
    largest = max(h, w)

    if largest <= max_dim:
        return image.copy(), 1.0

    scale = max_dim / float(largest)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    return resized, scale


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)

    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    if max_width < 50 or max_height < 50:
        return image

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
        borderValue=(255, 255, 255),
    )

    return warped


def try_perspective_correction(image):
    resized, scale = resize_for_detection(image, max_dim=1000)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_area = resized.shape[0] * resized.shape[1]

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)

            if area < image_area * 0.15:
                continue

            pts = approx.reshape(4, 2).astype("float32")
            pts = pts / scale

            return four_point_transform(image, pts), "perspective_corrected"

    return None, "no_contour_found"


def try_trim_border(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = gray < 245
    coords = np.argwhere(mask)

    if coords.size == 0:
        return image, "no_trim"

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    h, w = image.shape[:2]
    margin = int(min(h, w) * 0.02)

    y0 = max(y0 - margin, 0)
    x0 = max(x0 - margin, 0)
    y1 = min(y1 + margin, h - 1)
    x1 = min(x1 + margin, w - 1)

    cropped = image[y0:y1 + 1, x0:x1 + 1]

    if cropped.shape[0] < 100 or cropped.shape[1] < 100:
        return image, "trim_failed"

    return cropped, "border_trimmed"


def rotate_to_landscape(image):
    h, w = image.shape[:2]

    if h > w:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), "rotated_landscape"

    return image, "no_rotation"


def fit_on_white_canvas(image, target_width_mm, target_height_mm):
    """
    Creates a final image with the exact target physical size,
    but leaves a safe white margin around the document so that
    the full document remains clearly visible.
    """

    image, rotation_status = rotate_to_landscape(image)

    canvas_w = mm_to_px(target_width_mm)
    canvas_h = mm_to_px(target_height_mm)

    margin_px = mm_to_px(DOCUMENT_SAFE_MARGIN_MM)

    inner_w = max(canvas_w - 2 * margin_px, 1)
    inner_h = max(canvas_h - 2 * margin_px, 1)

    h, w = image.shape[:2]

    scale = min(inner_w / w, inner_h / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas_img = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    x = (canvas_w - new_w) // 2
    y = (canvas_h - new_h) // 2

    canvas_img[y:y + new_h, x:x + new_w] = resized

    return canvas_img, rotation_status


def process_image(input_path: Path, output_path: Path, target_width_mm, target_height_mm):
    image = read_image(input_path)

    corrected, method = try_perspective_correction(image)

    if corrected is None:
        corrected, method = try_trim_border(image)

    final_img, rotation_status = fit_on_white_canvas(
        corrected,
        target_width_mm,
        target_height_mm,
    )

    save_png(output_path, final_img)

    return {
        "input_file": input_path.name,
        "output_file": output_path.name,
        "method": method,
        "rotation": rotation_status,
    }


def create_docx(docx_path: Path, doc_type: str, front_image: Path, back_image: Optional[Path]):
    doc = Document()

    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)

    if doc_type == "id":
        add_centered_image_to_docx(
            doc,
            front_image,
            ID_CARD_WIDTH_MM,
            ID_CARD_HEIGHT_MM,
        )

        doc.add_paragraph("")

        add_centered_image_to_docx(
            doc,
            back_image,
            ID_CARD_WIDTH_MM,
            ID_CARD_HEIGHT_MM,
        )

    elif doc_type == "passport":
        add_centered_image_to_docx(
            doc,
            front_image,
            PASSPORT_WIDTH_MM,
            PASSPORT_HEIGHT_MM,
        )

    else:
        raise ValueError(f"Unsupported document type: {doc_type}")

    doc.save(docx_path)


def add_centered_image_to_docx(doc, image_path: Path, width_mm: float, height_mm: float):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    run = paragraph.add_run()
    run.add_picture(
        str(image_path),
        width=Cm(width_mm / 10.0),
        height=Cm(height_mm / 10.0),
    )


def create_pdf(pdf_path: Path, doc_type: str, front_image: Path, back_image: Optional[Path]):
    page_w, page_h = A4
    c = canvas.Canvas(str(pdf_path), pagesize=A4)

    if doc_type == "id":
        img_w = ID_CARD_WIDTH_MM * mm
        img_h = ID_CARD_HEIGHT_MM * mm

        x = (page_w - img_w) / 2
        y_front = page_h - 55 * mm - img_h
        y_back = y_front - 25 * mm - img_h

        c.drawImage(ImageReader(str(front_image)), x, y_front, width=img_w, height=img_h)
        c.drawImage(ImageReader(str(back_image)), x, y_back, width=img_w, height=img_h)

    elif doc_type == "passport":
        img_w = PASSPORT_WIDTH_MM * mm
        img_h = PASSPORT_HEIGHT_MM * mm

        x = (page_w - img_w) / 2
        y = page_h - 60 * mm - img_h

        c.drawImage(ImageReader(str(front_image)), x, y, width=img_w, height=img_h)

    else:
        raise ValueError(f"Unsupported document type: {doc_type}")

    c.showPage()
    c.save()


async def save_upload(upload_file: UploadFile, destination: Path):
    content = await upload_file.read()
    destination.write_bytes(content)


def file_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


@app.post("/process-document")
async def process_document(
    first_name: str = Form(...),
    last_name: str = Form(...),
    doc_type: str = Form(...),
    output_base_name: str = Form(...),
    front_image: UploadFile = File(...),
    back_image: Optional[UploadFile] = File(None),
    x_api_key: Optional[str] = Header(None),
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
        tmp = Path(tmpdir)

        front_input = tmp / "front_input"
        back_input = tmp / "back_input"

        await save_upload(front_image, front_input)

        if back_image:
            await save_upload(back_image, back_input)

        processed_info = []

        if doc_type == "id":
            front_processed = tmp / "front_processed.png"
            back_processed = tmp / "back_processed.png"

            processed_info.append(
                process_image(
                    front_input,
                    front_processed,
                    ID_CARD_WIDTH_MM,
                    ID_CARD_HEIGHT_MM,
                )
            )

            processed_info.append(
                process_image(
                    back_input,
                    back_processed,
                    ID_CARD_WIDTH_MM,
                    ID_CARD_HEIGHT_MM,
                )
            )

        else:
            front_processed = tmp / "passport_processed.png"
            back_processed = None

            processed_info.append(
                process_image(
                    front_input,
                    front_processed,
                    PASSPORT_WIDTH_MM,
                    PASSPORT_HEIGHT_MM,
                )
            )

        docx_filename = f"{output_base_name}.docx"
        pdf_filename = f"{output_base_name}.pdf"

        docx_path = tmp / docx_filename
        pdf_path = tmp / pdf_filename

        create_docx(
            docx_path=docx_path,
            doc_type=doc_type,
            front_image=front_processed,
            back_image=back_processed,
        )

        create_pdf(
            pdf_path=pdf_path,
            doc_type=doc_type,
            front_image=front_processed,
            back_image=back_processed,
        )

        return {
            "status": "success",
            "doc_type": doc_type,
            "first_name": first_name,
            "last_name": last_name,
            "output_base_name": output_base_name,
            "docx_filename": docx_filename,
            "pdf_filename": pdf_filename,
            "docx_base64": file_to_base64(docx_path),
            "pdf_base64": file_to_base64(pdf_path),
            "processed_images": processed_info,
        }


@app.post("/process-document-test")
async def process_document_test(
    first_name: str = Form(...),
    last_name: str = Form(...),
    doc_type: str = Form(...),
    output_base_name: str = Form(...),
    front_image: UploadFile = File(...),
    back_image: Optional[UploadFile] = File(None),
    x_api_key: Optional[str] = Header(None),
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if doc_type == "id" and back_image is None:
        raise HTTPException(status_code=400, detail="Back image is required for ID documents")

    return {
        "status": "received",
        "first_name": first_name,
        "last_name": last_name,
        "doc_type": doc_type,
        "output_base_name": output_base_name,
        "front_filename": front_image.filename,
        "back_filename": back_image.filename if back_image else None,
    }
