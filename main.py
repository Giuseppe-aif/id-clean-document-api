from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from typing import Optional
import os

app = FastAPI()

API_KEY = os.getenv("API_KEY", "")


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
