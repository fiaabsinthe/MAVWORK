"""
FastAPI orchestrator.

POST /grade-exam
  form-data:
    pdf_file: the scanned handwritten exam (PDF)
    rubric_json: the rubric, as a JSON string (see examples/example_rubric.json)

Returns: ExamReport (JSON)

Run with:
    uvicorn main:app --reload --port 8000
"""
import json
import os
import shutil
import uuid

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import UPLOAD_DIR
from models.schemas import Rubric, ExamReport, PageText, OCRLine
from services.pdf_processor import pdf_to_images, deskew_and_clean
from services.ocr_detector import detect_line_boxes, sort_boxes_reading_order
from services.ocr_recognizer import recognize_lines
from services.segmenter import segment_by_markers
from services.grading_engine import grade_exam

app = FastAPI(title="AI Exam Marker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_ocr_pipeline(pdf_path: str) -> str:
    """
    Runs stages 1-3 and returns the full reconstructed exam text
    (all pages concatenated, in reading order).
    """
    pages = pdf_to_images(pdf_path)
    full_text_parts = []

    for page_number, page_image in enumerate(pages, start=1):
        cleaned = deskew_and_clean(page_image)

        raw_boxes = detect_line_boxes(cleaned)
        ordered_boxes = sort_boxes_reading_order(raw_boxes)

        line_texts = recognize_lines(cleaned, ordered_boxes)

        page_text = PageText(
            page_number=page_number,
            lines=[
                OCRLine(text=t, bbox=b, page_number=page_number)
                for t, b in zip(line_texts, ordered_boxes)
            ],
            full_text="\n".join(line_texts),
        )
        full_text_parts.append(page_text.full_text)

    return "\n".join(full_text_parts)


@app.post("/grade-exam", response_model=ExamReport)
async def grade_exam_endpoint(
    pdf_file: UploadFile = File(...),
    rubric_json: str = Form(...),
):
    # --- parse & validate rubric first (fail fast before doing expensive OCR) ---
    try:
        rubric = Rubric(**json.loads(rubric_json))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid rubric JSON: {e}")

    # --- save upload ---
    job_id = str(uuid.uuid4())
    pdf_path = os.path.join(UPLOAD_DIR, f"{job_id}.pdf")
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(pdf_file.file, f)

    try:
        # Stages 1-3: PDF -> images -> detected boxes -> transcribed text
        full_text = run_ocr_pipeline(pdf_path)

        # Stage 4: split transcribed text into per-question chunks
        answers_by_question = segment_by_markers(full_text, rubric)

        # Stage 5: grade each question with Gemini
        report = grade_exam(rubric, answers_by_question)

        return report
    finally:
        os.remove(pdf_path)


@app.get("/health")
async def health():
    return {"status": "ok"}
