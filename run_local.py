"""
Quick local test harness — grades one PDF against one rubric from the command line.

Usage:
    python run_local.py path/to/exam.pdf examples/example_rubric.json
"""
import json
import sys

from models.schemas import Rubric
from services.pdf_processor import pdf_to_images, deskew_and_clean
from services.ocr_detector import detect_line_boxes, sort_boxes_reading_order
from services.ocr_recognizer import recognize_lines
from services.segmenter import segment_by_markers
from services.grading_engine import grade_exam


def main(pdf_path: str, rubric_path: str):
    with open(rubric_path) as f:
        rubric = Rubric(**json.load(f))

    print(f"[1/4] Rendering PDF pages...")
    pages = pdf_to_images(pdf_path)
    print(f"      -> {len(pages)} page(s)")

    full_text_parts = []
    for i, page in enumerate(pages, start=1):
        print(f"[2/4] Page {i}: detecting text lines (PaddleOCR)...")
        cleaned = deskew_and_clean(page)
        boxes = sort_boxes_reading_order(detect_line_boxes(cleaned))
        print(f"      -> {len(boxes)} line(s) detected")

        print(f"[3/4] Page {i}: recognizing handwriting (TrOCR)...")
        lines = recognize_lines(cleaned, boxes)
        full_text_parts.append("\n".join(lines))

    full_text = "\n".join(full_text_parts)
    print("\n--- RAW TRANSCRIBED TEXT ---")
    print(full_text)
    print("--- END TRANSCRIBED TEXT ---\n")

    print("[4/4] Segmenting answers and grading with Gemini...")
    answers = segment_by_markers(full_text, rubric)
    report = grade_exam(rubric, answers)

    print("\n=== GRADING REPORT ===")
    print(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_local.py <exam.pdf> <rubric.json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
