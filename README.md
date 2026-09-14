# AI Exam Marker

Teacher uploads a scanned/handwritten exam PDF + a rubric → the system
transcribes the handwriting and grades it against the rubric, question by
question, using Gemini as the grading engine.

## Pipeline

1. **PDF → images** (`services/pdf_processor.py`) — PyMuPDF renders each page
   at 300 DPI (configurable).
2. **Text-line detection** (`services/ocr_detector.py`) — PaddleOCR's
   *detector only* (its own recognizer is skipped; it's trained on print,
   not handwriting) locates bounding boxes for each line.
3. **Handwriting recognition** (`services/ocr_recognizer.py`) — each box is
   cropped (rotation-aware) and passed through
   `microsoft/trocr-large-handwritten`.
4. **Question segmentation** (`services/segmenter.py`) — regex-based split of
   the reconstructed text into per-question chunks, using markers you define
   per question in the rubric (`"Q1"`, `"1."`, etc.).
5. **Grading** (`services/grading_engine.py`) — each question's transcribed
   answer + its rubric criteria are sent to Gemini with a strict JSON
   response schema, so you get back per-criterion marks + justification +
   feedback, not just a free-text response.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-pro
TROCR_MODEL_NAME=microsoft/trocr-large-handwritten
PDF_RENDER_DPI=300
```

> First run will download the TrOCR model weights (~1.3GB for the large
> checkpoint) and PaddleOCR's detection model (~5MB) — both cached locally
> afterward.

## Run

**As an API:**
```bash
uvicorn main:app --reload --port 8000
```
Then POST to `/grade-exam` with `pdf_file` (file) and `rubric_json` (string,
see `examples/example_rubric.json`) as multipart form-data.

**As a CLI (for quick testing):**
```bash
python run_local.py path/to/exam.pdf examples/example_rubric.json
```

## Rubric format

See `examples/example_rubric.json`. Key fields per question:
- `criteria`: list of `{criterion, marks}` — Gemini grades against each one
  individually rather than giving one holistic score.
- `model_answer` (optional): guidance only, not a verbatim-match requirement.
- `start_markers` (optional): the literal strings that mark where this
  question begins in the student's handwriting (e.g. `["Q1", "1."]`). If a
  student doesn't write a recognizable marker, that question is flagged for
  manual review rather than silently misgraded.

## Known limitations & where to invest next

- **Segmentation is the weakest link.** If your exam has a fixed
  answer-booklet layout (e.g. one page per question, or a printed answer
  box), segmenting by *page/position* instead of by regex marker will be far
  more reliable than hoping students write "Q1" clearly. Swap
  `segment_by_markers` for a position-based segmenter if that fits your
  format.
- **TrOCR line-by-line has no page-level context.** For messy handwriting,
  consider a second pass where Gemini itself is shown the *raw line crops as
  images* (it accepts image input) alongside the TrOCR guess, and asked to
  correct obvious transcription errors before grading — this can meaningfully
  improve accuracy for cursive or low-quality scans.
- **No human-in-the-loop review yet.** For high-stakes grading, add a review
  step: show the teacher the transcribed text + Gemini's marks side by side
  before finalizing, especially for questions flagged as segmentation
  failures.
- **Single exam per request.** For a full classroom batch, wrap
  `/grade-exam` in a queue (e.g. Celery + Redis) so uploads don't block on
  the full OCR+grading pipeline synchronously.
- **GPU strongly recommended for TrOCR** at scale — CPU inference works but
  is slow per line; a class set of 30 exams × 10 questions will add up.
