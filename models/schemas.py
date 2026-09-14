from typing import List, Optional
from pydantic import BaseModel, Field


# ---------- Rubric (teacher-supplied) ----------

class Criterion(BaseModel):
    criterion: str                 # e.g. "Correct formula identified"
    marks: float                   # max marks this criterion is worth


class RubricQuestion(BaseModel):
    question_number: str           # e.g. "1", "2a", "3b"
    question_text: Optional[str] = None
    max_marks: float
    criteria: List[Criterion]
    model_answer: Optional[str] = None   # optional reference/ideal answer
    # markers used to detect where this question starts in the handwritten script,
    # e.g. ["Q1", "1)", "1."] — if omitted, the segmenter falls back to question_number
    start_markers: Optional[List[str]] = None


class Rubric(BaseModel):
    exam_title: str
    total_marks: float
    questions: List[RubricQuestion]


# ---------- OCR intermediate output ----------

class OCRLine(BaseModel):
    text: str
    bbox: List[List[float]]        # 4-point polygon [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    page_number: int
    confidence: Optional[float] = None


class PageText(BaseModel):
    page_number: int
    lines: List[OCRLine]
    full_text: str                 # lines joined in reading order


# ---------- Grading output ----------

class CriterionResult(BaseModel):
    criterion: str
    max_marks: float
    marks_awarded: float
    justification: str


class QuestionResult(BaseModel):
    question_number: str
    max_marks: float
    marks_awarded: float
    criteria_breakdown: List[CriterionResult]
    feedback: str
    extracted_student_answer: str


class ExamReport(BaseModel):
    exam_title: str
    total_marks: float
    total_awarded: float
    percentage: float
    questions: List[QuestionResult]
