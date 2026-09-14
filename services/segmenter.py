"""
Stage 4: Split the reconstructed full exam text into per-question chunks.

Handwritten question segmentation is inherently fuzzy (students don't always
write "Q1" neatly). This gives a solid regex-based default and is the main
place to invest effort if your exams have a very structured layout (e.g. a
fixed answer-booklet template) — you could instead segment purely by page/box
position if each question has a dedicated page.
"""
import re
from typing import Dict, List

from models.schemas import Rubric


def _build_marker_pattern(question_number: str, custom_markers: List[str] | None) -> re.Pattern:
    markers = custom_markers or [
        question_number,
        f"Q{question_number}",
        f"Question {question_number}",
    ]
    # Escape each marker, allow optional trailing '.', ')', ':' and surrounding whitespace
    escaped = [re.escape(m) for m in markers]
    pattern = r"(?:^|\n)\s*(?:" + "|".join(escaped) + r")\s*[\.\):\-]?\s*"
    return re.compile(pattern, flags=re.IGNORECASE)


def segment_by_markers(full_text: str, rubric: Rubric) -> Dict[str, str]:
    """
    Finds each question's start marker in the transcribed text and slices out
    the text between it and the next question's marker.

    Returns {question_number: extracted_answer_text}. If a marker isn't found,
    the whole remaining text is returned for that question with a note — this
    is the main lever for a human reviewer / confidence flag in the UI.
    """
    matches = []
    for q in rubric.questions:
        pattern = _build_marker_pattern(q.question_number, q.start_markers)
        m = pattern.search(full_text)
        if m:
            matches.append((q.question_number, m.start(), m.end()))

    matches.sort(key=lambda x: x[1])

    result: Dict[str, str] = {}
    for i, (qnum, start, end) in enumerate(matches):
        next_start = matches[i + 1][1] if i + 1 < len(matches) else len(full_text)
        result[qnum] = full_text[end:next_start].strip()

    # Any rubric question whose marker was never found gets an explicit flag
    for q in rubric.questions:
        if q.question_number not in result:
            result[q.question_number] = (
                "[SEGMENTATION FAILED — marker not found. Flag for manual review.]"
            )

    return result
