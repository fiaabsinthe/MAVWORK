"""
Stage 5: Grading engine — sends each question's transcribed answer + rubric
criteria to Gemini and asks for a structured mark breakdown.
"""
import json
from typing import Dict

import google.generativeai as genai

from config import GEMINI_API_KEY, GEMINI_MODEL
from models.schemas import Rubric, RubricQuestion, QuestionResult, CriterionResult, ExamReport

genai.configure(api_key=GEMINI_API_KEY)

_GRADING_SCHEMA = {
    "type": "object",
    "properties": {
        "criteria_breakdown": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string"},
                    "marks_awarded": {"type": "number"},
                    "justification": {"type": "string"},
                },
                "required": ["criterion", "marks_awarded", "justification"],
            },
        },
        "overall_feedback": {"type": "string"},
    },
    "required": ["criteria_breakdown", "overall_feedback"],
}


def _build_prompt(question: RubricQuestion, student_answer: str) -> str:
    criteria_lines = "\n".join(
        f"- {c.criterion} (max {c.marks} marks)" for c in question.criteria
    )
    model_answer_block = (
        f"\nReference/model answer (for guidance only, do not require verbatim match):\n{question.model_answer}\n"
        if question.model_answer
        else ""
    )
    return f"""You are an experienced, fair, and consistent exam marker.

Question {question.question_number}: {question.question_text or "(no question text provided)"}
Maximum marks for this question: {question.max_marks}

Marking criteria:
{criteria_lines}
{model_answer_block}
Student's answer (transcribed via OCR from handwriting — expect occasional
transcription noise like misread letters or dropped words; use your judgement
to interpret clear intent, and don't penalize the student for OCR artifacts
that are obviously not their fault, e.g. a single garbled word in an
otherwise coherent sentence):

\"\"\"
{student_answer}
\"\"\"

Grade the student's answer against EACH criterion individually. Award partial
marks where the student shows partial understanding. Be specific in your
justification — reference what the student actually wrote. Do not award marks
for content that is absent just because it seems implied.

Respond ONLY with JSON matching this shape:
{{
  "criteria_breakdown": [
    {{"criterion": "...", "marks_awarded": <number>, "justification": "..."}}
  ],
  "overall_feedback": "2-4 sentences of constructive feedback for the student"
}}
"""


def grade_question(question: RubricQuestion, student_answer: str) -> QuestionResult:
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = _build_prompt(question, student_answer)

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=_GRADING_SCHEMA,
            temperature=0.1,  # low temperature: consistency matters more than creativity here
        ),
    )

    parsed = json.loads(response.text)

    criteria_results = []
    total_awarded = 0.0
    # map back to rubric's max marks per criterion (don't trust the model to echo them back correctly)
    max_marks_by_name = {c.criterion: c.marks for c in question.criteria}

    for item in parsed["criteria_breakdown"]:
        crit_name = item["criterion"]
        max_marks = max_marks_by_name.get(crit_name, 0)
        awarded = min(max(item["marks_awarded"], 0), max_marks)  # clamp into valid range
        total_awarded += awarded
        criteria_results.append(
            CriterionResult(
                criterion=crit_name,
                max_marks=max_marks,
                marks_awarded=awarded,
                justification=item["justification"],
            )
        )

    total_awarded = min(total_awarded, question.max_marks)

    return QuestionResult(
        question_number=question.question_number,
        max_marks=question.max_marks,
        marks_awarded=round(total_awarded, 2),
        criteria_breakdown=criteria_results,
        feedback=parsed["overall_feedback"],
        extracted_student_answer=student_answer,
    )


def grade_exam(rubric: Rubric, answers_by_question: Dict[str, str]) -> ExamReport:
    question_results = []
    for question in rubric.questions:
        student_answer = answers_by_question.get(question.question_number, "")
        result = grade_question(question, student_answer)
        question_results.append(result)

    total_awarded = sum(r.marks_awarded for r in question_results)
    percentage = (total_awarded / rubric.total_marks * 100) if rubric.total_marks else 0

    return ExamReport(
        exam_title=rubric.exam_title,
        total_marks=rubric.total_marks,
        total_awarded=round(total_awarded, 2),
        percentage=round(percentage, 2),
        questions=question_results,
    )
