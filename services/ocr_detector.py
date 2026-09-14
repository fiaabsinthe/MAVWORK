"""
Stage 2: Text-line DETECTION using PaddleOCR.

We deliberately ignore PaddleOCR's own text *recognition* output — it's tuned
for printed text and performs poorly on handwriting. We only want its
detection boxes, which are then cropped and handed to TrOCR (ocr_recognizer.py)
for actual transcription.
"""
from typing import List
import numpy as np
from paddleocr import PaddleOCR

from config import PADDLE_LANG, PADDLE_USE_GPU

_paddle_instance = None


def _get_paddle():
    global _paddle_instance
    if _paddle_instance is None:
        # det=True, rec=False would be ideal, but PaddleOCR's high-level API
        # couples them; we call `.ocr(..., rec=False)` at inference time instead,
        # which returns detection boxes only.
        _paddle_instance = PaddleOCR(
            lang=PADDLE_LANG,
            use_angle_cls=True,
            use_gpu=PADDLE_USE_GPU,
            show_log=False,
        )
    return _paddle_instance


def detect_line_boxes(image: np.ndarray) -> List[List[List[float]]]:
    """
    Returns a list of 4-point polygons (each a list of [x, y] pairs),
    one per detected text line, in the order PaddleOCR finds them
    (not yet guaranteed to be reading order — see sort_boxes_reading_order).
    """
    paddle = _get_paddle()
    # rec=False -> skip PaddleOCR's own recognizer, we only want boxes
    result = paddle.ocr(image, rec=False, cls=True)

    boxes = []
    # result is a list-per-image; we pass a single image so take result[0]
    if result and result[0]:
        for box in result[0]:
            boxes.append(box)
    return boxes


def sort_boxes_reading_order(
    boxes: List[List[List[float]]], y_tolerance: int = 15
) -> List[List[List[float]]]:
    """
    Groups boxes into horizontal "rows" (within y_tolerance of each other),
    then sorts rows top-to-bottom and boxes within each row left-to-right.
    Handwritten answers are rarely perfectly horizontal, so y_tolerance
    gives some slack for slightly sloped lines.
    """
    def top_y(box):
        return min(pt[1] for pt in box)

    def left_x(box):
        return min(pt[0] for pt in box)

    sorted_by_y = sorted(boxes, key=top_y)

    rows: List[List[List[List[float]]]] = []
    for box in sorted_by_y:
        placed = False
        for row in rows:
            if abs(top_y(row[0]) - top_y(box)) <= y_tolerance:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    ordered = []
    for row in rows:
        row_sorted = sorted(row, key=left_x)
        ordered.extend(row_sorted)
    return ordered
