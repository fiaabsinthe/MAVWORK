"""
Stage 3: Handwriting RECOGNITION using TrOCR.

Takes cropped line-images (from the boxes PaddleOCR detected) and transcribes
each one with microsoft/trocr-large-handwritten.
"""
from typing import List
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from config import TROCR_MODEL_NAME

_processor = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model():
    global _processor, _model
    if _model is None:
        _processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
        _model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME).to(_device)
        _model.eval()
    return _processor, _model


def _crop_from_polygon(image: np.ndarray, polygon: List[List[float]], pad: int = 4) -> Image.Image:
    """
    Crops an axis-aligned bounding rectangle around a (possibly rotated) polygon.
    A tighter, rotation-aware crop is possible with cv2.warpPerspective if your
    handwriting is heavily slanted — see `crop_rotated` below for that variant.
    """
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x1, x2 = max(0, int(min(xs)) - pad), int(max(xs)) + pad
    y1, y2 = max(0, int(min(ys)) - pad), int(max(ys)) + pad
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        # degenerate box, return a tiny blank patch rather than crashing the batch
        crop = np.zeros((10, 10, 3), dtype=np.uint8)
    return Image.fromarray(crop).convert("RGB")


def crop_rotated(image: np.ndarray, polygon: List[List[float]]) -> Image.Image:
    """
    Rotation-aware alternative to _crop_from_polygon — use this if lines are
    visibly slanted (common in real handwriting) for cleaner TrOCR input.
    """
    import cv2

    pts = np.array(polygon, dtype=np.float32)
    width = int(max(
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[2] - pts[3]),
    ))
    height = int(max(
        np.linalg.norm(pts[0] - pts[3]),
        np.linalg.norm(pts[1] - pts[2]),
    ))
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return Image.fromarray(warped).convert("RGB")


def recognize_lines(
    image: np.ndarray,
    boxes: List[List[List[float]]],
    batch_size: int = 8,
    use_rotation_aware_crop: bool = True,
) -> List[str]:
    """
    Runs TrOCR over each cropped line box and returns the transcribed text,
    in the same order as `boxes` was given (so sort boxes first!).
    """
    processor, model = _load_model()
    crop_fn = crop_rotated if use_rotation_aware_crop else _crop_from_polygon

    crops = [crop_fn(image, box) for box in boxes]
    texts: List[str] = []

    for i in range(0, len(crops), batch_size):
        batch = crops[i : i + batch_size]
        pixel_values = processor(images=batch, return_tensors="pt").pixel_values.to(_device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=128)
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        texts.extend(decoded)

    return texts
