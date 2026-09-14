"""
Stage 1: Convert an uploaded exam PDF into a list of page images.

Uses PyMuPDF (fitz) instead of pdf2image/poppler so there's no external
binary dependency — just `pip install PyMuPDF`.
"""
from typing import List
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from config import PDF_RENDER_DPI


def pdf_to_images(pdf_path: str) -> List[np.ndarray]:
    """
    Renders every page of the PDF to an RGB numpy array (H, W, 3), uint8.
    Returns a list ordered page 1 -> N.
    """
    doc = fitz.open(pdf_path)
    zoom = PDF_RENDER_DPI / 72  # PDF base unit is 72 dpi
    matrix = fitz.Matrix(zoom, zoom)

    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(np.array(img))
    doc.close()
    return images


def deskew_and_clean(image: np.ndarray) -> np.ndarray:
    """
    Light preprocessing to help both PaddleOCR detection and TrOCR recognition:
    - grayscale + adaptive threshold can hurt TrOCR (it expects natural-looking
      images), so we keep this deliberately mild: just contrast normalization.
    """
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # CLAHE improves contrast on unevenly lit scanned/phone-photographed pages
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return enhanced_rgb
