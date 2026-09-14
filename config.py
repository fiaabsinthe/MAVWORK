"""
Central configuration. Load secrets from environment / .env file — never hardcode keys.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Gemini ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")  # use a "pro"-tier model for grading quality

# --- TrOCR ---
# "microsoft/trocr-large-handwritten" is the most accurate handwritten checkpoint.
# Use "microsoft/trocr-base-handwritten" for a much faster/lighter CPU-friendly option.
TROCR_MODEL_NAME = os.getenv("TROCR_MODEL_NAME", "microsoft/trocr-large-handwritten")

# --- PaddleOCR ---
PADDLE_LANG = os.getenv("PADDLE_LANG", "en")
PADDLE_USE_GPU = os.getenv("PADDLE_USE_GPU", "false").lower() == "true"

# --- PDF rendering ---
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "300"))  # higher DPI = better OCR, slower

# --- Paths ---
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./tmp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
