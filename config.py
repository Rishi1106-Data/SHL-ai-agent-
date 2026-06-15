"""
config.py
---------
Single source of truth for all environment variables and project-level constants.
Import this everywhere instead of calling os.getenv() directly.

Usage:
    import config
    key = config.GROQ_API_KEY
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# ── LLM provider selection ────────────────────────────────────────────────────
# Set LLM_PROVIDER=gemini to switch; defaults to groq (faster, lower latency).
LLM_PROVIDER  = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY= os.getenv("GEMINI_API_KEY", "")

# Model names — override per-provider via env if needed
GROQ_MODEL    = os.getenv("GROQ_MODEL",   "llama-3.1-8b-instant")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# LLM call budget: must stay well under the evaluator's 30-second timeout.
# Two LLM calls (state extraction + comparison) must fit inside this budget.
LLM_TIMEOUT   = float(os.getenv("LLM_TIMEOUT", "22"))

# ── Agent limits (assignment hard constraints) ────────────────────────────────
MAX_TURNS           = int(os.getenv("MAX_TURNS", "8"))   # user+assistant combined
MAX_RECOMMENDATIONS = 10                                   # per assignment spec

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR         = ROOT / "data"
CATALOG_PATH     = DATA_DIR / "shl_catalog.json"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
METADATA_PATH    = DATA_DIR / "metadata.json"
