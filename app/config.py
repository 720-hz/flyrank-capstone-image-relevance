"""Central config, loaded once from .env. Nothing secret ever has a
default that looks like a real credential — an unset GEMINI_API_KEY fails
loudly the first time it's actually needed, not silently."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.0-flash")
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "text-embedding-004")

DB_PATH = os.environ.get("DB_PATH", str(BASE_DIR / "image_relevance.db"))
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
IMAGES_DIR = DATA_DIR / "images"
POSTS_FILE = DATA_DIR / "posts" / "posts.json"

SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.55"))
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.6"))

PORT = int(os.environ.get("PORT", "8000"))

MOCK_AI = os.environ.get("MOCK_AI", "0") == "1"

# Gemini Flash free-tier pricing is $0 up to the free-tier quota; cost
# tracking still records a nominal per-call estimate (Gemini's paid-tier
# rate) so the cost_log and README numbers mean something if this is ever
# pointed at a paid project.
VISION_COST_PER_CALL_USD = 0.0001
EMBEDDING_COST_PER_CALL_USD = 0.00001

MAX_RETRIES = 3
