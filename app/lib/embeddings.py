"""Text embeddings for both image captions and post text, into one
shared semantic space — that's what makes "red fox", "Vulpes vulpes",
and "wild fox species" comparable even though the words differ. Real
path: Gemini's text-embedding-004 (free tier). Mock path: a small
deterministic bag-of-keywords vector, still built to preserve *relative*
similarity by animal category so mock-mode matching/ranking/guard
verification is meaningful, not just plumbing that returns zeros.
"""
import hashlib
import re

import httpx

from app.config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL, MOCK_AI

GEMINI_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
)


class EmbeddingCallError(Exception):
    pass


_MOCK_CATEGORY_KEYWORDS = {
    "fox": ["fox", "vulpes"],
    "wolf": ["wolf", "wolves", "pack"],
    "dog": ["dog", "puppy", "canine", "retriever", "beagle"],
    "bear": ["bear", "grizzly", "hibernat"],
    "deer": ["deer", "buck", "doe", "antler", "whitetail", "rut"],
}
_MOCK_DIM = len(_MOCK_CATEGORY_KEYWORDS) + 8  # 5 category dims + 8 hash-noise dims


def _mock_embed(text: str) -> list[float]:
    lowered = text.lower()
    vec = []
    for _, keywords in _MOCK_CATEGORY_KEYWORDS.items():
        count = sum(len(re.findall(kw, lowered)) for kw in keywords)
        vec.append(float(count))
    # Deterministic low-magnitude noise tail so no two distinct texts land
    # on the exact same vector, without swamping the category signal above.
    digest = hashlib.sha256(lowered.encode("utf-8")).digest()
    for b in digest[:8]:
        vec.append((b / 255.0) * 0.05)
    norm = sum(v * v for v in vec) ** 0.5
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def embed_text(text: str) -> list[float]:
    if MOCK_AI:
        return _mock_embed(text)

    if not GEMINI_API_KEY:
        raise EmbeddingCallError("GEMINI_API_KEY is not set — cannot make a real embedding call.")

    url = GEMINI_EMBED_URL.format(model=GEMINI_EMBEDDING_MODEL)
    # taskType is optional but Google's own docs recommend setting it;
    # SEMANTIC_SIMILARITY matches exactly what this app does with the
    # result — cosine-comparing a post's embedding against an image
    # caption's embedding.
    body = {"content": {"parts": [{"text": text}]}, "taskType": "SEMANTIC_SIMILARITY"}
    try:
        resp = httpx.post(url, params={"key": GEMINI_API_KEY}, json=body, timeout=30)
    except httpx.HTTPError as e:
        raise EmbeddingCallError(f"network error calling Gemini embeddings: {e}") from e

    if resp.status_code != 200:
        raise EmbeddingCallError(f"Gemini embeddings returned HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
        # Current API returns {"embeddings": [{"values": [...]}]} (plural,
        # list) — the older text-embedding-004 API returned a singular
        # {"embedding": {"values": [...]}}, which is what this used to
        # parse before that model was retired. Handle both shapes so this
        # keeps working if the response format shifts again.
        if "embeddings" in data:
            return data["embeddings"][0]["values"]
        return data["embedding"]["values"]
    except (KeyError, IndexError, TypeError) as e:
        raise EmbeddingCallError(f"could not extract embedding from response: {e}") from e
