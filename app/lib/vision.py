"""Vision model calls. Real path: Gemini Flash, image bytes + a prompt,
forced into VisionTags' exact JSON shape via responseSchema (structured
output) so there's no free-text parsing to get wrong. Mock path: fixed,
deterministic responses keyed off the corpus manifest's ground-truth
category, for fast iteration and CI-free grading without spending quota.

Never trusts a response by construction: this module either returns a
validated VisionTags or raises. The caller (jobs.py) owns retries; this
module owns exactly one call attempt.
"""
import base64
import json
import mimetypes
import random

import httpx
from pydantic import ValidationError

from app.config import GEMINI_API_KEY, GEMINI_VISION_MODEL, MOCK_AI
from app.schemas import VisionTags

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

VISION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "category": {"type": "string"},
        "attributes": {"type": "array", "items": {"type": "string"}},
        "caption": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["subject", "category", "attributes", "caption", "confidence"],
}

PROMPT = (
    "Look at this image and identify its single main subject. Respond with "
    "structured tags: a short 'subject' (e.g. 'red fox', 'gray wolf', "
    "'golden retriever'), a broad 'category' (e.g. 'animal'), 3-6 short "
    "'attributes' describing what's visible (color, setting, pose), a "
    "one-sentence 'caption' describing the scene, and your own 'confidence' "
    "from 0 to 1 in how certain you are about the subject identification. "
    "Be honest about confidence — a blurry, distant, or ambiguous subject "
    "should score low, not be guessed at with false certainty."
)


class VisionCallError(Exception):
    """A single vision-model attempt failed (network, API error, or the
    response didn't validate against VisionTags). The batch job retries
    on this; it is never silently swallowed."""


# Deterministic mock responses, one family of plausible outputs per
# category so repeated mock runs are stable and reproducible.
_MOCK_BY_CATEGORY = {
    "fox": [
        {"subject": "red fox", "category": "animal", "attributes": ["orange fur", "wild", "forest"], "caption": "A red fox standing in a forest clearing.", "confidence": 0.93},
        {"subject": "red fox", "category": "animal", "attributes": ["orange fur", "bushy tail", "alert"], "caption": "A red fox looking alert in tall grass.", "confidence": 0.88},
    ],
    "wolf": [
        {"subject": "gray wolf", "category": "animal", "attributes": ["gray fur", "wild", "pack animal"], "caption": "A gray wolf standing in the wilderness.", "confidence": 0.91},
        {"subject": "gray wolf", "category": "animal", "attributes": ["thick coat", "snow", "watchful"], "caption": "A gray wolf in a snowy landscape.", "confidence": 0.86},
    ],
    "dog": [
        {"subject": "domestic dog", "category": "animal", "attributes": ["pet", "friendly", "collar"], "caption": "A domestic dog sitting outdoors.", "confidence": 0.95},
        {"subject": "domestic dog", "category": "animal", "attributes": ["pet", "playful", "leash"], "caption": "A dog playing in a park.", "confidence": 0.92},
    ],
    "bear": [
        {"subject": "brown bear", "category": "animal", "attributes": ["large", "brown fur", "wild"], "caption": "A brown bear standing near trees.", "confidence": 0.9},
        {"subject": "brown bear", "category": "animal", "attributes": ["fur", "forest", "powerful"], "caption": "A bear foraging in the forest.", "confidence": 0.84},
    ],
    "deer": [
        {"subject": "white-tailed deer", "category": "animal", "attributes": ["antlers", "brown coat", "alert"], "caption": "A deer standing alert in a meadow.", "confidence": 0.89},
        {"subject": "white-tailed deer", "category": "animal", "attributes": ["grazing", "brown coat", "forest edge"], "caption": "A deer grazing at the edge of a forest.", "confidence": 0.87},
    ],
}


def _mock_classify(filename: str, true_category: str | None, seed_index: int) -> dict:
    family = _MOCK_BY_CATEGORY.get(true_category or "", None)
    if not family:
        return {
            "subject": "unknown subject",
            "category": "unknown",
            "attributes": ["unclear"],
            "caption": "The subject of this image could not be confidently identified.",
            "confidence": 0.3,
        }
    base = dict(family[seed_index % len(family)])
    # Every 7th image in its category comes back deliberately low-confidence,
    # deterministically (never random per run) — this is what Probe 1
    # ("at least one low-confidence image is flagged") demonstrates without
    # depending on luck.
    if seed_index % 7 == 6:
        base = dict(base)
        base["confidence"] = round(0.35 + (seed_index % 3) * 0.05, 2)
    return base


def classify_image(image_path: str, true_category: str | None, seed_index: int) -> VisionTags:
    if MOCK_AI:
        raw = _mock_classify(image_path, true_category, seed_index)
        return VisionTags.model_validate(raw)

    if not GEMINI_API_KEY:
        raise VisionCallError("GEMINI_API_KEY is not set — cannot make a real vision call.")

    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("ascii")

    body = {
        "contents": [
            {
                "parts": [
                    {"text": PROMPT},
                    {"inline_data": {"mime_type": mime_type, "data": image_b64}},
                ]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "response_schema": VISION_RESPONSE_SCHEMA,
        },
    }

    url = GEMINI_URL.format(model=GEMINI_VISION_MODEL)
    try:
        resp = httpx.post(url, params={"key": GEMINI_API_KEY}, json=body, timeout=30)
    except httpx.HTTPError as e:
        raise VisionCallError(f"network error calling Gemini: {e}") from e

    if resp.status_code != 200:
        raise VisionCallError(f"Gemini returned HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise VisionCallError(f"could not extract JSON from Gemini response: {e}") from e

    try:
        return VisionTags.model_validate(parsed)
    except ValidationError as e:
        raise VisionCallError(f"Gemini response failed schema validation: {e}") from e
