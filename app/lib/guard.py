"""The mismatch guard — the production-critical safety layer. Given a
post and a candidate image (already ranked by similarity), decides
"is this recommendation actually good enough?" and never returns a bare
boolean: every verdict carries a human-readable reason.

Three checks, in this order, any one of which can reject:
  1. category sanity   — the image's tagged subject must not conflict
                          with an animal category the post text clearly
                          names a *different* one of.
  2. confidence floor   — an image flagged low-confidence at ingestion
                           is never suggested, whatever its similarity.
  3. similarity bar     — cosine similarity must clear SIMILARITY_THRESHOLD.
"""
import re
from dataclasses import dataclass

from app.config import SIMILARITY_THRESHOLD

# The same small set of known animal categories used by the mock
# embedder (app/lib/embeddings.py) — reused here so "what is this post
# clearly about" is decided the same way in both mock and real mode:
# literal keyword presence in the post's own text, never a ground-truth
# label (that would be answering the question with the eval key).
CATEGORY_KEYWORDS = {
    "fox": ["fox", "vulpes"],
    "wolf": ["wolf", "wolves"],
    "dog": ["dog", "puppy", "canine", "retriever", "beagle", "labrador"],
    "bear": ["bear", "grizzly"],
    "deer": ["deer", "buck", "doe", "antler", "whitetail"],
}


def expected_categories(post_title: str) -> set[str]:
    """Which known animal categories the post's TITLE clearly names —
    title, not full body. A real article's body routinely makes
    incidental comparisons ("Unlike wolves, foxes are largely solitary
    hunters...") that would make a naive full-body keyword scan see the
    post as being about both animals, defeating the exact mismatch this
    check exists to catch. A title is a much cleaner signal of what a
    post is actually about. Empty set = the title doesn't clearly name
    any of them (guard skips the category check and relies on the
    similarity threshold alone)."""
    lowered = post_title.lower()
    found = set()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(kw)}", lowered) for kw in keywords):
            found.add(category)
    return found


def _image_category(subject: str, category: str) -> str | None:
    haystack = f"{subject} {category}".lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in haystack for kw in keywords):
            return cat
    return None


@dataclass
class GuardVerdict:
    decision: str  # "suggested" | "rejected"
    reason: str


def evaluate_candidate(
    post_title: str,
    image_subject: str,
    image_category: str,
    image_low_confidence: bool,
    similarity: float,
) -> GuardVerdict:
    expected = expected_categories(post_title)
    detected = _image_category(image_subject, image_category)

    if expected and detected and detected not in expected:
        expected_str = " / ".join(sorted(expected))
        return GuardVerdict(
            "rejected",
            f"Animal category mismatch: expected {expected_str}, detected {detected}.",
        )

    if image_low_confidence:
        return GuardVerdict(
            "rejected",
            "Image was flagged low-confidence at ingestion; not eligible for suggestion "
            "regardless of similarity.",
        )

    if similarity < SIMILARITY_THRESHOLD:
        return GuardVerdict(
            "rejected",
            f"Similarity {similarity:.3f} is below the threshold {SIMILARITY_THRESHOLD:.3f}.",
        )

    return GuardVerdict(
        "suggested",
        f"Similarity {similarity:.3f} clears the {SIMILARITY_THRESHOLD:.3f} threshold; "
        f"subject '{image_subject}' does not conflict with the post's content; "
        f"not flagged low-confidence.",
    )
