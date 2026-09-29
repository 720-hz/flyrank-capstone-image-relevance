# Design — AI Image Understanding & Content Matching Engine

One page, written before Phase 2, per the brief's Phase 1 gate.

## Image metadata schema

Every image is run through the vision model and must produce this exact shape (validated with Pydantic; anything that doesn't fit is a failed call, retried, and eventually quarantined — never silently accepted):

```json
{
  "subject": "red fox",
  "category": "animal",
  "attributes": ["orange fur", "wild", "forest"],
  "caption": "A red fox standing in a forest",
  "confidence": 0.94
}
```

`confidence` is the model's own 0-1 self-estimate. Below `MIN_CONFIDENCE` (`.env`, default 0.6), the row is stored but flagged `low_confidence = 1` — visible to the matching step and the review API, never treated as trustworthy ground truth.

## Matching strategy

1. Embed every image's **caption** (not the raw pixels) and every post's **title + body** into the same vector space (Gemini's `text-embedding-004`, free tier).
2. For a given post, rank all images by cosine similarity between the post's embedding and each image's caption embedding.
3. Hand the ranked list to the mismatch guard before anything is "suggested."

Matching on caption text rather than a raw image embedding keeps the whole system in one semantic space (post text and image *meaning*, both as text) and makes the guard's reasoning inspectable — every rejection can quote the actual tag/category that didn't fit, not an opaque vector distance.

## The mismatch guard

Three checks, in order, on the top-ranked candidate (and cascading to the next if one fails):

1. **Category/tag sanity** — if the image's `subject` is a known *different* labeled subject than what the post is clearly about (e.g. post mentions "fox", image's subject is "wolf"), reject outright with an explanit category-mismatch reason, regardless of similarity score. This is what makes the wolf-on-a-fox-post case provably fail, not just usually fail.
2. **Similarity threshold** — the cosine similarity must clear `SIMILARITY_THRESHOLD` (`.env`, default 0.55, tuned against the labeled eval set — see README). Below it: "no confident match."
3. **Confidence floor** — an image flagged `low_confidence` at ingestion is never suggested, even if it would otherwise rank first and pass the similarity bar. A low-confidence tag is not a trustworthy basis for a recommendation.

A rejection always carries a human-readable `reason` string — never a bare boolean. That's the difference between a safety layer and a coin flip.

## Database design

```
posts
  id, title, body, created_at

images
  id, filename, subject, category, attributes (JSON array),
  caption, confidence, low_confidence (bool),
  vision_provider, vision_model, status (pending|done|failed),
  error, created_at, processed_at
  INDEX(status)

image_embeddings
  id, image_id -> images.id, model, embedding (JSON array of floats),
  created_at
  INDEX(image_id)

post_embeddings
  id, post_id -> posts.id, model, embedding (JSON array of floats),
  created_at
  INDEX(post_id)

suggestions
  id, post_id -> posts.id, image_id -> images.id (nullable — null means
  "no confident match"), similarity, decision (suggested|rejected|none),
  reason, rank, created_at
  INDEX(post_id)

reviews
  id, suggestion_id -> suggestions.id, decision (approved|rejected),
  note, reviewed_at

cost_log
  id, call_type (vision|embedding), provider, model, units,
  cost_usd, post_or_image_ref, created_at

batch_jobs
  id, kind (vision_ingest), status (running|done|failed),
  total, processed, failed, started_at, finished_at
```

The labeled eval set is `posts.category` itself — every post is hand-labeled with its ground-truth animal category (or `none` for two deliberately off-topic posts). Top-1 "correctness" is the suggested image's `true_category` matching the post's `category`. A separate table pointing at one single canonical "correct" image id per post was considered and dropped: with 8-10 equally valid photos per category, forcing one photo as *the* answer would make the precision number reflect which interchangeable photo happened to rank highest, not whether the system found the right kind of image. See `scripts/eval.py`.

Embeddings are stored as JSON-encoded float arrays in SQLite (the brief explicitly allows "in-DB arrays fine at this scale" for ~50 images) — cosine similarity is computed in application code at query time. No pgvector, no separate vector store; the corpus is small enough that a full scan per query is milliseconds.

## The embed / ingest flow

```
1. Images land in data/images/ (committed to the repo, licensed-free,
   ~40-50 across 5 categories) with a manifest (data/images/manifest.json)
   mapping filename -> category (ground truth label, used only for
   building the eval set — never fed to the vision model).

2. POST /batch-jobs/ingest kicks off a background batch job: every
   `pending` image is sent to the vision model, response validated
   against the schema, retried up to N times on failure, then stored.
   Progress and per-call cost are tracked as it runs.

3. POST /batch-jobs/embed embeds every image's caption and every post's
   text, storing vectors.

4. GET /posts/{id}/images -> similarity ranking -> mismatch guard ->
   either a suggested image (ranked, with its similarity + reason it
   passed) or "no confident match" (with the reason it failed).

5. POST /suggestions/{id}/review -> approve or reject a suggestion;
   GET /suggestions/{id} -> inspect why an image was selected or refused.
```

## One explicit non-goal

**No frontend.** The review workflow is a validated JSON API plus a small script that prints a readable table — not a web UI. The brief says a full UI isn't required ("validated endpoints plus a table are enough"); building one here would be scope creep away from what's actually graded.
