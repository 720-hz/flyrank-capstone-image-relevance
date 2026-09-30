# AI Image Understanding & Content Matching Engine

Vision-tags an image corpus with Gemini, embeds captions and posts into a shared semantic space, ranks candidate images per post, and runs every candidate through a mismatch guard before anything is "suggested" — with a review API on top so a human has the final say.

**Real top-1 precision: 12/12 = 100%**, measured against a 50-image / 12-post labeled eval set, on a live run against the real Gemini API. Full numbers, the eval table, and how that number was arrived at (including the one real tuning fix along the way): [`EVIDENCE.md`](./EVIDENCE.md). How this was built, what went wrong, what changed: [`BUILDLOG.md`](./BUILDLOG.md). One-page design doc: [`DESIGN.md`](./DESIGN.md).

## What it does

1. **Vision ingest** — every image in `data/images/` is sent to Gemini (structured-output schema), producing `subject` / `category` / `attributes` / `caption` / `confidence`. Below `MIN_CONFIDENCE`, an image is stored but flagged `low_confidence` — visible downstream, never trusted as ground truth.
2. **Embedding** — every image's caption and every post's title+body are embedded into the same vector space with Gemini's embedding model. Matching happens on caption text, not raw pixels, so the whole system stays in one inspectable semantic space.
3. **Matching + mismatch guard** — for a post, images are ranked by cosine similarity; the top candidates are walked through three checks in order (category sanity → similarity threshold → confidence floor). A rejection always carries a human-readable reason, never a bare boolean.
4. **Review API** — a validated JSON API (no frontend — see `DESIGN.md`'s explicit non-goal) to inspect a post's suggestion, see why an image was picked or refused, and approve/reject it.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# edit .env: set GEMINI_API_KEY (free, no credit card: https://aistudio.google.com/apikey)
```

## Run it for real

```bash
python scripts/run_live.py
```

One command: seeds the 12 posts + 50 images (idempotent), runs the real vision ingest job, runs the real embedding job, prints the cost summary, then runs `scripts/eval.py` for the real precision number. Needs `MOCK_AI=0` in `.env` (the default) and a real `GEMINI_API_KEY`. Takes a few minutes — Gemini's free tier is paced with deliberate delays between calls (see `app/jobs.py`) rather than bursting through the corpus and eating rate limits.

To iterate on application logic without spending API quota, set `MOCK_AI=1` in `.env` first — every vision/embedding call returns deterministic, category-aware fake data instead (see `BUILDLOG.md` for why mock mode is built the way it is).

## Run the API

```bash
uvicorn app.main:app --reload
```

| Endpoint | What it does |
|---|---|
| `POST /batch-jobs/ingest` | Kick off vision tagging for every pending image (background job) |
| `POST /batch-jobs/embed` | Kick off embedding for every tagged image + every post (background job) |
| `GET /batch-jobs/{id}` / `GET /batch-jobs` | Job progress |
| `GET /images` / `GET /images/{id}` | Inspect tagged images |
| `GET /posts` / `GET /posts/{id}` | Inspect posts |
| `GET /posts/{id}/images` | Rank + guard candidates for a post, return the suggestion |
| `POST /posts/{id}/evaluate` | Force-evaluate one specific image against one post (used to verify the guard directly — see Probe 3 in `EVIDENCE.md`) |
| `GET /posts/{id}/suggestions` / `GET /suggestions/{id}` | Inspect a stored suggestion and its reasoning |
| `POST /suggestions/{id}/review` | Approve or reject a suggestion |
| `GET /cost-log` | Per-call cost log (every call, success or failure) |
| `GET /health` | Liveness |

`scripts/eval.py` calls the exact same matching function the API uses (`app/lib/matching.py`'s `suggest_for_post`), so the precision number in `EVIDENCE.md` reflects the live API's actual behavior, not a separate reimplementation.

## Config (`.env`)

| Var | Default | What it does |
|---|---|---|
| `GEMINI_API_KEY` | — | Required for any real call |
| `GEMINI_VISION_MODEL` | `gemini-3.5-flash-lite` | See `BUILDLOG.md` addendum for why not the flagship model |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | |
| `SIMILARITY_THRESHOLD` | `0.74` | Tuned against real embeddings, not the mock ones — see `EVIDENCE.md` |
| `MIN_CONFIDENCE` | `0.6` | Below this, an image is flagged `low_confidence` and never suggested |
| `MOCK_AI` | `0` | `1` = deterministic fake vision/embedding responses, no API calls |

## Data

- `data/images/` — 50 real, freely-licensed photos (10 each: fox, wolf, dog, bear, deer), with `manifest.json` mapping filename → ground-truth category.
- `data/posts/posts.json` — 12 posts: 10 with a clear animal-category ground truth (2 per category) and 2 deliberately off-topic, whose correct system behavior is "no confident match."
- `image_relevance.db` (gitignored, never committed) — SQLite; recreated locally by `scripts/run_live.py` or the API's own startup.

## Known limitation

The mismatch guard's category-sanity check only fires for posts whose title names a known animal category. Off-topic posts rely on `SIMILARITY_THRESHOLD` alone for protection — tuned with real margin against this corpus, but not a structural guarantee for every possible off-topic post. Full explanation in `EVIDENCE.md`'s "Known limitation" section.
