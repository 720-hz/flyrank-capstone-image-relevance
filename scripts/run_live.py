"""One-command runner for the REAL Gemini pass — no FastAPI server needed.

This calls the exact same functions the API's background jobs use
(app.jobs.run_vision_ingest_job / run_embed_job), just invoked directly and
synchronously so you can watch it run in one terminal. Use this on a machine
with normal internet access (this sandbox's own network is locked down to a
package-registry allowlist — see BUILDLOG.md for why this script exists).

Usage:
    pip install -r requirements.txt
    # GEMINI_API_KEY should already be set in .env (see .env.example)
    python scripts/run_live.py

What it does, in order:
  1. init_db() + seed_posts() + seed_images()  (idempotent — safe to re-run)
  2. Vision ingest batch job — tags every pending image with Gemini Flash
  3. Embed batch job — embeds every tagged image's caption + every post
  4. Prints the cost summary and the eval script's top-1 precision

After this finishes, image_relevance.db has real Gemini tags, embeddings,
and a real cost log — no mock data left anywhere.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import MOCK_AI  # noqa: E402
from app.db import db, init_db  # noqa: E402
from app.jobs import create_job, run_embed_job, run_vision_ingest_job  # noqa: E402
from app.lib.cost import cost_summary  # noqa: E402
from scripts.seed import seed_images, seed_posts  # noqa: E402


def _job_summary(job_id: int) -> str:
    with db() as conn:
        row = conn.execute("SELECT * FROM batch_jobs WHERE id = ?", (job_id,)).fetchone()
    return f"status={row['status']} total={row['total']} processed={row['processed']} failed={row['failed']}"


def main():
    if MOCK_AI:
        print("MOCK_AI=1 in .env — this would run mock data, not real Gemini calls.")
        print("Set MOCK_AI=0 (or remove it) in .env before running this script.")
        sys.exit(1)

    print("== 1/4: seeding posts + images (idempotent) ==")
    init_db()
    seed_posts()
    seed_images()

    with db() as conn:
        pending = conn.execute("SELECT COUNT(*) AS c FROM images WHERE status = 'pending'").fetchone()["c"]
    print(f"\n== 2/4: vision ingest — {pending} pending image(s) ==")
    if pending:
        t0 = time.time()
        job_id = create_job("vision_ingest", pending)
        run_vision_ingest_job(job_id)
        print(f"done in {time.time()-t0:.1f}s — {_job_summary(job_id)}")
    else:
        print("nothing pending, skipping")

    print("\n== 3/4: embedding job ==")
    with db() as conn:
        pending_img = conn.execute(
            "SELECT COUNT(*) AS c FROM images WHERE status='done' AND id NOT IN (SELECT image_id FROM image_embeddings)"
        ).fetchone()["c"]
        pending_post = conn.execute(
            "SELECT COUNT(*) AS c FROM posts WHERE id NOT IN (SELECT post_id FROM post_embeddings)"
        ).fetchone()["c"]
    total = pending_img + pending_post
    if total:
        t0 = time.time()
        job_id = create_job("embed", total)
        run_embed_job(job_id)
        print(f"done in {time.time()-t0:.1f}s — {_job_summary(job_id)}")
    else:
        print("nothing pending, skipping")

    print("\n== 4/4: cost summary ==")
    summary = cost_summary()
    print(summary)

    print("\nRunning scripts/eval.py for the real top-1 precision number...\n")
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "eval.py")])

    print("\nAll done. image_relevance.db now has real Gemini tags, embeddings, and cost log entries.")


if __name__ == "__main__":
    main()
