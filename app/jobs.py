"""Batch jobs: vision ingestion and embedding generation. Both run in
the background (kicked off via FastAPI BackgroundTasks, so the endpoint
that starts one returns immediately with a job id) with per-item
retries, live progress tracked in batch_jobs, and a cost_log row for
every single call — success or failure.

Important structural rule, learned the hard way (see BUILDLOG.md): never
call a function that opens its own `with db()` connection from inside
another already-open `with db()` block. Two connections to the same
SQLite file in one process/thread, one holding an uncommitted write
transaction while waiting on the other, self-deadlocks — each blocks on
the other's lock until busy_timeout expires. Every DB-touching call in
this file is sequential and connections are always closed before the
next one opens.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import (
    GEMINI_EMBEDDING_MODEL,
    GEMINI_VISION_MODEL,
    IMAGES_DIR,
    MAX_RETRIES,
    MIN_CONFIDENCE,
    MOCK_AI,
)
from app.db import db
from app.lib.cost import log_cost
from app.lib.embeddings import EmbeddingCallError, embed_text
from app.lib.vision import VisionCallError, classify_image

PROVIDER = "gemini"


def _now():
    return datetime.now(timezone.utc).isoformat()


def create_job(kind: str, total: int) -> int:
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO batch_jobs (kind, status, total, processed, failed, started_at) VALUES (?, 'running', ?, 0, 0, ?)",
            (kind, total, _now()),
        )
        return cur.lastrowid


def _bump_job(job_id: int, processed_delta: int = 0, failed_delta: int = 0, finish: bool = False):
    with db() as conn:
        conn.execute(
            "UPDATE batch_jobs SET processed = processed + ?, failed = failed + ? WHERE id = ?",
            (processed_delta, failed_delta, job_id),
        )
        if finish:
            conn.execute(
                "UPDATE batch_jobs SET status = 'done', finished_at = ? WHERE id = ?",
                (_now(), job_id),
            )


def run_vision_ingest_job(job_id: int):
    with db() as conn:
        rows = conn.execute(
            "SELECT id, filename, true_category FROM images WHERE status = 'pending' ORDER BY id"
        ).fetchall()

    for row in rows:
        image_id, filename, true_category = row["id"], row["filename"], row["true_category"]
        image_path = str(Path(IMAGES_DIR) / filename)

        last_error = None
        attempt = 0
        tags = None
        while attempt < MAX_RETRIES:
            attempt += 1
            try:
                tags = classify_image(image_path, true_category, image_id)
                log_cost("vision", PROVIDER, GEMINI_VISION_MODEL, "image", image_id, success=True)
                break
            except VisionCallError as e:
                last_error = str(e)
                log_cost("vision", PROVIDER, GEMINI_VISION_MODEL, "image", image_id, success=False)
                if not MOCK_AI:
                    time.sleep(min(2 ** attempt, 8))  # backoff between retries on real calls

        if tags is not None:
            low_conf = 1 if tags.confidence < MIN_CONFIDENCE else 0
            with db() as conn:
                conn.execute(
                    """UPDATE images SET subject=?, category=?, attributes=?, caption=?, confidence=?,
                       low_confidence=?, vision_provider=?, vision_model=?, status='done',
                       error=NULL, attempts=?, processed_at=? WHERE id=?""",
                    (
                        tags.subject, tags.category, json.dumps(tags.attributes), tags.caption,
                        tags.confidence, low_conf, PROVIDER, GEMINI_VISION_MODEL, attempt, _now(), image_id,
                    ),
                )
            _bump_job(job_id, processed_delta=1)
        else:
            # Every retry failed: quarantined, never silently accepted.
            with db() as conn:
                conn.execute(
                    "UPDATE images SET status='failed', error=?, attempts=?, processed_at=? WHERE id=?",
                    (last_error, attempt, _now(), image_id),
                )
            _bump_job(job_id, processed_delta=1, failed_delta=1)

    _bump_job(job_id, finish=True)


def run_embed_job(job_id: int):
    with db() as conn:
        images = conn.execute(
            """SELECT id, caption FROM images
               WHERE status = 'done' AND id NOT IN (SELECT image_id FROM image_embeddings)"""
        ).fetchall()
        posts = conn.execute(
            """SELECT id, title, body FROM posts
               WHERE id NOT IN (SELECT post_id FROM post_embeddings)"""
        ).fetchall()

    for row in images:
        image_id, caption = row["id"], row["caption"]
        vector = None
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                vector = embed_text(caption)
                log_cost("embedding", PROVIDER, GEMINI_EMBEDDING_MODEL, "image", image_id, success=True)
                break
            except EmbeddingCallError as e:
                last_error = str(e)
                log_cost("embedding", PROVIDER, GEMINI_EMBEDDING_MODEL, "image", image_id, success=False)
                if not MOCK_AI:
                    time.sleep(min(2 ** attempt, 8))

        if vector is not None:
            with db() as conn:
                conn.execute(
                    "INSERT INTO image_embeddings (image_id, model, embedding, created_at) VALUES (?, ?, ?, ?)",
                    (image_id, GEMINI_EMBEDDING_MODEL, json.dumps(vector), _now()),
                )
            _bump_job(job_id, processed_delta=1)
        else:
            _bump_job(job_id, processed_delta=1, failed_delta=1)
            print(f"[embed] giving up on image {image_id}: {last_error}")

    for row in posts:
        post_id, title, body = row["id"], row["title"], row["body"]
        text = f"{title}\n\n{body}"
        vector = None
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                vector = embed_text(text)
                log_cost("embedding", PROVIDER, GEMINI_EMBEDDING_MODEL, "post", post_id, success=True)
                break
            except EmbeddingCallError as e:
                last_error = str(e)
                log_cost("embedding", PROVIDER, GEMINI_EMBEDDING_MODEL, "post", post_id, success=False)
                if not MOCK_AI:
                    time.sleep(min(2 ** attempt, 8))

        if vector is not None:
            with db() as conn:
                conn.execute(
                    "INSERT INTO post_embeddings (post_id, model, embedding, created_at) VALUES (?, ?, ?, ?)",
                    (post_id, GEMINI_EMBEDDING_MODEL, json.dumps(vector), _now()),
                )
            _bump_job(job_id, processed_delta=1)
        else:
            _bump_job(job_id, processed_delta=1, failed_delta=1)
            print(f"[embed] giving up on post {post_id}: {last_error}")

    _bump_job(job_id, finish=True)
