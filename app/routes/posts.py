import json
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.db import db
from app.lib.guard import evaluate_candidate
from app.lib.matching import load_post_embedding, suggest_for_post
from app.lib.similarity import cosine_similarity

router = APIRouter()


def _now():
    return datetime.now(timezone.utc).isoformat()


@router.get("/posts")
def list_posts():
    with db() as conn:
        rows = conn.execute("SELECT * FROM posts ORDER BY id").fetchall()
    return [dict(r) for r in rows]


@router.get("/posts/{post_id}")
def get_post(post_id: int):
    with db() as conn:
        row = conn.execute("SELECT * FROM posts WHERE id = ?", (post_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Post not found.")
    return dict(row)


@router.get("/posts/{post_id}/images")
def suggest_images_for_post(post_id: int):
    """The core matching + guard endpoint. Ranks eligible images by
    semantic similarity to the post, walks the ranked list through the
    mismatch guard, and returns either the first candidate that clears
    every check (suggested) or an explicit "no confident match" verdict
    — plus the full evaluated trail, so a reviewer (or an evaluator) can
    see exactly why each candidate was accepted or refused."""
    with db() as conn:
        post = conn.execute("SELECT * FROM posts WHERE id = ?", (post_id,)).fetchone()
        if not post:
            raise HTTPException(404, "Post not found.")
        if load_post_embedding(conn, post_id) is None:
            raise HTTPException(409, "Post has no embedding yet — run POST /batch-jobs/embed first.")

        winner, evaluated = suggest_for_post(conn, post, record=True)

        if winner is None:
            reason = (
                f"No confident match: evaluated {len(evaluated)} candidate(s), none cleared both "
                "the category check and the similarity threshold."
                if evaluated
                else "No confident match: no images have been processed and embedded yet."
            )
            return {"post_id": post_id, "suggestion": None, "reason": reason, "evaluated": evaluated}

        return {"post_id": post_id, "suggestion": winner, "evaluated": evaluated}


@router.post("/posts/{post_id}/evaluate")
def force_evaluate_candidate(post_id: int, image_id: int):
    """Runs the guard against one SPECIFIC (post, image) pair, regardless
    of that image's rank — the direct way to demonstrate "force the wolf
    as a candidate for the fox post" without needing the wolf photo to
    accidentally out-rank everything else."""
    with db() as conn:
        post = conn.execute("SELECT * FROM posts WHERE id = ?", (post_id,)).fetchone()
        if not post:
            raise HTTPException(404, "Post not found.")
        image = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        if not image:
            raise HTTPException(404, "Image not found.")
        if image["status"] != "done":
            raise HTTPException(409, f"Image {image_id} has not been successfully tagged yet.")

        post_vec = load_post_embedding(conn, post_id)
        img_emb_row = conn.execute(
            "SELECT embedding FROM image_embeddings WHERE image_id = ? ORDER BY id DESC LIMIT 1", (image_id,)
        ).fetchone()
        if post_vec is None or img_emb_row is None:
            raise HTTPException(409, "Post or image has no embedding yet — run POST /batch-jobs/embed first.")

        sim = cosine_similarity(post_vec, json.loads(img_emb_row["embedding"]))
        verdict = evaluate_candidate(
            post["title"], image["subject"] or "", image["category"] or "", bool(image["low_confidence"]), sim
        )

        conn.execute(
            """INSERT INTO suggestions (post_id, image_id, similarity, decision, reason, rank, created_at)
               VALUES (?, ?, ?, ?, ?, NULL, ?)""",
            (post_id, image_id, sim, verdict.decision, verdict.reason, _now()),
        )

        return {
            "post_id": post_id,
            "image_id": image_id,
            "filename": image["filename"],
            "similarity": round(sim, 4),
            "decision": verdict.decision,
            "reason": verdict.reason,
        }
