from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.db import db
from app.schemas import ReviewIn

router = APIRouter()


def _now():
    return datetime.now(timezone.utc).isoformat()


@router.get("/suggestions/{suggestion_id}")
def get_suggestion(suggestion_id: int):
    """Inspect why a specific image was selected or refused for a post
    — the whole point of the review API: nothing here is a black box."""
    with db() as conn:
        row = conn.execute("SELECT * FROM suggestions WHERE id = ?", (suggestion_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Suggestion not found.")
        reviews = conn.execute(
            "SELECT * FROM reviews WHERE suggestion_id = ? ORDER BY id DESC", (suggestion_id,)
        ).fetchall()
    d = dict(row)
    d["reviews"] = [dict(r) for r in reviews]
    return d


@router.get("/posts/{post_id}/suggestions")
def list_suggestions_for_post(post_id: int):
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM suggestions WHERE post_id = ? ORDER BY created_at DESC", (post_id,)
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("/suggestions/{suggestion_id}/review")
def review_suggestion(suggestion_id: int, body: ReviewIn):
    with db() as conn:
        row = conn.execute("SELECT * FROM suggestions WHERE id = ?", (suggestion_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Suggestion not found.")
        cur = conn.execute(
            "INSERT INTO reviews (suggestion_id, decision, note, reviewed_at) VALUES (?, ?, ?, ?)",
            (suggestion_id, body.decision, body.note, _now()),
        )
    return {"id": cur.lastrowid, "suggestion_id": suggestion_id, "decision": body.decision, "note": body.note}
