"""The core "suggest an image for this post" logic, factored out of the
route handler so both the API (app/routes/posts.py) and the standalone
eval script (scripts/eval.py) call exactly the same code path — the
eval script's precision number is only meaningful if it's measuring the
same logic the live API actually runs, not a reimplementation of it."""
import json
from datetime import datetime, timezone

from app.lib.guard import evaluate_candidate
from app.lib.similarity import cosine_similarity

MAX_CANDIDATES_EVALUATED = 5


def _now():
    return datetime.now(timezone.utc).isoformat()


def load_post_embedding(conn, post_id: int):
    row = conn.execute(
        "SELECT embedding FROM post_embeddings WHERE post_id = ? ORDER BY id DESC LIMIT 1", (post_id,)
    ).fetchone()
    return json.loads(row["embedding"]) if row else None


def load_ranked_candidates(conn, post_vec):
    rows = conn.execute(
        """SELECT i.id, i.filename, i.subject, i.category, i.true_category, i.low_confidence, ie.embedding
           FROM images i JOIN image_embeddings ie ON ie.image_id = i.id
           WHERE i.status = 'done'"""
    ).fetchall()
    scored = []
    for r in rows:
        vec = json.loads(r["embedding"])
        sim = cosine_similarity(post_vec, vec)
        scored.append((sim, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def suggest_for_post(conn, post_row, record: bool = True):
    """Returns (winner_row_or_None, evaluated_list). If record=True, also
    writes a suggestions row for every candidate evaluated, exactly as
    the live API does."""
    post_id = post_row["id"]
    post_vec = load_post_embedding(conn, post_id)
    if post_vec is None:
        return None, []

    ranked = load_ranked_candidates(conn, post_vec)

    evaluated = []
    winner = None
    for rank, (sim, img) in enumerate(ranked[:MAX_CANDIDATES_EVALUATED], start=1):
        verdict = evaluate_candidate(
            post_row["title"], img["subject"] or "", img["category"] or "", bool(img["low_confidence"]), sim
        )
        row = {
            "image_id": img["id"],
            "filename": img["filename"],
            "subject": img["subject"],
            "true_category": img["true_category"],
            "similarity": round(sim, 4),
            "rank": rank,
            "decision": verdict.decision,
            "reason": verdict.reason,
        }
        evaluated.append(row)
        if record:
            conn.execute(
                """INSERT INTO suggestions (post_id, image_id, similarity, decision, reason, rank, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (post_id, img["id"], sim, verdict.decision, verdict.reason, rank, _now()),
            )
        if verdict.decision == "suggested" and winner is None:
            winner = row

    if winner is None and record:
        reason = (
            f"No confident match: evaluated {len(evaluated)} candidate(s), none cleared both "
            "the category check and the similarity threshold."
            if evaluated
            else "No confident match: no images have been processed and embedded yet."
        )
        conn.execute(
            """INSERT INTO suggestions (post_id, image_id, similarity, decision, reason, rank, created_at)
               VALUES (?, NULL, NULL, 'none', ?, NULL, ?)""",
            (post_id, reason, _now()),
        )

    return winner, evaluated
