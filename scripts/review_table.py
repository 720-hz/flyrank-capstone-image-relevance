"""A plain-text review table — the brief explicitly allows this instead
of a UI ("validated endpoints plus a table are enough"). Prints every
post's latest suggestion, its status, and any human review decision.

Run with the API server NOT required (reads the DB directly):
    python scripts/review_table.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import db  # noqa: E402


def main():
    with db() as conn:
        posts = conn.execute("SELECT * FROM posts ORDER BY id").fetchall()
        rows = []
        for post in posts:
            sugg = conn.execute(
                "SELECT * FROM suggestions WHERE post_id = ? ORDER BY id DESC LIMIT 1", (post["id"],)
            ).fetchone()
            if not sugg:
                rows.append((post["slug"], "-", "not evaluated yet", "-"))
                continue
            review = conn.execute(
                "SELECT * FROM reviews WHERE suggestion_id = ? ORDER BY id DESC LIMIT 1", (sugg["id"],)
            ).fetchone()
            image = "-"
            if sugg["image_id"]:
                img = conn.execute("SELECT filename FROM images WHERE id = ?", (sugg["image_id"],)).fetchone()
                image = img["filename"] if img else f"#{sugg['image_id']}"
            review_str = f"{review['decision']}" if review else "unreviewed"
            rows.append((post["slug"], image, sugg["decision"], review_str))

    w0 = max(len(r[0]) for r in rows + [("post",)]) + 2
    w1 = max(len(r[1]) for r in rows + [("", "image")]) + 2
    w2 = max(len(r[2]) for r in rows + [("", "", "decision")]) + 2
    print(f"{'post':<{w0}}{'image':<{w1}}{'decision':<{w2}}{'human review'}")
    print("-" * (w0 + w1 + w2 + 14))
    for slug, image, decision, review_str in rows:
        print(f"{slug:<{w0}}{image:<{w1}}{decision:<{w2}}{review_str}")


if __name__ == "__main__":
    main()
