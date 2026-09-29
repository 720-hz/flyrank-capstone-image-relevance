"""Top-1 precision against the labeled eval set (every post's own
`category` field in data/posts/posts.json). For a category post,
"correct" means the suggested image's true_category (also a hand-set
label, from data/images/manifest.json — never the vision model's own
guess) matches the post's category. For an off-topic post (category
'none'), "correct" means the system correctly returned no confident
match. Prints a plain table and the headline number that belongs in
README.md.

Run with images and posts already ingested + embedded:
    python scripts/eval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import db  # noqa: E402
from app.lib.matching import suggest_for_post  # noqa: E402


def main():
    with db() as conn:
        posts = conn.execute("SELECT * FROM posts ORDER BY id").fetchall()
        if not posts:
            print("No posts found — run scripts/seed.py first.")
            return

        rows = []
        correct = 0
        for post in posts:
            winner, evaluated = suggest_for_post(conn, post, record=False)
            expected_category = post["category"]

            if expected_category == "none":
                is_correct = winner is None
                got = "no match" if winner is None else f"{winner['filename']} ({winner['true_category']})"
            else:
                is_correct = winner is not None and winner["true_category"] == expected_category
                got = f"{winner['filename']} ({winner['true_category']})" if winner else "no match"

            if is_correct:
                correct += 1
            rows.append((post["slug"], expected_category, got, "correct" if is_correct else "WRONG"))

        total = len(posts)
        precision = correct / total if total else 0.0

        width = max(len(r[0]) for r in rows) + 2
        print(f"{'post':<{width}} {'expected':<12} {'got':<40} verdict")
        print("-" * (width + 12 + 40 + 10))
        for slug, expected, got, verdict in rows:
            print(f"{slug:<{width}} {expected:<12} {got:<40} {verdict}")

        print()
        print(f"Top-1 precision: {correct}/{total} = {precision:.1%}")


if __name__ == "__main__":
    main()
