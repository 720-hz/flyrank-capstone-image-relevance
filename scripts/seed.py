"""Idempotent seed: loads data/posts/posts.json into `posts`, and every
image listed in data/images/manifest.json into `images` (status=
'pending', ready for the ingest batch job). Safe to re-run — existing
rows (matched by slug / filename) are left alone."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import IMAGES_DIR, POSTS_FILE  # noqa: E402
from app.db import db, init_db  # noqa: E402


def _now():
    return datetime.now(timezone.utc).isoformat()


def seed_posts():
    posts = json.loads(Path(POSTS_FILE).read_text())
    added = 0
    with db() as conn:
        for p in posts:
            existing = conn.execute("SELECT id FROM posts WHERE slug = ?", (p["slug"],)).fetchone()
            if existing:
                continue
            conn.execute(
                "INSERT INTO posts (slug, title, body, category, created_at) VALUES (?, ?, ?, ?, ?)",
                (p["slug"], p["title"], p["body"], p.get("category"), _now()),
            )
            added += 1
    print(f"posts: {added} added, {len(posts) - added} already present")


def seed_images():
    manifest_path = Path(IMAGES_DIR) / "manifest.json"
    if not manifest_path.exists():
        print(f"images: no manifest at {manifest_path} yet — nothing to seed")
        return
    manifest = json.loads(manifest_path.read_text())
    added = 0
    missing_files = []
    with db() as conn:
        for filename, category in manifest.items():
            if not (Path(IMAGES_DIR) / filename).exists():
                missing_files.append(filename)
                continue
            existing = conn.execute("SELECT id FROM images WHERE filename = ?", (filename,)).fetchone()
            if existing:
                continue
            conn.execute(
                "INSERT INTO images (filename, true_category, status, created_at) VALUES (?, ?, 'pending', ?)",
                (filename, category, _now()),
            )
            added += 1
    print(f"images: {added} added, {len(manifest) - added - len(missing_files)} already present, "
          f"{len(missing_files)} missing from disk")
    if missing_files:
        print(f"  missing: {missing_files}")


if __name__ == "__main__":
    init_db()
    seed_posts()
    seed_images()
