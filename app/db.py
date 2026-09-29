"""SQLite storage. Same pattern as every other capstone this track: one
file, WAL mode, foreign keys on, zero external services. Embeddings are
stored as JSON-encoded float arrays — the brief explicitly allows
"in-DB arrays fine at this scale" for a ~50-image corpus; cosine
similarity is computed in Python at query time (see lib/similarity.py)."""
import sqlite3
from contextlib import contextmanager

from app.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  slug TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  category TEXT,                  -- ground-truth label for eval only, e.g. 'fox' or 'none'
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  filename TEXT NOT NULL UNIQUE,
  true_category TEXT,             -- ground-truth label for eval only, never fed to the model
  subject TEXT,
  category TEXT,
  attributes TEXT,                -- JSON array
  caption TEXT,
  confidence REAL,
  low_confidence INTEGER NOT NULL DEFAULT 0,
  vision_provider TEXT,
  vision_model TEXT,
  status TEXT NOT NULL DEFAULT 'pending',   -- pending | done | failed
  error TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  processed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);

CREATE TABLE IF NOT EXISTS image_embeddings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  image_id INTEGER NOT NULL REFERENCES images(id),
  model TEXT NOT NULL,
  embedding TEXT NOT NULL,        -- JSON array of floats
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_image_embeddings_image ON image_embeddings(image_id);

CREATE TABLE IF NOT EXISTS post_embeddings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  post_id INTEGER NOT NULL REFERENCES posts(id),
  model TEXT NOT NULL,
  embedding TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_post_embeddings_post ON post_embeddings(post_id);

CREATE TABLE IF NOT EXISTS suggestions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  post_id INTEGER NOT NULL REFERENCES posts(id),
  image_id INTEGER REFERENCES images(id),   -- NULL = "no confident match"
  similarity REAL,
  decision TEXT NOT NULL,          -- suggested | rejected | none
  reason TEXT NOT NULL,
  rank INTEGER,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_suggestions_post ON suggestions(post_id);

CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  suggestion_id INTEGER NOT NULL REFERENCES suggestions(id),
  decision TEXT NOT NULL,          -- approved | rejected
  note TEXT,
  reviewed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviews_suggestion ON reviews(suggestion_id);

CREATE TABLE IF NOT EXISTS cost_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  call_type TEXT NOT NULL,        -- vision | embedding
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  ref_type TEXT NOT NULL,         -- image | post
  ref_id INTEGER NOT NULL,
  cost_usd REAL NOT NULL,
  success INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cost_log_created ON cost_log(created_at);

CREATE TABLE IF NOT EXISTS batch_jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,             -- vision_ingest | embed
  status TEXT NOT NULL DEFAULT 'running',  -- running | done | failed
  total INTEGER NOT NULL DEFAULT 0,
  processed INTEGER NOT NULL DEFAULT 0,
  failed INTEGER NOT NULL DEFAULT 0,
  started_at TEXT NOT NULL,
  finished_at TEXT
);
"""
# The labeled eval set is posts.category itself (set in data/posts/posts.json)
# — every post already carries its own ground-truth animal category (or
# 'none' for the two off-topic posts used to prove "no confident match").
# A separate eval_labels table pointing at one single "correct" image id
# per post was considered and dropped: with 8-10 equally valid photos per
# category, forcing a single canonical photo would make the precision
# number reflect which photo happened to rank highest among interchangeable
# options, not whether the system found the right *kind* of image. See
# scripts/eval.py and BUILDLOG.md for the reasoning.


def get_conn():
    # timeout=30: with several short-lived connections opened per batch-job
    # iteration (one per image/post) plus concurrent API reads polling job
    # progress, a writer can otherwise hit SQLite's default zero-wait lock
    # and fail immediately with "database is locked" instead of just
    # waiting the few milliseconds for the other connection to finish.
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def db():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as conn:
        conn.executescript(SCHEMA)


if __name__ == "__main__":
    init_db()
    print(f"Schema applied to {DB_PATH}")
