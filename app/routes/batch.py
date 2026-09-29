from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.db import db
from app.jobs import create_job, run_embed_job
from app.schemas import BatchJobOut

router = APIRouter()


@router.post("/batch-jobs/embed")
def start_embed_job(background_tasks: BackgroundTasks):
    """Embeds every done-but-unembedded image caption and every
    unembedded post. Background job, same as ingestion."""
    with db() as conn:
        pending_images = conn.execute(
            "SELECT COUNT(*) AS c FROM images WHERE status = 'done' AND id NOT IN (SELECT image_id FROM image_embeddings)"
        ).fetchone()["c"]
        pending_posts = conn.execute(
            "SELECT COUNT(*) AS c FROM posts WHERE id NOT IN (SELECT post_id FROM post_embeddings)"
        ).fetchone()["c"]
    total = pending_images + pending_posts
    if total == 0:
        return {"message": "Nothing to embed.", "total": 0}
    job_id = create_job("embed", total)
    background_tasks.add_task(run_embed_job, job_id)
    return {"job_id": job_id, "status": "running", "total": total}


@router.get("/batch-jobs/{job_id}", response_model=BatchJobOut)
def get_job(job_id: int):
    with db() as conn:
        row = conn.execute("SELECT * FROM batch_jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Batch job not found.")
    return dict(row)


@router.get("/batch-jobs")
def list_jobs():
    with db() as conn:
        rows = conn.execute("SELECT * FROM batch_jobs ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]
