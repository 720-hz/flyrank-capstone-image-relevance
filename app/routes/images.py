import json

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.db import db
from app.jobs import create_job, run_vision_ingest_job

router = APIRouter()


@router.post("/batch-jobs/ingest")
def start_ingest_job(background_tasks: BackgroundTasks):
    """Kicks off vision tagging for every image currently 'pending'.
    Returns immediately with the job id — the actual (slow, bulk) work
    happens off the request path, in the background."""
    with db() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM images WHERE status = 'pending'").fetchone()["c"]
    if total == 0:
        return {"message": "No pending images to ingest.", "total": 0}
    job_id = create_job("vision_ingest", total)
    background_tasks.add_task(run_vision_ingest_job, job_id)
    return {"job_id": job_id, "status": "running", "total": total}


@router.get("/images")
def list_images(status: str | None = None):
    with db() as conn:
        if status:
            rows = conn.execute("SELECT * FROM images WHERE status = ? ORDER BY id", (status,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM images ORDER BY id").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("attributes"):
            d["attributes"] = json.loads(d["attributes"])
        out.append(d)
    return out


@router.get("/images/{image_id}")
def get_image(image_id: int):
    with db() as conn:
        row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Image not found.")
    d = dict(row)
    if d.get("attributes"):
        d["attributes"] = json.loads(d["attributes"])
    return d
