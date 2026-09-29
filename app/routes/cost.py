from fastapi import APIRouter

from app.db import db
from app.lib.cost import cost_summary

router = APIRouter()


@router.get("/cost-log")
def get_cost_log(limit: int = 50):
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM cost_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return {"summary": cost_summary(), "recent": [dict(r) for r in rows]}
