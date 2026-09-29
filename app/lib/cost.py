"""Per-call cost tracking. Every vision or embedding call — success or
failure — gets one row, attributed to the image/post it was for. At
free-tier quota the real dollar cost is $0, but the log still records
Gemini's standard paid-tier rate so the numbers mean something the
moment this is pointed at a billed project."""
from datetime import datetime, timezone

from app.config import EMBEDDING_COST_PER_CALL_USD, VISION_COST_PER_CALL_USD
from app.db import db


def log_cost(call_type: str, provider: str, model: str, ref_type: str, ref_id: int, success: bool):
    cost = VISION_COST_PER_CALL_USD if call_type == "vision" else EMBEDDING_COST_PER_CALL_USD
    with db() as conn:
        conn.execute(
            """INSERT INTO cost_log (call_type, provider, model, ref_type, ref_id, cost_usd, success, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (call_type, provider, model, ref_type, ref_id, cost, 1 if success else 0,
             datetime.now(timezone.utc).isoformat()),
        )


def cost_summary():
    with db() as conn:
        rows = conn.execute(
            """SELECT call_type, COUNT(*) AS calls, SUM(cost_usd) AS total_cost,
                      SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failures
               FROM cost_log GROUP BY call_type"""
        ).fetchall()
        return [dict(r) for r in rows]
