from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.db import init_db
from app.routes import batch, cost, images, posts, review

app = FastAPI(title="AI Image Understanding & Content Matching Engine")

app.include_router(images.router)
app.include_router(batch.router)
app.include_router(posts.router)
app.include_router(review.router)
app.include_router(cost.router)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.exception_handler(ValidationError)
async def pydantic_validation_handler(request: Request, exc: ValidationError):
    # Request-body validation errors already come back as clean 422s from
    # FastAPI itself; this covers a ValidationError raised anywhere else in
    # application code so it never surfaces as a raw 500.
    return JSONResponse(status_code=400, content={"error": "Validation failed.", "details": exc.errors()})
