"""Pydantic models. VisionTags is the one contract the whole capstone
hinges on: the vision model's raw response is never trusted until it
parses against this exact shape. Anything that doesn't validate is a
failed call (retried, then quarantined) — never coerced or guessed at."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class VisionTags(BaseModel):
    subject: str = Field(..., min_length=1, max_length=100)
    category: str = Field(..., min_length=1, max_length=50)
    attributes: list[str] = Field(default_factory=list, max_length=10)
    caption: str = Field(..., min_length=1, max_length=300)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("attributes")
    @classmethod
    def cap_attribute_length(cls, v: list[str]) -> list[str]:
        return [a[:100] for a in v]


class PostIn(BaseModel):
    slug: str
    title: str
    body: str
    category: Optional[str] = None


class SuggestionOut(BaseModel):
    post_id: int
    image_id: Optional[int]
    filename: Optional[str] = None
    similarity: Optional[float]
    decision: Literal["suggested", "rejected", "none"]
    reason: str
    rank: Optional[int] = None


class ReviewIn(BaseModel):
    decision: Literal["approved", "rejected"]
    note: Optional[str] = None


class BatchJobOut(BaseModel):
    id: int
    kind: str
    status: str
    total: int
    processed: int
    failed: int
    started_at: str
    finished_at: Optional[str] = None
