"""The shared graph state; only trace uses an append reducer."""

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Document(TypedDict):
    id: str
    title: str
    text: str
    source: str
    score: float


class Grade(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    relevant: bool = Field(strict=True)
    reason: str = Field(min_length=1, max_length=700)


class GradeBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grades: list[Grade]


class RunRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    mode: Literal["demo", "live"] = "demo"
    top_k: int = Field(default=3, ge=1, le=6)
    relevance_threshold: float = Field(default=0.5, gt=0, le=1)
    max_retries: int = Field(default=2, ge=0, le=3)
    use_web: bool = False

    @field_validator("question")
    @classmethod
    def clean_question(cls, value):
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Enter a question of at least 3 non-whitespace characters.")
        return value


class CRAGState(TypedDict):
    question: str  # Immutable user intent
    query: str  # Changes after rewrite_query
    documents: list[Document]  # Current retrieval batch
    grades: list[dict]
    relevant_documents: list[Document]
    quality: float  # relevant / retrieved; zero for an empty batch
    retries: int
    max_retries: int
    top_k: int
    relevance_threshold: float
    use_web: bool
    web_attempted: bool
    answer: str
    outcome: str
    trace: Annotated[list[dict], operator.add]
