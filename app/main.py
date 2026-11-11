import logging
import time
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from src.generation import (OPENAI_MODEL, check_openai_config, generate_answer,)

logger = logging.getLogger(__name__)

app = FastAPI(title="The Librarian API", description=("A multilingual retrieval-augmented question-answering API " "using public South Mediterranean University sources."), version="0.3.0",)

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Question to answer using the available SMU sources.",)
    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Please provide a non-empty question.")

        return value

class Source(BaseModel):
    citation: str
    title: str
    source_url: str
    chunk_id: str

class Timings(BaseModel):
    total: float | None = None
    generation: float | None = None
    prompt_evaluation: float | None = None

class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict[str, Any]]
    generation_model: str
    timings_ms: dict[str, Any] = Field(default_factory=dict)
    service_latency_ms: float

class HealthResponse(BaseModel):
    status: str
    generation_model: str
    model_available: bool

@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "The Librarian API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model_available = False
    try:
        check_openai_config()
        model_available = True

    except RuntimeError:
        logger.warning("OpenAI configuration check failed.", exc_info=True,)

    return HealthResponse(status="ok" if model_available else "degraded", generation_model=OPENAI_MODEL, model_available=model_available,)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    start_time = time.perf_counter()
    try:
        result = generate_answer(request.question)

        elapsed_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        return AskResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result.get("sources", []),
            generation_model=result.get(
                "generation_model",
                OPENAI_MODEL,
            ),
            timings_ms=result.get("timings_ms", {}),
            service_latency_ms=elapsed_ms,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        logger.warning(
            "Question processing failed: %s",
            error,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The question could not be processed. "
                "Check the OpenAI API configuration and retrieval services."
            ),
        ) from error

    except Exception as error:
        logger.exception(
            "Unexpected error while answering a question."
        )

        raise HTTPException(
            status_code=500,
            detail="An unexpected server error occurred.",
        ) from error
