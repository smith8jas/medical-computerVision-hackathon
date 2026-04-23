from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    filename: str = Field(..., description="Uploaded filename")
    probability: float = Field(..., ge=0.0, le=1.0)
    prediction: int = Field(..., ge=0, le=1)
    summary: str | None = Field(
        default=None,
        description="Optional LLM-generated explanation of the model output.",
    )
    summary_source: str | None = Field(
        default=None,
        description="Where the summary came from, or why it is unavailable.",
    )


class PredictResponse(BaseModel):
    model_status: str
    llm_status: str
    results: list[PredictionResult]
    message: str | None = None
