from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    filename: str = Field(..., description="Uploaded filename")
    probability: float = Field(..., ge=0.0, le=1.0)
    prediction: int = Field(..., ge=0, le=1)


class PredictResponse(BaseModel):
    model_status: str
    results: list[PredictionResult]
    message: str | None = None
