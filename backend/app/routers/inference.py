from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.model_service import UnconfiguredModelError, model_service
from backend.app.schemas import PredictResponse, PredictionResult


router = APIRouter(tags=["inference"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/predict", response_model=PredictResponse)
async def predict(files: list[UploadFile] = File(...)) -> PredictResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image.")

    payload: list[tuple[str, bytes]] = []
    for file in files:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"{file.filename} was empty.")
        payload.append((file.filename or "uploaded_image", content))

    try:
        predictions = model_service.predict_many(payload)
    except UnconfiguredModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive API wrapper
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictResponse(
        model_status="ready",
        results=[
            PredictionResult(
                filename=pred.filename,
                probability=pred.probability,
                prediction=pred.prediction,
            )
            for pred in predictions
        ],
    )
