from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.llm_service import llm_service
from backend.app.model_service import UnconfiguredModelError, model_service
from backend.app.schemas import PredictResponse, PredictionResult


router = APIRouter(tags=["inference"])


@router.get("/health")
def health() -> dict[str, str | bool | list[str]]:
    return {
        "status": "ok",
        "model_ready": model_service.is_ready,
        "llm_status": llm_service.status,
        "llm_key_env_var": llm_service.api_key_env_var or "",
        "llm_checked_key_env_vars": list(llm_service.checked_key_env_vars),
        "llm_model": llm_service.model,
    }


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

    results: list[PredictionResult] = []
    for pred in predictions:
        summary = llm_service.summarize_prediction(
            filename=pred.filename,
            probability=pred.probability,
            prediction=pred.prediction,
            threshold=model_service.threshold,
        )
        results.append(
            PredictionResult(
                filename=pred.filename,
                probability=pred.probability,
                prediction=pred.prediction,
                summary=summary.text,
                summary_source=summary.source,
            )
        )

    return PredictResponse(
        model_status="ready",
        llm_status=llm_service.status,
        results=results,
    )
