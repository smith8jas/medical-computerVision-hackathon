from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image


@dataclass
class BackendPrediction:
    filename: str
    probability: float
    prediction: int


class UnconfiguredModelError(RuntimeError):
    """Raised when the backend model has not been wired yet."""


class ModelService:
    """
    Placeholder inference service.

    The training notebooks still own model experimentation. Once the team picks
    the final checkpoint + architecture, wire the real preprocessing/model
    loading into this service and keep the HTTP layer unchanged.
    """

    def __init__(self) -> None:
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def _validate_image(self, image_bytes: bytes) -> Image.Image:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image

    def predict_many(self, files: list[tuple[str, bytes]]) -> list[BackendPrediction]:
        if not self.is_ready:
            raise UnconfiguredModelError(
                "Inference model is not configured yet. Plug the final selected "
                "checkpoint into backend/app/model_service.py."
            )

        predictions: list[BackendPrediction] = []
        for filename, image_bytes in files:
            _ = self._validate_image(image_bytes)
            predictions.append(
                BackendPrediction(
                    filename=filename,
                    probability=0.0,
                    prediction=0,
                )
            )
        return predictions


model_service = ModelService()
