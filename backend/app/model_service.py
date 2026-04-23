from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.models import densenet121, efficientnet_b0

from backend.app.config import DEFAULT_MODEL_DIR


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class BackendPrediction:
    filename: str
    probability: float
    prediction: int


class UnconfiguredModelError(RuntimeError):
    """Raised when the backend model has not been wired yet."""


class EfficientDenseEnsemble(nn.Module):
    def __init__(self, dropout: float = 0.3, efnet_weight: float = 0.5) -> None:
        super().__init__()
        self.efnet = efficientnet_b0(weights=None)
        ef_in = self.efnet.classifier[1].in_features
        self.efnet.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(ef_in, 1))

        self.efnet_weight = efnet_weight

        self.dnet = densenet121(weights=None)
        dn_in = self.dnet.classifier.in_features
        self.dnet.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(dn_in, 1))

    def forward_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ef_logits = self.efnet(x).squeeze(1)
        dn_logits = self.dnet(x).squeeze(1)
        return ef_logits, dn_logits

    def forward(self, x: torch.Tensor, efnet_weight: float | None = None) -> torch.Tensor:
        ef_logits, dn_logits = self.forward_logits(x)
        weight = self.efnet_weight if efnet_weight is None else efnet_weight
        return weight * ef_logits + (1.0 - weight) * dn_logits


class ModelService:
    """
    Inference service for the current production checkpoint.

    Defaults are wired to the 05.7.1 EfficientNet-B0 + DenseNet121 ensemble.
    Override with Render env vars if we swap to a different production model:

    - MODEL_CHECKPOINT_PATH
    - MODEL_THRESHOLD
    - MODEL_IMG_SIZE
    - MODEL_DROPOUT
    - MODEL_EFNET_WEIGHT
    """

    def __init__(self) -> None:
        self.checkpoint_path = Path(
            os.getenv(
                "MODEL_CHECKPOINT_PATH",
                str(DEFAULT_MODEL_DIR / "05_7_1_efnet_densenet_joint_best.pt"),
            )
        )
        self.threshold = float(os.getenv("MODEL_THRESHOLD", "0.638586"))
        self.img_size = int(os.getenv("MODEL_IMG_SIZE", "256"))
        self.dropout = float(os.getenv("MODEL_DROPOUT", "0.3"))
        self.efnet_weight = float(os.getenv("MODEL_EFNET_WEIGHT", "0.5"))
        self.device = self._pick_device()
        self.transform = T.Compose(
            [
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self._model: EfficientDenseEnsemble | None = None
        self._load_error: str | None = None
        self._try_load()

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def _pick_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_model(self) -> EfficientDenseEnsemble:
        return EfficientDenseEnsemble(
            dropout=self.dropout,
            efnet_weight=self.efnet_weight,
        )

    def _try_load(self) -> None:
        if not self.checkpoint_path.exists():
            self._load_error = (
                f"Checkpoint not found at {self.checkpoint_path}. "
                "Add the selected .pt file to backend/models or set MODEL_CHECKPOINT_PATH."
            )
            return

        try:
            model = self._build_model()
            state = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            self._model = model
            self._load_error = None
        except Exception as exc:
            self._model = None
            self._load_error = (
                f"Failed to load checkpoint {self.checkpoint_path.name}: {exc}"
            )

    def _validate_image(self, image_bytes: bytes) -> Image.Image:
        return Image.open(BytesIO(image_bytes)).convert("RGB")

    def _prepare_tensor(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict_many(self, files: list[tuple[str, bytes]]) -> list[BackendPrediction]:
        if not self.is_ready or self._model is None:
            detail = self._load_error or (
                "Inference model is not configured yet. "
                "Plug the final selected checkpoint into backend/app/model_service.py."
            )
            raise UnconfiguredModelError(detail)

        predictions: list[BackendPrediction] = []
        with torch.no_grad():
            for filename, image_bytes in files:
                image = self._validate_image(image_bytes)
                tensor = self._prepare_tensor(image)
                logits = self._model(tensor)
                probability = float(torch.sigmoid(logits).cpu().item())
                prediction = int(probability >= self.threshold)
                predictions.append(
                    BackendPrediction(
                        filename=filename,
                        probability=probability,
                        prediction=prediction,
                    )
                )
        return predictions


model_service = ModelService()
