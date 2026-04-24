from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageOps
from torchvision.models import (
    densenet121,
    efficientnet_b0,
    resnet18,
)

try:
    import cv2

    HAVE_CV2 = True
except ImportError:
    cv2 = None
    HAVE_CV2 = False

from backend.app.config import DEFAULT_MODEL_DIR


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CROP_MARGIN = 20
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


@dataclass
class BackendPrediction:
    filename: str
    probability: float
    prediction: int


class UnconfiguredModelError(RuntimeError):
    """Raised when the backend model has not been wired yet."""


class InvalidImageError(ValueError):
    """Raised when an uploaded image is not suitable for X-ray inference."""


class WeightedCNNEnsemble(nn.Module):
    def __init__(
        self,
        dropout: float = 0.3,
        densenet_weight: float = 0.30,
        efnet_weight: float = 0.35,
        resnet_weight: float = 0.35,
    ) -> None:
        super().__init__()
        self.dnet = densenet121(weights=None)
        dn_in = self.dnet.classifier.in_features
        self.dnet.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(dn_in, 1))

        self.efnet = efficientnet_b0(weights=None)
        ef_in = self.efnet.classifier[1].in_features
        self.efnet.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(ef_in, 1))

        self.resnet = resnet18(weights=None)
        rs_in = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(rs_in, 1))

        self.densenet_weight = densenet_weight
        self.efnet_weight = efnet_weight
        self.resnet_weight = resnet_weight

    def forward_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dn_logits = self.dnet(x).squeeze(1)
        ef_logits = self.efnet(x).squeeze(1)
        rs_logits = self.resnet(x).squeeze(1)
        return dn_logits, ef_logits, rs_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dn_logits, ef_logits, rs_logits = self.forward_logits(x)
        return (
            self.densenet_weight * dn_logits
            + self.efnet_weight * ef_logits
            + self.resnet_weight * rs_logits
        )


class ModelService:
    """
    Inference service for the current production checkpoint.

    Defaults are wired to the 101 weighted CNN ensemble.
    Override with Render env vars if we swap to a different production model:

    - MODEL_DENSENET_CHECKPOINT_PATH
    - MODEL_EFNET_CHECKPOINT_PATH
    - MODEL_RESNET_CHECKPOINT_PATH
    - MODEL_THRESHOLD
    - MODEL_IMG_SIZE
    - MODEL_DROPOUT
    - MODEL_DENSENET_WEIGHT
    - MODEL_EFNET_WEIGHT
    - MODEL_RESNET_WEIGHT
    """

    def __init__(self) -> None:
        self.densenet_checkpoint_path = Path(
            os.getenv(
                "MODEL_DENSENET_CHECKPOINT_PATH",
                str(DEFAULT_MODEL_DIR / "101_weighted_cnn_preprocessed_ensemble_densenet121_best.pt"),
            )
        )
        self.efnet_checkpoint_path = Path(
            os.getenv(
                "MODEL_EFNET_CHECKPOINT_PATH",
                str(DEFAULT_MODEL_DIR / "101_weighted_cnn_preprocessed_ensemble_efficientnet_b0_best.pt"),
            )
        )
        self.resnet_checkpoint_path = Path(
            os.getenv(
                "MODEL_RESNET_CHECKPOINT_PATH",
                str(DEFAULT_MODEL_DIR / "101_weighted_cnn_preprocessed_ensemble_resnet18_best.pt"),
            )
        )
        self.threshold = float(os.getenv("MODEL_THRESHOLD", "0.574242"))
        self.img_size = int(os.getenv("MODEL_IMG_SIZE", "384"))
        self.dropout = float(os.getenv("MODEL_DROPOUT", "0.3"))
        self.densenet_weight = float(os.getenv("MODEL_DENSENET_WEIGHT", "0.30"))
        self.efnet_weight = float(os.getenv("MODEL_EFNET_WEIGHT", "0.35"))
        self.resnet_weight = float(os.getenv("MODEL_RESNET_WEIGHT", "0.35"))
        self.device = self._pick_device()
        self.transform = T.Compose(
            [
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self._model: WeightedCNNEnsemble | None = None
        self._load_error: str | None = None
        self._is_loading = False
        self._load_lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def is_loading(self) -> bool:
        return self._is_loading

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _pick_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_model(self) -> WeightedCNNEnsemble:
        return WeightedCNNEnsemble(
            dropout=self.dropout,
            densenet_weight=self.densenet_weight,
            efnet_weight=self.efnet_weight,
            resnet_weight=self.resnet_weight,
        )

    def _try_load(self) -> None:
        missing = [
            path
            for path in (
                self.densenet_checkpoint_path,
                self.efnet_checkpoint_path,
                self.resnet_checkpoint_path,
            )
            if not path.exists()
        ]
        if missing:
            self._load_error = (
                "Checkpoint(s) not found: "
                + ", ".join(str(path) for path in missing)
                + ". Add the selected .pt files to backend/models or set the MODEL_*_CHECKPOINT_PATH env vars."
            )
            return

        try:
            self._is_loading = True
            model = self._build_model()
            densenet_state = torch.load(self.densenet_checkpoint_path, map_location=self.device)
            efnet_state = torch.load(self.efnet_checkpoint_path, map_location=self.device)
            resnet_state = torch.load(self.resnet_checkpoint_path, map_location=self.device)
            model.dnet.load_state_dict(densenet_state)
            model.efnet.load_state_dict(efnet_state)
            model.resnet.load_state_dict(resnet_state)
            model.to(self.device)
            model.eval()
            self._model = model
            self._load_error = None
        except Exception as exc:
            self._model = None
            self._load_error = (
                "Failed to load 101 ensemble checkpoints: "
                f"{exc}"
            )
        finally:
            self._is_loading = False

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._try_load()

    def _foreground_crop(self, gray_img: Image.Image) -> Image.Image:
        gray_np = np.asarray(gray_img)
        mask = gray_np > np.quantile(gray_np, 0.05)
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return gray_img
        x0 = max(0, int(xs.min()) - CROP_MARGIN)
        x1 = min(gray_np.shape[1], int(xs.max()) + CROP_MARGIN)
        y0 = max(0, int(ys.min()) - CROP_MARGIN)
        y1 = min(gray_np.shape[0], int(ys.max()) + CROP_MARGIN)
        return gray_img.crop((x0, y0, x1, y1))

    def _apply_clahe(self, gray_img: Image.Image) -> Image.Image:
        if HAVE_CV2:
            gray_np = np.asarray(gray_img)
            clahe = cv2.createCLAHE(
                clipLimit=CLAHE_CLIP_LIMIT,
                tileGridSize=CLAHE_TILE_GRID_SIZE,
            )
            return Image.fromarray(clahe.apply(gray_np)).convert("L")
        return ImageOps.autocontrast(gray_img)

    def _looks_like_chest_xray(self, gray_img: Image.Image) -> bool:
        gray_np = np.asarray(gray_img, dtype=np.uint8)
        if gray_np.ndim != 2:
            return False
        height, width = gray_np.shape
        if min(height, width) < 128:
            return False

        unique_levels = np.unique(gray_np).size
        if unique_levels < 32:
            return False

        hist = np.bincount(gray_np.ravel(), minlength=256).astype(np.float64)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        entropy = float(-(probs * np.log2(probs)).sum())
        if entropy < 4.5:
            return False

        return True

    def _validate_image(self, image_bytes: bytes) -> Image.Image:
        try:
            pil_image = Image.open(BytesIO(image_bytes))
            pil_image.load()
        except Exception as exc:
            raise InvalidImageError(
                "We could not read that file as an image. Please upload a frontal chest radiograph."
            ) from exc

        if "A" in pil_image.getbands():
            raise InvalidImageError(
                "This image has transparency and does not appear to be a chest X-ray. Please upload a frontal chest radiograph."
            )

        image = pil_image.convert("L")
        if not self._looks_like_chest_xray(image):
            raise InvalidImageError(
                "This image doesn't appear to be a chest X-ray. Please upload a frontal chest radiograph."
            )
        image = self._foreground_crop(image)
        image = self._apply_clahe(image)
        return image.convert("RGB")

    def _prepare_tensor(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict_many(self, files: list[tuple[str, bytes]]) -> list[BackendPrediction]:
        self.ensure_loaded()
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
