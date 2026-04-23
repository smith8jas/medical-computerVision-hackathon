from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from backend.app.config import load_local_env


load_local_env()

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency fallback
    OpenAI = None  # type: ignore[assignment]


@dataclass
class FindingSummary:
    text: str | None
    source: str


class LLMService:
    """
    Optional OpenAI-powered explanation service.

    The LLM receives the classifier output and, when available, a resized copy
    of the uploaded X-ray. It should describe cautious visual observations only,
    not make an independent diagnosis.
    """

    KEY_ENV_VARS = (
        "OPENAI_API_KEY",
        "OPENAI_KEY",
        "OPENAI_APIKEY",
        "OPEN_AI_API_KEY",
        "OPEN_API_KEY",
        "CHATGPT_API_KEY",
        "GPT_API_KEY",
    )

    def __init__(self) -> None:
        self.enabled = os.getenv("LLM_SUMMARIES_ENABLED", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key, self.api_key_env_var = self._load_api_key()
        self.client = self._build_client()

    def _load_api_key(self) -> tuple[str | None, str | None]:
        load_local_env()
        for key_name in self.KEY_ENV_VARS:
            value = os.getenv(key_name)
            if value and value.strip():
                return value.strip(), key_name
        return None, None

    def _build_client(self):
        if not self.enabled or not self.api_key or OpenAI is None:
            return None
        return OpenAI(api_key=self.api_key)

    def refresh(self) -> None:
        """Pick up env/config changes without leaking the API key."""
        self.enabled = os.getenv("LLM_SUMMARIES_ENABLED", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key, api_key_env_var = self._load_api_key()
        if api_key != self.api_key or api_key_env_var != self.api_key_env_var or self.client is None:
            self.api_key = api_key
            self.api_key_env_var = api_key_env_var
            self.client = self._build_client()

    @property
    def is_ready(self) -> bool:
        self.refresh()
        return self.client is not None

    @property
    def status(self) -> str:
        self.refresh()
        if not self.enabled:
            return "disabled"
        if OpenAI is None:
            return "missing_openai_package"
        if not self.api_key:
            return "missing_api_key_expected_OPENAI_API_KEY"
        if self.client is None:
            return "unavailable"
        return "ready"

    @property
    def checked_key_env_vars(self) -> tuple[str, ...]:
        return self.KEY_ENV_VARS

    def _image_data_url(self, image_bytes: bytes) -> str:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((768, 768))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def summarize_prediction(
        self,
        *,
        filename: str,
        probability: float,
        prediction: int,
        threshold: float,
        image_bytes: bytes | None = None,
    ) -> FindingSummary:
        self.refresh()
        if not self.is_ready or self.client is None:
            return FindingSummary(text=None, source=self.status)

        label = "cardiomegaly likely" if prediction == 1 else "no cardiomegaly detected"
        prompt = (
            "Create a concise clinician-facing explanation of a cardiomegaly classifier result.\n"
            "Important constraints:\n"
            "- Do not claim to make a diagnosis or replace radiologist review.\n"
            "- If an image is provided, describe only visible chest X-ray features you can cautiously observe.\n"
            "- If the image quality/projection limits assessment, say so.\n"
            "- Relate the classifier probability to possible visual context, such as cardiac silhouette size, cardiomediastinal contour, projection/rotation, and lung field visibility.\n"
            "- Do not invent measurements, cardiothoracic ratio values, or findings you cannot see.\n"
            "- Recommend clinician review of the image and clinical context.\n"
            "- Keep it to 3 to 4 short sentences.\n\n"
            f"Filename: {filename}\n"
            f"Classifier output: {label}\n"
            f"Probability: {probability:.4f}\n"
            f"Decision threshold: {threshold:.4f}\n"
        )

        try:
            content: list[dict[str, str]] = [{"type": "input_text", "text": prompt}]
            source = "openai"
            if image_bytes is not None:
                content.append(
                    {
                        "type": "input_image",
                        "image_url": self._image_data_url(image_bytes),
                        "detail": "low",
                    }
                )
                source = "openai_vision"

            response = self.client.responses.create(
                model=self.model,
                instructions=(
                    "You are a medical AI explanation assistant for clinician review. "
                    "Use cautious language, distinguish model probability from visual observation, "
                    "and avoid definitive diagnosis."
                ),
                input=[{"role": "user", "content": content}],
                max_output_tokens=220,
            )
            return FindingSummary(text=response.output_text.strip(), source=source)
        except Exception:  # pragma: no cover - external API failure
            return FindingSummary(text=None, source="error")


llm_service = LLMService()
