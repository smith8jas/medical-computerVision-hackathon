from __future__ import annotations

import os
from dataclasses import dataclass

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

    The LLM receives only structured classifier output. It does not inspect the
    X-ray image, so summaries must stay framed as model-output explanations.
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

    def summarize_prediction(
        self,
        *,
        filename: str,
        probability: float,
        prediction: int,
        threshold: float,
    ) -> FindingSummary:
        self.refresh()
        if not self.is_ready or self.client is None:
            return FindingSummary(text=None, source=self.status)

        label = "cardiomegaly likely" if prediction == 1 else "no cardiomegaly detected"
        prompt = (
            "Create a concise, patient-safe explanation of a cardiomegaly classifier result.\n"
            "Important constraints:\n"
            "- Do not claim to be making a diagnosis.\n"
            "- Do not describe visual findings that were not provided.\n"
            "- Explain that the statement is based on the model probability.\n"
            "- Recommend clinician review of the image and clinical context.\n"
            "- Keep it to 2 short sentences.\n\n"
            f"Filename: {filename}\n"
            f"Classifier output: {label}\n"
            f"Probability: {probability:.4f}\n"
            f"Decision threshold: {threshold:.4f}\n"
        )

        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=(
                    "You are a medical AI explanation assistant. Write cautious, "
                    "clear support text for clinicians reviewing a chest X-ray AI result."
                ),
                input=prompt,
                max_output_tokens=140,
            )
            return FindingSummary(text=response.output_text.strip(), source="openai")
        except Exception:  # pragma: no cover - external API failure
            return FindingSummary(text=None, source="error")


llm_service = LLMService()
