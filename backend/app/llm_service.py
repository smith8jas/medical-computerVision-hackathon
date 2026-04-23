from __future__ import annotations

import os
from dataclasses import dataclass


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

    def __init__(self) -> None:
        self.enabled = os.getenv("LLM_SUMMARIES_ENABLED", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.enabled and self.api_key and OpenAI else None

    @property
    def is_ready(self) -> bool:
        return self.client is not None

    @property
    def status(self) -> str:
        if not self.enabled:
            return "disabled"
        if OpenAI is None:
            return "missing_openai_package"
        if not self.api_key:
            return "missing_api_key"
        if self.client is None:
            return "unavailable"
        return "ready"

    def summarize_prediction(
        self,
        *,
        filename: str,
        probability: float,
        prediction: int,
        threshold: float,
    ) -> FindingSummary:
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
