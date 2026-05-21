from __future__ import annotations

from typing import Any

import requests
from google import genai

from config import Config

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"


def _groq_text(prompt: str, model: str | None = None) -> str:
    if not Config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")

    response = requests.post(
        GROQ_CHAT_COMPLETIONS_URL,
        headers={
            "Authorization": f"Bearer {Config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or Config.GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=120,
    )
    response.raise_for_status()

    payload: dict[str, Any] = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("Groq response did not include choices")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n".join(parts).strip()

    raise RuntimeError("Groq response did not include text content")


def _gemini_text(prompt: str, model: str | None = None) -> str:
    if not Config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=Config.GEMINI_API_KEY)
    response = client.models.generate_content(model=model or Config.GEMINI_MODEL, contents=prompt)
    return (getattr(response, "text", "") or "").strip()


def generate_text_with_fallback(prompt: str) -> tuple[str, str]:
    last_error: Exception | None = None
    for provider_name, runner in (("groq", _groq_text), ("gemini", _gemini_text)):
        try:
            text = runner(prompt)
            if text:
                return text, provider_name
            raise RuntimeError(f"{provider_name} returned empty text")
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"All LLM providers failed: {last_error}") from last_error