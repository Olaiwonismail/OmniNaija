from __future__ import annotations

import json
import re
import time
from typing import Any

import requests
from google import genai

from config import Config

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"


def _extract_section(prompt: str, start_marker: str, end_marker: str | None = None) -> str:
    pattern = re.escape(start_marker) + r"\s*(.*)"
    if end_marker:
        pattern += r"(?=\n" + re.escape(end_marker) + r"|\Z)"
    match = re.search(pattern, prompt, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_json_value(prompt: str, label: str) -> Any:
    section = _extract_section(prompt, label)
    if not section:
        return {}

    decoder = json.JSONDecoder()
    for index, char in enumerate(section):
        if char not in "[{":
            continue
        try:
            parsed, _offset = decoder.raw_decode(section[index:])
        except json.JSONDecodeError:
            continue
        return parsed

    return {}


def _infer_intent_from_message(message: str) -> tuple[str, float, str | None]:
    normalized = message.lower()

    def has(*keywords: str) -> bool:
        return any(keyword in normalized for keyword in keywords)

    if has("power outage", "power cuts", "outage", "generator", "backup power", "ups", "inverter", "charging", "battery backup"):
        if has("work", "laptop", "client", "meeting", "developer", "remote"):
            return "remote_work_setup", 0.92, "cafes_with_power"
        return "home_power_resilience", 0.9, "power_backup"

    if has("work out", "working out", "gym", "fitness", "exercise", "training", "workout"):
        return "fitness_journey", 0.9, "gyms_and_fitness"

    if has("cook", "cooking", "recipe", "recipes", "ingredients", "kitchen", "meal"):
        return "cooking_exploration", 0.88, "kitchen_tools_and_recipes"

    if has("book", "books", "reading", "novel", "ebook", "audiobook", "library"):
        return "reading_habit", 0.88, "books_and_reading"

    if has("baby", "pregnan", "child", "parent", "family needs", "new born", "newborn"):
        return "baby_family_prep", 0.9, "baby_and_family"

    if has("owambe", "party", "wedding", "aso ebi", "gele", "makeup", "decoration", "celebration"):
        return "owambe_event_prep", 0.9, "events_and_owambe"

    if has("laptop", "client work", "meetings", "remote work", "work from home"):
        return "remote_work_setup", 0.85, "cafes_with_power"

    return "general_browsing", 0.45, None


def _fallback_intent_json(prompt: str) -> str:
    message = _extract_section(prompt, "Chat message:", "Optional cart/history:")
    intent, confidence, bridge_category = _infer_intent_from_message(message)
    return json.dumps(
        {
            "intent": intent,
            "confidence": confidence,
            "bridge_category": bridge_category,
        },
        ensure_ascii=False,
    )


def _fallback_review_json(prompt: str) -> str:
    product_metadata = _extract_json_value(prompt, "Product metadata:")
    if not isinstance(product_metadata, dict):
        product_metadata = {}
    title = str(product_metadata.get("title") or product_metadata.get("name") or "this product").strip()
    category = str(product_metadata.get("category") or "product").strip().lower()
    price = product_metadata.get("price")
    rating = 4
    if isinstance(price, (int, float)) and price > 100:
        rating = 3
    review = (
        f"I used {title} for a while and it is solid for everyday use. "
        f"It does the job well and feels practical, especially for someone buying in the {category} space. "
        f"The only thing I would say is make sure the price makes sense for you, but overall it is worth it."
    )
    return json.dumps({"rating": rating, "review": review}, ensure_ascii=False)


def _fallback_recommendation_text(prompt: str) -> str:
    persona = _extract_json_value(prompt, "Persona JSON:")
    products = _extract_json_value(prompt, "Retrieved products:")
    locations = _extract_json_value(prompt, "Retrieved locations:")

    if not isinstance(persona, dict):
        persona = {}

    persona_name = str(persona.get("name") or "this user").strip()
    product_name = "the top product"
    if isinstance(products, list) and products:
        first_product = products[0]
        if isinstance(first_product, dict):
            product_name = str(first_product.get("title") or first_product.get("product_id") or product_name).strip()

    location_name = None
    if isinstance(locations, list) and locations:
        first_location = locations[0]
        if isinstance(first_location, dict):
            location_name = str(first_location.get("name") or first_location.get("venue_id") or "").strip()

    lines = [
        f"1. Best recommendation: {product_name}",
        f"2. Why it fits: it is the safest practical choice for {persona_name} based on the retrieved options and keeps value for money in mind.",
        "3. Tradeoffs or cautions: if the user needs a more specific feature set, they should compare the top few options before buying.",
    ]
    if location_name:
        lines.append(f"Bridge suggestion: {location_name}")
    return "\n".join(lines)


def _local_fallback_text(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if '"intent"' in prompt_lower and 'bridge_category' in prompt_lower:
        return _fallback_intent_json(prompt)
    if '"rating"' in prompt_lower and '"review"' in prompt_lower:
        return _fallback_review_json(prompt)
    return _fallback_recommendation_text(prompt)


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
        timeout=150,
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
        for attempt in range(2):
            try:
                text = runner(prompt)
                if text:
                    return text, provider_name
                raise RuntimeError(f"{provider_name} returned empty text")
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(2)

    return _local_fallback_text(prompt), "fallback"


def generate_text(prompt: str, timeout: int | None = None) -> tuple[str, str]:
    """
    Timeout-safe wrapper around `generate_text_with_fallback`.
    Runs the LLM calls in a thread and falls back to local text if the
    external providers don't return within `timeout` seconds.
    """
    if timeout is None or timeout <= 0:
        try:
            # No explicit timeout: let external providers take their time.
            return generate_text_with_fallback(prompt)
        except Exception:
            # Providers failed — use local fallback
            return _local_fallback_text(prompt), "fallback"

    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(generate_text_with_fallback, prompt)
        try:
            return fut.result(timeout=timeout)
        except TimeoutError:
            # external providers too slow — use local fallback
            return _local_fallback_text(prompt), "fallback-timeout"
        except Exception:
            return _local_fallback_text(prompt), "fallback-exception"