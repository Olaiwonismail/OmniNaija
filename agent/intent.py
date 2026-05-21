import json
import re
from pathlib import Path
from typing import Any

from llm import generate_text_with_fallback

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

DEFAULT_BRIDGE_CATEGORIES = {
    "remote_work_setup": "cafes_with_power",
    "fitness_journey": "gyms_and_fitness",
    "cooking_exploration": "kitchen_tools_and_recipes",
    "reading_habit": "books_and_reading",
    "baby_family_prep": "baby_and_family",
    "owambe_event_prep": "events_and_owambe",
    "home_power_resilience": "power_backup",
    "general_browsing": None,
}


def load_prompt_template(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {name}")
    return path.read_text(encoding="utf-8")


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Could not parse JSON from Gemini response")


def classify_user_intent(
    chat_message: str,
    persona: Any,
    cart_history: Any = None,
) -> dict[str, Any]:
    persona_value = _normalize_value(persona)
    cart_value = _normalize_value(cart_history)

    if isinstance(persona_value, str):
        persona_value = {"description": persona_value}

    template = load_prompt_template("intent_extraction_prompt.txt")
    prompt = (
        template.replace("{{persona_json}}", _stringify(persona_value))
        .replace("{{chat_message}}", chat_message)
        .replace("{{cart_history}}", _stringify(cart_value))
    )

    response_text, _provider = generate_text_with_fallback(prompt)

    try:
        result = _extract_json_object(response_text)
    except ValueError:
        repair_prompt = prompt + "\n\nIMPORTANT: Return only valid JSON with intent, confidence, and bridge_category."
        retry_text, _provider = generate_text_with_fallback(repair_prompt)
        result = _extract_json_object(retry_text)

    intent = str(result.get("intent", "general_browsing")).strip()
    bridge_category = result.get("bridge_category")
    confidence_value = result.get("confidence", 0.0)

    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError):
        confidence = 0.0

    if not intent:
        intent = "general_browsing"

    if intent == "general_browsing" and confidence >= 0.6:
        confidence = 0.59

    if confidence < 0:
        confidence = 0.0
    if confidence > 1:
        confidence = 1.0

    if intent in DEFAULT_BRIDGE_CATEGORIES:
        bridge_category = DEFAULT_BRIDGE_CATEGORIES[intent]
    elif bridge_category is None:
        bridge_category = None

    return {
        "intent": intent,
        "confidence": confidence,
        "bridge_category": bridge_category,
    }