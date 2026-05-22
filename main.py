import json
import re
import uuid
from pathlib import Path
from typing import Any

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import Config
from demo_cache import get_demo_cache_response
from llm import generate_text_with_fallback
from agent.graph import (
    understand_user,
    retrieve_products,
    should_bridge,
    retrieve_locations,
    compose_response,
    build_retrieval_query,
)

app = FastAPI()

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
CHROMA_COLLECTION = "amazon_products"
MAX_SESSION_MESSAGES = 12


class SimulateRequest(BaseModel):
    persona_description: Any = Field(..., description="Persona JSON or free-form persona description")
    product_id: str = Field(..., description="Product identifier or ASIN")
    demo_mode: bool | None = Field(None, description="Use cached demo responses when true")


class SimulateResponse(BaseModel):
    rating: int
    review: str


class GraphSimulateRequest(BaseModel):
    persona_description: Any = Field(..., description="Persona JSON or free-form persona description")
    chat_message: str = Field(..., description="User chat message or intent expression")


class GraphSimulateResponse(BaseModel):
    recommendation: str
    debug: dict[str, Any]


def load_prompt_template(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Prompt template not found: {name}")
    return path.read_text(encoding="utf-8")


def parse_persona(persona_description: Any) -> dict[str, Any]:
    if isinstance(persona_description, dict):
        return persona_description

    if isinstance(persona_description, str):
        text = persona_description.strip()
        if not text:
            raise HTTPException(status_code=422, detail="persona_description cannot be empty")
        if text.startswith("{") or text.startswith("["):
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) else {"persona": parsed}
            except json.JSONDecodeError:
                pass
        return {"description": text}

    raise HTTPException(status_code=422, detail="persona_description must be a string or JSON object")


def get_collection():
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return client.get_collection(name=CHROMA_COLLECTION)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to open Chroma collection: {exc}") from exc


def lookup_product_metadata(product_id: str) -> dict[str, Any]:
    collection = get_collection()

    candidate_ids = [product_id]
    if ":" not in product_id:
        candidate_ids.append(product_id.strip())

    for candidate_id in candidate_ids:
        try:
            result = collection.get(ids=[candidate_id])
            if result.get("ids"):
                return normalize_chroma_record(result, 0, candidate_id)
        except Exception:
            pass

    try:
        result = collection.get(where={"asin": product_id})
        if result.get("ids"):
            return normalize_chroma_record(result, 0, product_id)
    except Exception:
        pass

    raise HTTPException(status_code=404, detail=f"Product not found for product_id={product_id}")


def normalize_chroma_record(result: dict[str, Any], index: int, fallback_id: str) -> dict[str, Any]:
    metadata = dict(result.get("metadatas", [{}])[index] or {})
    document = result.get("documents", [None])[index]
    normalized = {
        "product_id": result.get("ids", [fallback_id])[index] or fallback_id,
        "asin": metadata.get("asin") or fallback_id,
        "parent_asin": metadata.get("parent_asin") or metadata.get("asin") or fallback_id,
        "title": metadata.get("title") or "",
        "brand": metadata.get("brand") or "",
        "store": metadata.get("store") or "",
        "category": metadata.get("category") or "",
        "price": metadata.get("price"),
        "avg_rating": metadata.get("avg_rating"),
        "review_count": metadata.get("review_count"),
        "features": safe_json_load(metadata.get("features")),
        "details": safe_json_load(metadata.get("details")),
        "images": safe_json_load(metadata.get("images")),
        "bundled_reviews": safe_json_load(metadata.get("bundled_reviews")),
        "document": document or "",
    }
    return normalized


def safe_json_load(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def render_review_prompt(persona: dict[str, Any], product: dict[str, Any]) -> str:
    template = load_prompt_template("review_simulation_prompt.txt")
    return template.replace("{{persona_json}}", json.dumps(persona, ensure_ascii=False, indent=2)).replace(
        "{{product_metadata}}", json.dumps(product, ensure_ascii=False, indent=2)
    )


def extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("Empty model response")

    cleaned = text.strip()
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


def request_review_from_gemini(prompt: str) -> dict[str, Any]:
    response_text, _provider = generate_text(prompt)

    try:
        return extract_json_object(response_text)
    except ValueError:
        repair_prompt = (
            prompt
            + "\n\nIMPORTANT: Return only a valid JSON object with keys rating and review. No markdown, no code fences."
        )
        retry_text, _provider = generate_text(repair_prompt)
        try:
            return extract_json_object(retry_text)
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "Gemini returned unparseable output",
                    "raw_response": response_text,
                    "retry_response": retry_text,
                },
            ) from exc


@app.get("/")
def read_root():
    return {"message": "OmniNaija API is running"}


@app.post("/simulate", response_model=SimulateResponse)
def simulate_review(payload: SimulateRequest):
    request_payload = payload.model_dump(exclude={"demo_mode"})
    demo_mode_enabled = Config.DEMO_MODE or bool(payload.demo_mode)
    if demo_mode_enabled:
        cached = get_demo_cache_response("/simulate", request_payload)
        if cached:
            return SimulateResponse(**cached)

    persona = parse_persona(payload.persona_description)
    product = lookup_product_metadata(payload.product_id)
    prompt = render_review_prompt(persona, product)

    result = request_review_from_gemini(prompt)
    try:
        rating = int(result["rating"])
        review = str(result["review"]).strip()
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Gemini response missing rating/review: {exc}") from exc

    if rating < 1 or rating > 5:
        raise HTTPException(status_code=502, detail=f"Gemini returned invalid rating: {rating}")

    return SimulateResponse(rating=rating, review=review)


def run_graph_simulation(persona: dict[str, Any], message: str) -> dict[str, Any]:
    return run_recommendation_flow(persona, message, chat_history=[{"role": "user", "content": message}])


def run_recommendation_flow(
    persona: dict[str, Any],
    message: str,
    chat_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    history = chat_history or [{"role": "user", "content": message}]
    state = understand_user(message, persona, chat_history=history)

    retrieval_query = build_retrieval_query(history, message)
    products = retrieve_products(retrieval_query, top_k=5)
    state["products"] = products

    bridged = should_bridge(state)
    state["bridged"] = bridged
    if bridged:
        bridge_cat = state["intent"].get("bridge_category")
        locations = retrieve_locations(bridge_category=bridge_cat, top_k=3)
        state["locations"] = locations
    else:
        state["locations"] = []

    recommendation = compose_response(state, top_k_products=3)
    return {"recommendation": recommendation, "state": state}


def trim_session_history(history: list[dict[str, Any]], max_messages: int = MAX_SESSION_MESSAGES) -> list[dict[str, Any]]:
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


# Simple in-memory session store: { session_id: [ {role, content}, ... ] }
SESSION_HISTORIES: dict[str, list[dict[str, Any]]] = {}


class RecommendRequest(BaseModel):
    persona_description: Any = Field(..., description="Persona JSON or free-form persona description")
    message: str = Field(..., description="User message for this turn")
    session_id: str | None = Field(None, description="Optional session id for multi-turn history")
    demo_mode: bool | None = Field(None, description="Use cached demo responses when true")


class RecommendResponse(BaseModel):
    session_id: str
    recommendation: str
    intent: dict[str, Any] | None = None
    debug: dict[str, Any]
    history: list[dict[str, Any]]


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest):
    request_payload = payload.model_dump(exclude={"demo_mode"})
    demo_mode_enabled = Config.DEMO_MODE or bool(payload.demo_mode)
    if demo_mode_enabled:
        cached = get_demo_cache_response("/recommend", request_payload)
        if cached:
            cached_session_id = cached.get("session_id") or payload.session_id or str(uuid.uuid4())
            cached_history = cached.get("history") if isinstance(cached.get("history"), list) else []
            if cached_history:
                SESSION_HISTORIES[cached_session_id] = cached_history
            return RecommendResponse(**cached)

    persona = parse_persona(payload.persona_description)

    # Use provided session_id or create a new one
    sid = payload.session_id or str(uuid.uuid4())

    # Ensure session history exists
    hist = SESSION_HISTORIES.setdefault(sid, [])

    # Append incoming user message
    user_entry = {"role": "user", "content": payload.message}
    hist.append(user_entry)
    hist = trim_session_history(hist)

    # Run the graph simulation using full history
    result = run_recommendation_flow(persona, payload.message, chat_history=hist)
    state = result["state"]
    recommendation = result["recommendation"]

    # Append assistant response to history
    assistant_entry = {"role": "assistant", "content": recommendation}
    hist.append(assistant_entry)
    hist = trim_session_history(hist)

    SESSION_HISTORIES[sid] = hist

    debug = {
        "intent": state.get("intent"),
        "products": [p.get("product_id") for p in state.get("products", [])[:10]],
        "bridged": state.get("bridged"),
        "locations": [l.get("venue_id") for l in state.get("locations", [])],
    }

    return RecommendResponse(
        session_id=sid,
        recommendation=recommendation,
        intent=state.get("intent"),
        debug=debug,
        history=hist,
    )


@app.post("/graph_simulate", response_model=GraphSimulateResponse)
def graph_simulate(payload: GraphSimulateRequest):
    persona = parse_persona(payload.persona_description)
    result = run_graph_simulation(persona, payload.chat_message)
    debug = {
        "intent": result["state"].get("intent"),
        "products": [p.get("product_id") for p in result["state"].get("products", [])[:10]],
        "bridged": result["state"].get("bridged"),
        "locations": [l.get("venue_id") for l in result["state"].get("locations", [])],
    }
    return GraphSimulateResponse(recommendation=result["recommendation"], debug=debug)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)