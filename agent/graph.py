from .intent import classify_user_intent
from typing import Any, Dict
from pathlib import Path
import json

from config import Config
from google import genai


def understand_user(message: str, persona: Any, chat_history: Any = None, state: Dict[str, Any] | None = None) -> Dict[str, Any]:
	"""Node 1 — understand_user

	- Calls `classify_user_intent` to extract an intent object.
	- Writes `intent`, `persona`, and `chat_history` into `state` and returns it.

	This function is intentionally small and dependency-free so it can be
	used by the eventual LangGraph node implementation.
	"""
	if state is None:
		state = {}

	intent_obj = classify_user_intent(message, persona, cart_history=chat_history)

	state["intent"] = intent_obj
	state["persona"] = persona
	state["chat_history"] = chat_history

	return state


def retrieve_products(query: str, top_k: int = 5, persist_dir: str = "chroma_db", collection_name: str = "amazon_products") -> list[Dict[str, Any]]:
	"""Node 2 — retrieve_products

	Performs a semantic search against a local ChromaDB persistent collection using
	a SentenceTransformer embedding. Returns a list of normalized product dicts.
	"""
	import chromadb
	from sentence_transformers import SentenceTransformer

	model_name = "all-MiniLM-L6-v2"
	model = SentenceTransformer(model_name)

	client = chromadb.PersistentClient(path=persist_dir)
	collection = client.get_collection(name=collection_name)

	embedding = model.encode(query)
	result = collection.query(query_embeddings=[embedding], n_results=top_k)

	normalized_results = []
	ids = result.get("ids", [])
	metadatas = result.get("metadatas", [])
	documents = result.get("documents", [])

	rows = max(len(ids[0]) if ids else 0, len(metadatas[0]) if metadatas else 0)
	for i in range(rows):
		doc_id = ids[0][i] if ids and len(ids[0]) > i else None
		metadata = metadatas[0][i] if metadatas and len(metadatas[0]) > i else {}
		document = documents[0][i] if documents and len(documents[0]) > i else ""

		def _safe_json_load(v):
			if v is None:
				return None
			if isinstance(v, (dict, list, int, float, bool)):
				return v
			if isinstance(v, str):
				t = v.strip()
				if not t:
					return v
				try:
					import json

					return json.loads(t)
				except Exception:
					return v
			return v

		normalized = {
			"product_id": doc_id or metadata.get("asin"),
			"asin": metadata.get("asin"),
			"parent_asin": metadata.get("parent_asin") or metadata.get("asin"),
			"title": metadata.get("title") or "",
			"brand": metadata.get("brand") or "",
			"store": metadata.get("store") or "",
			"category": metadata.get("category") or "",
			"price": metadata.get("price"),
			"avg_rating": metadata.get("avg_rating"),
			"review_count": metadata.get("review_count"),
			"features": _safe_json_load(metadata.get("features")),
			"details": _safe_json_load(metadata.get("details")),
			"images": _safe_json_load(metadata.get("images")),
			"bundled_reviews": _safe_json_load(metadata.get("bundled_reviews")),
			"document": document or "",
		}
		normalized_results.append(normalized)

	return normalized_results


def should_bridge(state: Dict[str, Any], threshold: float = 0.6) -> bool:
	"""Return True if the intent confidence meets or exceeds the threshold."""
	intent = state.get("intent") if isinstance(state, dict) else None
	if not intent:
		return False
	try:
		conf = float(intent.get("confidence", 0))
	except Exception:
		conf = 0.0
	return conf >= float(threshold)


def retrieve_locations(bridge_category: str | None = None, top_k: int = 5, persist_dir: str = "chroma_db", collection_name: str = "venue_locations") -> list[Dict[str, Any]]:
	"""Node 4 — retrieve_locations

	Uses semantic search over the `venue_locations` Chroma collection. If
	`bridge_category` is provided, builds a focused natural-language query.
	"""
	import chromadb
	from sentence_transformers import SentenceTransformer
	import json

	model_name = "all-MiniLM-L6-v2"
	model = SentenceTransformer(model_name)

	client = chromadb.PersistentClient(path=persist_dir)
	collection = client.get_collection(name=collection_name)

	if bridge_category:
		# Map some known bridge categories into friendly search phrases
		mapping = {
			"cafes_with_power": "cafes with generator and reliable power, good for remote work",
			"remote_work_setup": "cafes or coworking spaces with power backup and WiFi",
			"gyms_nearby": "gyms and fitness centers nearby",
		}
		query_text = mapping.get(bridge_category, bridge_category.replace("_", " "))
	else:
		query_text = "popular cafes and coworking spaces"

	embedding = model.encode(query_text)
	result = collection.query(query_embeddings=[embedding], n_results=top_k)

	normalized_results = []
	ids = result.get("ids", [])
	metadatas = result.get("metadatas", [])
	documents = result.get("documents", [])

	rows = max(len(ids[0]) if ids else 0, len(metadatas[0]) if metadatas else 0)
	for i in range(rows):
		doc_id = ids[0][i] if ids and len(ids[0]) > i else None
		metadata = metadatas[0][i] if metadatas and len(metadatas[0]) > i else {}
		document = documents[0][i] if documents and len(documents[0]) > i else ""

		normalized = {
			"venue_id": doc_id,
			"name": metadata.get("name"),
			"city": metadata.get("city"),
			"area": metadata.get("area"),
			"has_generator": bool(metadata.get("has_generator")),
			"has_wifi": bool(metadata.get("has_wifi")),
			"family_friendly": bool(metadata.get("family_friendly")),
			"stars": metadata.get("stars"),
			"categories": metadata.get("categories"),
			"document": document,
		}
		normalized_results.append(normalized)

	return normalized_results


def compose_response(state: Dict[str, Any], top_k_products: int = 3) -> str:
	"""Node 5 — compose_response

	Renders the `prompts/recommender_prompt.txt` template and calls Gemini
	to generate a plain-text recommendation. Returns the assistant text.
	"""
	prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
	prompt_path = prompts_dir / "recommender_prompt.txt"
	if not prompt_path.exists():
		raise RuntimeError(f"Prompt template not found: {prompt_path}")

	template = prompt_path.read_text(encoding="utf-8")

	persona = state.get("persona") or {}
	chat_history = state.get("chat_history") or []
	products = state.get("products") or []
	locations = state.get("locations") or []

	retrieved_products = json.dumps(products[:top_k_products], ensure_ascii=False, indent=2)
	retrieved_locations = json.dumps(locations[:top_k_products], ensure_ascii=False, indent=2)

	prompt = (
		template.replace("{{persona_json}}", json.dumps(persona, ensure_ascii=False, indent=2))
		.replace("{{chat_history}}", json.dumps(chat_history, ensure_ascii=False, indent=2))
		.replace("{{retrieved_products}}", retrieved_products)
	)

	# Attach locations if present
	if "Retrieved products:" not in prompt and retrieved_products:
		prompt += f"\n\nRetrieved locations:\n{retrieved_locations}"

	if not Config.GEMINI_API_KEY:
		raise RuntimeError("GEMINI_API_KEY is not configured in the environment")

	client = genai.Client(api_key=Config.GEMINI_API_KEY)
	response = client.models.generate_content(model=Config.GEMINI_MODEL, contents=prompt)
	text = getattr(response, "text", "") or ""
	return text.strip()
