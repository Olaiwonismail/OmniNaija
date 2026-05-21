import json
from pathlib import Path
from typing import Any

from config import Config

DEMO_CACHE_DIR = Path(__file__).resolve().parent / "demo_cache"
_DEMO_CACHE_INDEX: dict[str, dict[str, Any]] | None = None


def _canonical_request_key(endpoint: str, request_payload: dict[str, Any]) -> str:
    return json.dumps(
        {
            "endpoint": endpoint,
            "request": request_payload,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def load_demo_cache_index() -> dict[str, dict[str, Any]]:
    global _DEMO_CACHE_INDEX

    if _DEMO_CACHE_INDEX is not None:
        return _DEMO_CACHE_INDEX

    index: dict[str, dict[str, Any]] = {}
    if not DEMO_CACHE_DIR.exists():
        _DEMO_CACHE_INDEX = index
        return index

    for path in sorted(DEMO_CACHE_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        endpoint = payload.get("endpoint")
        request = payload.get("request")
        response = payload.get("response")
        if not endpoint or not isinstance(request, dict):
            continue

        index[_canonical_request_key(endpoint, request)] = {
            "path": str(path),
            "endpoint": endpoint,
            "request": request,
            "response": response,
        }

    _DEMO_CACHE_INDEX = index
    return index


def get_demo_cache_response(endpoint: str, request_payload: dict[str, Any]) -> dict[str, Any] | None:
    if not Config.DEMO_MODE:
        return None

    index = load_demo_cache_index()
    cached = index.get(_canonical_request_key(endpoint, request_payload))
    if not cached:
        return None
    response = cached.get("response")
    return response if isinstance(response, dict) else None
