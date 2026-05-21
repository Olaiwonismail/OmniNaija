import os
import sys

from main import parse_persona
from agent.graph import (
    understand_user,
    retrieve_products,
    should_bridge,
    retrieve_locations,
    compose_response,
)

from agent.intent import classify_user_intent
from config import Config


def run():
    persona = {
        "name": "Tobi",
        "age": 27,
        "location": "Yaba, Lagos",
        "occupation": "freelance developer",
        "budget": "careful with money",
        "traits": ["budget-conscious", "likes practical gadgets", "hates anything without USB-C"],
    }

    msg1 = "I need something to help me work during blackouts"
    msg2 = "just looking for a phone case"

    failures = 0

    print("Testing classify_user_intent...")
    intent1 = classify_user_intent(msg1, persona)
    print(" - intent1:", intent1)
    if not isinstance(intent1, dict) or "intent" not in intent1:
        print("FAIL: classify_user_intent returned unexpected structure")
        failures += 1

    print("Testing understand_user...")
    state = understand_user(msg1, persona, chat_history=[{"role": "user", "content": msg1}])
    print(" - state keys:", list(state.keys()))
    if "intent" not in state:
        print("FAIL: understand_user did not set intent in state")
        failures += 1

    print("Testing retrieve_products (may download embeddings)...")
    prods = retrieve_products(msg1, top_k=3)
    print(f" - got {len(prods)} products")
    if not prods:
        print("FAIL: retrieve_products returned empty list")
        failures += 1

    print("Testing should_bridge...")
    br = should_bridge(state)
    print(" - should_bridge:", br)

    print("Testing retrieve_locations (may download embeddings)...")
    locs = retrieve_locations(bridge_category=state.get("intent", {}).get("bridge_category"), top_k=3)
    print(f" - got {len(locs)} locations")
    if not isinstance(locs, list):
        print("FAIL: retrieve_locations returned non-list")
        failures += 1

    if Config.GROQ_API_KEY or Config.GEMINI_API_KEY:
        try:
            print("Testing compose_response (will call Groq first, Gemini fallback)...")
            state["products"] = prods
            state["locations"] = locs
            text = compose_response(state, top_k_products=2)
            print(" - compose_response produced text length:", len(text))
        except Exception as e:
            print("FAIL: compose_response raised:", e)
            failures += 1
    else:
        print("Skipping compose_response test because neither GROQ_API_KEY nor GEMINI_API_KEY is set in env")

    if failures:
        print(f"\nCompleted with {failures} failures.")
        sys.exit(1)
    print("\nAll node tests passed.")


if __name__ == "__main__":
    run()
