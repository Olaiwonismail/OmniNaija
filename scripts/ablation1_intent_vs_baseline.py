"""Ablation 1: compare Intent Graph bridge_category vs a keyword baseline.

This script runs a small set of queries through the intent classifier and
compares its `bridge_category` to a simple keyword->bridge baseline.

Usage:
    python scripts/ablation1_intent_vs_baseline.py

Notes:
- This script avoids heavy retrieval (Chroma) and focuses on the intent
  extraction + bridge category decision. It will call `agent.intent.classify_user_intent`
  which uses the project's `llm.generate_text` wrapper. If you run this in
  demo mode or without an LLM provider, results may be mocked or require
  network access.
"""
from __future__ import annotations

from typing import List, Dict, Tuple
import json
import sys

from agent.intent import classify_user_intent, DEFAULT_BRIDGE_CATEGORIES


def keyword_baseline_bridge(product_category: str) -> str | None:
    """Dumb category-to-category mapping."""
    mapping = {
        "Electronics": "cafes",
        "Books": "libraries",
        "Sports": "gyms",
        "Home": "restaurants",
        "Baby": "family_restaurants",
    }
    return mapping.get(product_category)


TEST_QUERIES: List[Tuple[str, str]] = [
    ("I need something to help me work during blackouts", "Electronics"),
    ("I'm looking for a good novel to read", "Books"),
    ("I want to start working out and need equipment", "Sports"),
    ("I need a blender for cooking and trying new recipes", "Home"),
    ("Looking for a stroller for my baby", "Baby"),
    ("Just browsing cases for my phone", "Electronics"),
]


def run_case(message: str, assumed_product_category: str, persona: Dict | None = None) -> Dict:
    persona = persona or {"name": "Tobi", "location": "Yaba, Lagos"}

    # Call the intent classifier from the agent package
    try:
        intent = classify_user_intent(message, persona)
    except Exception as exc:
        intent = {"intent": "error", "confidence": 0.0, "bridge_category": None, "error": str(exc)}

    baseline = keyword_baseline_bridge(assumed_product_category)

    result = {
        "message": message,
        "assumed_product_category": assumed_product_category,
        "intent_graph": intent,
        "keyword_baseline_bridge": baseline,
    }
    return result


def main():
    results = []
    for msg, cat in TEST_QUERIES:
        print("\n---\nQuery:", msg)
        r = run_case(msg, cat)
        results.append(r)
        print("Assumed product category:", cat)
        print("Intent Graph -> intent:", r["intent_graph"].get("intent"),
              "confidence:", r["intent_graph"].get("confidence"),
              "bridge_category:", r["intent_graph"].get("bridge_category"))
        print("Keyword baseline ->", r["keyword_baseline_bridge"])

    # Summarize
    summary = {"cases": results}
    out_path = "ablation1_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
