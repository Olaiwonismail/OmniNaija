"""
OmniNaija Lite — Standalone RMSE Evaluation (Task A)
=====================================================
Computes RMSE for review simulation by comparing:
- predicted rating from /simulate
- ground-truth held-out rating from interactions data

USAGE:
  python evaluate_rmse.py
  python evaluate_rmse.py --quick
  python evaluate_rmse.py --num-cases 100
  python evaluate_rmse.py --output rmse_results.json
"""

import argparse
import json
from datetime import datetime

import requests

from evaluate_v2 import (
    API_BASE_URL,
    AMAZON_DATA_PATH,
    CHROMA_COLLECTION,
    CHROMA_PATH,
    DEFAULT_NUM_TEST_CASES,
    build_test_set,
    find_review_dataset,
    load_amazon_data,
    load_corpus,
    run_review_simulation_evaluation,
    to_bare_id,
)

from pathlib import Path

DEFAULT_OUTPUT_PATH = str(Path(__file__).resolve().parent / "rmse_results.json")



def main():
    parser = argparse.ArgumentParser(description="OmniNaija Task A RMSE Evaluation")
    parser.add_argument("--quick", action="store_true", help="Quick run with 10 test cases")
    parser.add_argument("--num-cases", type=int, default=DEFAULT_NUM_TEST_CASES,
                        help="Number of test cases when not using --quick")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH,
                        help="Output JSON path")
    args = parser.parse_args()

    print("=" * 60)
    print("OmniNaija Lite — Task A RMSE Evaluation")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}  |  API: {API_BASE_URL}")
    print("=" * 60)

    try:
        requests.get(f"{API_BASE_URL}/docs", timeout=5)
        print("API connection: OK")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot reach API at {API_BASE_URL}. Start: uvicorn main:app")
        return

    try:
        corpus_ids, _id_to_category, _ = load_corpus(CHROMA_PATH, CHROMA_COLLECTION)
    except Exception as e:
        print(f"ERROR loading ChromaDB corpus: {e}")
        print("Check CHROMA_PATH and CHROMA_COLLECTION in evaluate_v2.py CONFIG.")
        return

    num_cases = 10 if args.quick else args.num_cases
    print(f"\nLoading interactions from: {AMAZON_DATA_PATH}")

    df = None
    try:
        df = load_amazon_data(AMAZON_DATA_PATH)
    except (FileNotFoundError, ValueError) as e:
        print(f"WARNING: {e}")
        candidate = find_review_dataset()
        if candidate:
            print(f"Using auto-detected dataset: {candidate}")
            df = load_amazon_data(candidate)

    if df is None:
        print("ERROR: Could not load interactions dataset.")
        return

    print(f"Loaded {len(df)} interactions")
    all_interaction_products = set(df["product_id"].apply(to_bare_id))
    overlap = all_interaction_products & corpus_ids
    print(f"Overlap: {len(overlap)} products appear in BOTH the "
          f"interactions and the corpus (of {len(corpus_ids)} corpus items)")

    test_cases = build_test_set(df, num_cases, corpus_ids)
    review_sim = run_review_simulation_evaluation(test_cases)

    result = {
        "review_simulation": review_sim,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "api_url": API_BASE_URL,
            "mode": "standalone_rmse",
            "num_cases_requested": num_cases,
            "quick": bool(args.quick),
            "chroma_path": CHROMA_PATH,
            "chroma_collection": CHROMA_COLLECTION,
        },
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nResults saved to: {args.output}")
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Task A RMSE:             {review_sim['rmse']:.4f}")
    print(f"  Task A Valid Predictions:{review_sim['num_valid_predictions']}/{review_sim['num_total_cases']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
