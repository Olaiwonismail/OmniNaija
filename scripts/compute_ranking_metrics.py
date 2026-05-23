"""Compute ranking metrics (Hit Rate@K, NDCG@K) from stored results or by calling the /recommend API.

Usage examples:
  # Recompute from existing evaluation_results.json
  python scripts/compute_ranking_metrics.py --results evaluation_results.json --out results_ranking.json

  # Build test cases from dataset and call API to get recommendations (requires API running)
  python scripts/compute_ranking_metrics.py --input data/processed/user_interactions.jsonl --call-api --api-url http://localhost:8000 --out results_ranking.json
"""
from __future__ import annotations

import json
import time
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import ndcg_score

# Default config
DEFAULT_K = 10
DEFAULT_API_URL = "http://localhost:8000"


def load_amazon_data(path: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    df = df.rename(columns={
        "reviewerID": "user_id",
        "asin": "product_id",
        "overall": "rating",
        "reviewText": "review_text",
        "unixReviewTime": "timestamp",
        "reviewerName": "user_name",
        "summary": "review_summary",
    })

    required_cols = ["user_id", "product_id", "rating"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in dataset. Found: {list(df.columns)}")
    return df


def build_test_set(df: pd.DataFrame, num_cases: int, min_user_reviews: int = 5) -> List[dict]:
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    user_counts = df["user_id"].value_counts()
    eligible = user_counts[user_counts >= min_user_reviews].index.tolist()
    if len(eligible) == 0:
        raise ValueError("No users with enough reviews found; lower min_user_reviews")

    if len(eligible) > num_cases:
        rng = np.random.RandomState(42)
        eligible = rng.choice(eligible, size=num_cases, replace=False).tolist()

    cases = []
    for uid in eligible:
        ur = df[df["user_id"] == uid].copy()
        gt = ur.iloc[-1]
        history = ur.iloc[:-1]

        persona = {
            "name": f"User_{str(uid)[:8]}",
            "user_id": uid,
            "avg_rating": float(history["rating"].mean()) if len(history) else None,
            "review_count": int(len(history)),
            "recent_products": history["product_id"].tolist()[-5:],
        }

        cases.append({
            "persona": persona,
            "ground_truth_product": gt["product_id"],
            "ground_truth_rating": float(gt["rating"]),
            "history_product_ids": history["product_id"].tolist(),
        })

    return cases


def call_recommend(api_url: str, persona: dict, message: str, session_id: str = None) -> dict | None:
    payload = {"persona_description": persona, "message": message, "session_id": session_id or f"eval_{int(time.time())}"}
    try:
        resp = requests.post(f"{api_url.rstrip('/')}/recommend", json=payload, timeout=150)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print("API call failed:", e)
        return None


def extract_recommended_products(response: dict) -> List[str]:
    if not response:
        return []
    debug = response.get("debug", {})
    products = debug.get("products", [])
    return [p.split(":")[-1] if ":" in p else p for p in products]


def compute_hit_rate(details: List[dict], k: int = DEFAULT_K) -> float:
    hits = 0
    total = 0
    for r in details:
        recs = r.get("recommended_products") or []
        if recs is None:
            continue
        total += 1
        if r["ground_truth_product"] in recs[:k]:
            hits += 1
    return hits / total if total else 0.0


def compute_ndcg(details: List[dict], k: int = DEFAULT_K) -> float:
    scores = []
    for r in details:
        recs = r.get("recommended_products") or []
        gt = r["ground_truth_product"]
        rel = [1.0 if pid == gt else 0.0 for pid in recs[:k]]
        while len(rel) < k:
            rel.append(0.0)
        try:
            sc = ndcg_score([ [1.0] + [0.0]*(k-1) ], [rel], k=k)
        except Exception:
            sc = 0.0
        scores.append(sc)
    return float(np.mean(scores)) if scores else 0.0


def per_query_ndcg(recommended: List[str], gt: str, k: int) -> float:
    rel = [1.0 if pid == gt else 0.0 for pid in recommended[:k]]
    while len(rel) < k:
        rel.append(0.0)
    try:
        return float(ndcg_score([[1.0] + [0.0]*(k-1)], [rel], k=k))
    except Exception:
        return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", help="Path to raw dataset (CSV/JSONL) to build test cases from")
    p.add_argument("--results", help="Path to existing evaluation results JSON containing ranking.details")
    p.add_argument("--call-api", action="store_true", help="Call API to fetch recommendations for test cases")
    p.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL for the API")
    p.add_argument("--k", type=int, default=DEFAULT_K, help="Top-K for metrics")
    p.add_argument("--num-cases", type=int, default=50, help="Number of test cases when building from dataset")
    p.add_argument("--out", default="results_ranking.json", help="Output JSON path")
    args = p.parse_args()

    details = []

    if args.results and not args.input:
        obj = json.loads(Path(args.results).read_text())
        if "ranking" not in obj or "details" not in obj["ranking"]:
            raise ValueError("Provided results file does not contain ranking.details")
        details = obj["ranking"]["details"]

    elif args.input:
        df = load_amazon_data(args.input)
        test_cases = build_test_set(df, num_cases=args.num_cases)

        if args.call_api:
            for i, case in enumerate(test_cases):
                titles = case["persona"].get("recent_product_titles", [])
                if titles:
                    msg = f"I recently bought: {', '.join(titles[-5:])}. What else would you recommend for me?"
                else:
                    msg = "Based on my purchase history, what products would you recommend for me?"

                resp = call_recommend(args.api_url, case["persona"], msg)
                recs = extract_recommended_products(resp)
                details.append({
                    "user": case["persona"]["name"],
                    "ground_truth_product": case["ground_truth_product"],
                    "recommended_products": recs,
                    "hit": case["ground_truth_product"] in recs,
                })
                time.sleep(0.3)
        else:
            # If not calling API, leave recommended_products empty
            for case in test_cases:
                details.append({
                    "user": case["persona"]["name"],
                    "ground_truth_product": case["ground_truth_product"],
                    "recommended_products": [],
                    "hit": False,
                })
    else:
        raise ValueError("Provide either --results or --input (dataset)")

    # Compute aggregated + per-query metrics
    k = args.k
    aggregated = {
        "hit_rate": compute_hit_rate(details, k=k),
        "ndcg": compute_ndcg(details, k=k),
        "num_test_cases": len(details),
        "num_hits": sum(1 for d in details if d.get("hit")),
    }

    # Add per-query ndcg
    for d in details:
        recs = d.get("recommended_products") or []
        d["ndcg_at_k"] = per_query_ndcg(recs, d["ground_truth_product"], k)

    out_obj = {"aggregated": aggregated, "details": details, "k": k}
    Path(args.out).write_text(json.dumps(out_obj, indent=2))
    print(f"Wrote results to: {args.out}")
    print(f"Hit Rate@{k}: {aggregated['hit_rate']:.4f} ({aggregated['hit_rate']*100:.1f}%)")
    print(f"NDCG@{k}:     {aggregated['ndcg']:.4f}")


if __name__ == '__main__':
    main()
