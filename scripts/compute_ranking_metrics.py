"""Compute ranking metrics (Hit Rate@K, NDCG@K, ROUGE, BERTScore) from stored results or by calling the /recommend API.

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

import chromadb

try:
    from rouge_score import rouge_scorer  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    rouge_scorer = None

try:
    from bert_score import score as bertscore_score  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    bertscore_score = None

# Default config
DEFAULT_K = 10
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_CHROMA_PATH = "chroma_db"
DEFAULT_CHROMA_COLLECTION = "amazon_products"


def to_bare_id(pid: str | None) -> str | None:
    if isinstance(pid, str) and ":" in pid:
        return pid.split(":", 1)[1]
    return pid


def _clean_text_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        parts = [_clean_text_value(item) for item in value]
        return "; ".join(part for part in parts if part)
    if isinstance(value, dict):
        parts = []
        for key in ("title", "name", "category", "brand", "store", "description", "summary"):
            item = _clean_text_value(value.get(key))
            if item:
                parts.append(item)
        if parts:
            return ", ".join(parts)
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def build_reference_text(metadata) -> str:
    if not metadata:
        return ""

    parts = []
    for key in ("title", "brand", "store", "category"):
        item = _clean_text_value(metadata.get(key))
        if item:
            parts.append(item)

    for key in ("features", "details", "bundled_reviews"):
        item = _clean_text_value(metadata.get(key))
        if item:
            parts.append(item)

    for key in ("price", "avg_rating", "review_count"):
        item = metadata.get(key)
        if item is not None and item != "":
            parts.append(f"{key.replace('_', ' ')}: {item}")

    return ". ".join(parts)


def load_corpus(chroma_path: str, collection_name: str) -> dict[str, dict]:
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(collection_name)
    data = collection.get(include=["metadatas"])
    ids = data.get("ids", []) or []
    metadatas = data.get("metadatas", []) or []

    id_to_metadata: dict[str, dict] = {}
    for raw_id, meta in zip(ids, metadatas):
        bare = to_bare_id(raw_id)
        if bare and meta:
            id_to_metadata[bare] = dict(meta)
    return id_to_metadata


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


def extract_recommendation_text(response: dict) -> str:
    if not response:
        return ""
    recommendation = response.get("recommendation")
    if isinstance(recommendation, str):
        return recommendation.strip()
    return ""


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
        y_true = [rel]
        y_score = [[float(k - i) for i in range(k)]]
        try:
            sc = ndcg_score(y_true, y_score, k=k)
        except Exception:
            sc = 0.0
        scores.append(sc)
    return float(np.mean(scores)) if scores else 0.0


def per_query_ndcg(recommended: List[str], gt: str, k: int) -> float:
    rel = [1.0 if pid == gt else 0.0 for pid in recommended[:k]]
    while len(rel) < k:
        rel.append(0.0)
    y_true = [rel]
    y_score = [[float(k - i) for i in range(k)]]
    try:
        return float(ndcg_score(y_true, y_score, k=k))
    except Exception:
        return 0.0


def compute_text_metrics(details: List[dict]) -> dict:
    rouge_available = rouge_scorer is not None
    bert_available = bertscore_score is not None
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True) if rouge_available else None

    rouge_scores = []
    bert_f1_scores = []
    valid_examples = []

    for d in details:
        generated = (d.get("generated_text") or "").strip()
        reference = (d.get("reference_text") or "").strip()
        if not generated or not reference:
            d["rouge"] = None
            d["bertscore"] = None
            continue

        valid_examples.append(d)
        if scorer is not None:
            scores = scorer.score(reference, generated)
            rouge_entry = {
                "rouge1_f": round(float(scores["rouge1"].fmeasure), 4),
                "rouge2_f": round(float(scores["rouge2"].fmeasure), 4),
                "rougeL_f": round(float(scores["rougeL"].fmeasure), 4),
            }
            rouge_scores.append(rouge_entry)
            d["rouge"] = rouge_entry
        else:
            d["rouge"] = None

    if bert_available and valid_examples:
        try:
            predictions = [d["generated_text"] for d in valid_examples]
            references = [d["reference_text"] for d in valid_examples]
            _precision, _recall, f1 = bertscore_score(
                predictions,
                references,
                lang="en",
                model_type="distilbert-base-uncased",
                verbose=False,
                rescale_with_baseline=True,
            )
            bert_f1_scores = [float(value) for value in f1]
            for d, score in zip(valid_examples, bert_f1_scores):
                d["bertscore"] = {
                    "f1": round(score, 4),
                    "model_type": "distilbert-base-uncased",
                }
        except Exception as exc:
            print("BERTScore unavailable:", exc)
            for d in valid_examples:
                d["bertscore"] = None

    return {
        "num_examples": len(valid_examples),
        "rouge": {
            "rouge1_f": round(float(np.mean([item["rouge1_f"] for item in rouge_scores])), 4) if rouge_scores else 0.0,
            "rouge2_f": round(float(np.mean([item["rouge2_f"] for item in rouge_scores])), 4) if rouge_scores else 0.0,
            "rougeL_f": round(float(np.mean([item["rougeL_f"] for item in rouge_scores])), 4) if rouge_scores else 0.0,
            "available": rouge_available,
        },
        "bertscore": {
            "f1": round(float(np.mean(bert_f1_scores)), 4) if bert_f1_scores else None,
            "available": bert_available and bool(bert_f1_scores),
            "model_type": "distilbert-base-uncased" if bert_f1_scores else None,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", help="Path to raw dataset (CSV/JSONL) to build test cases from")
    p.add_argument("--results", help="Path to existing evaluation results JSON containing ranking.details")
    p.add_argument("--call-api", action="store_true", help="Call API to fetch recommendations for test cases")
    p.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL for the API")
    p.add_argument("--corpus-path", default=DEFAULT_CHROMA_PATH, help="Path to the ChromaDB corpus for reference text lookup")
    p.add_argument("--collection-name", default=DEFAULT_CHROMA_COLLECTION, help="ChromaDB collection name for reference text lookup")
    p.add_argument("--k", type=int, default=DEFAULT_K, help="Top-K for metrics")
    p.add_argument("--num-cases", type=int, default=50, help="Number of test cases when building from dataset")
    p.add_argument("--out", default="results_ranking.json", help="Output JSON path")
    args = p.parse_args()

    details = []
    id_to_metadata: dict[str, dict] = {}

    if args.results or args.call_api:
        try:
            id_to_metadata = load_corpus(args.corpus_path, args.collection_name)
        except Exception as exc:
            print("Warning: could not load corpus metadata for text metrics:", exc)
            id_to_metadata = {}

    if args.results and not args.input:
        obj = json.loads(Path(args.results).read_text())
        ranking = obj.get("ranking")
        if not isinstance(ranking, dict) or "details" not in ranking:
            available_sections = ", ".join(sorted(k for k in obj.keys() if isinstance(k, str)))
            raise ValueError(
                "Provided results file does not contain ranking.details. "
                f"Available top-level sections: {available_sections or 'none'}. "
                "Run `python evaluate_v2.py` without `--cross-domain`, or pass a JSON file "
                "that includes the ranking section."
            )
        details = ranking["details"]

        for d in details:
            if not d.get("reference_text"):
                gt = to_bare_id(d.get("ground_truth_product"))
                d["reference_text"] = build_reference_text(id_to_metadata.get(gt)) or " ".join(
                    part for part in [gt] if part
                )
            if not d.get("generated_text"):
                d["generated_text"] = ""

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
                recommendation_text = extract_recommendation_text(resp)
                gt = to_bare_id(case["ground_truth_product"])
                details.append({
                    "user": case["persona"]["name"],
                    "ground_truth_product": gt,
                    "recommended_products": recs,
                    "generated_text": recommendation_text,
                    "reference_text": build_reference_text(id_to_metadata.get(gt)) or " ".join(part for part in [gt] if part),
                    "hit": gt in recs,
                })
                time.sleep(0.3)
        else:
            # If not calling API, leave recommended_products empty
            for case in test_cases:
                gt = to_bare_id(case["ground_truth_product"])
                details.append({
                    "user": case["persona"]["name"],
                    "ground_truth_product": gt,
                    "recommended_products": [],
                    "generated_text": "",
                    "reference_text": build_reference_text(id_to_metadata.get(gt)) or " ".join(part for part in [gt] if part),
                    "hit": False,
                })
    else:
        raise ValueError("Provide either --results or --input (dataset)")

    # Compute aggregated + per-query metrics
    k = args.k
    text_metrics = compute_text_metrics(details)
    aggregated = {
        "hit_rate": compute_hit_rate(details, k=k),
        "ndcg": compute_ndcg(details, k=k),
        "text_metrics": text_metrics,
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
    rouge = text_metrics["rouge"]
    bertscore = text_metrics["bertscore"]
    print(f"ROUGE-1 F1:   {rouge['rouge1_f']:.4f}")
    print(f"ROUGE-2 F1:   {rouge['rouge2_f']:.4f}")
    print(f"ROUGE-L F1:   {rouge['rougeL_f']:.4f}")
    bert_f1 = bertscore.get("f1")
    print(f"BERTScore F1: {bert_f1:.4f}" if bert_f1 is not None else "BERTScore F1: n/a")


if __name__ == '__main__':
    main()
