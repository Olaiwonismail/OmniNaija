"""
OmniNaija Lite — Offline Evaluation Script (v2)
================================================
Computes Hit Rate@10, NDCG@10, Category-Match Rate@10, ROUGE, BERTScore,
and Cross-Domain Accuracy.

WHAT CHANGED IN v2:
- Loads the ChromaDB product corpus at startup (CHROMA_PATH / CHROMA_COLLECTION below)
- Filters the test set so each user's held-out (ground-truth) product actually
  EXISTS in the retrieval corpus — otherwise a hit is mathematically impossible
- Prefix-aware ID matching (corpus stores "Books:0761456228"; we compare on bare IDs)
- Adds Category-Match Rate@10 — did the agent recommend the right *kind* of product,
  even if not the exact held-out SKU? This is the fairer metric for semantic retrieval.

BEFORE RUNNING:
1. pip install scikit-learn requests pandas chromadb rouge-score bert-score --break-system-packages
2. Start your FastAPI server: uvicorn main:app
3. Confirm CHROMA_PATH and CHROMA_COLLECTION below match your setup

USAGE:
    python evaluate_v2.py
    python evaluate_v2.py --quick
    python evaluate_v2.py --review-sim           # Task A RMSE only in addition to default tasks
    python evaluate_v2.py --quick --review-sim
"""

import json
import time
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import chromadb
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# ============================================================
# CONFIG — adjust these to match your project
# ============================================================

API_BASE_URL = "http://localhost:8000"

AMAZON_DATA_PATH = "data/processed/user_interactions.jsonl"

# --- ChromaDB (the retrieval corpus the agent actually recommends from) ---
CHROMA_PATH = "/workspaces/OmniNaija/chroma_db"
CHROMA_COLLECTION = "amazon_products"

MIN_USER_REVIEWS = 5
DEFAULT_NUM_TEST_CASES = 50
TOP_K = 10
BRIDGE_CONFIDENCE_THRESHOLD = 0.6
RESULTS_OUTPUT_PATH = "evaluation_results.json"


# ============================================================
# ID NORMALISATION (prefix-aware)
# ============================================================

def to_bare_id(pid):
    """Strip a category prefix like 'Books:0761456228' -> '0761456228'."""
    if isinstance(pid, str) and ":" in pid:
        return pid.split(":", 1)[1]
    return pid


def _clean_text_value(value):
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


def build_reference_text(metadata):
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


# ============================================================
# CORPUS LOADING (from ChromaDB)
# ============================================================

def load_corpus(chroma_path, collection_name):
    """
    Load all product IDs + categories from the ChromaDB collection.
    Returns (bare_ids set, id_to_category dict, id_to_metadata dict).
    """
    client = chromadb.PersistentClient(path=chroma_path)
    col = client.get_collection(collection_name)

    data = col.get(include=["metadatas"])  # ids returned by default
    ids = data.get("ids", []) or []
    metas = data.get("metadatas", []) or []

    bare_ids = set()
    id_to_category = {}
    id_to_metadata = {}
    for raw_id, meta in zip(ids, metas):
        bare = to_bare_id(raw_id)
        bare_ids.add(bare)
        if meta:
            id_to_metadata[bare] = dict(meta)
            cat = meta.get("category") or meta.get("main_category") or meta.get("categories")
            if cat:
                id_to_category[bare] = cat

    print(f"Corpus loaded: {len(bare_ids)} products, "
          f"{len(set(id_to_category.values()))} distinct categories")
    return bare_ids, id_to_category, id_to_metadata


# ============================================================
# DATA LOADING
# ============================================================

def load_amazon_data(path):
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in (".json", ".jsonl"):
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
        "title": "product_title",
    })

    for col in ["user_id", "product_id", "rating"]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Available: {list(df.columns)}. "
                f"Edit the rename mapping in load_amazon_data()."
            )
    return df


def find_review_dataset(search_dirs=None):
    search_dirs = search_dirs or ["data/processed", "data/raw", "data"]
    review_indicators = {"user_id", "reviewerID", "reviewer_id", "reviewer",
                         "overall", "rating", "asin", "product_id"}
    for d in search_dirs:
        p = Path(d)
        if not p.exists():
            continue
        for f in sorted(p.iterdir()):
            if not f.is_file() or f.suffix.lower() not in {".csv", ".json", ".jsonl"}:
                continue
            try:
                cols = []
                if f.suffix.lower() == ".csv":
                    cols = pd.read_csv(f, nrows=0).columns.tolist()
                else:
                    with open(f, "r", encoding="utf-8") as fh:
                        first = next((ln.strip() for ln in fh if ln.strip()), None)
                    if not first:
                        continue
                    obj = json.loads(first)
                    if isinstance(obj, dict):
                        cols = list(obj.keys())
                    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
                        cols = list(obj[0].keys())
                if review_indicators & set(cols or []):
                    return str(f)
            except Exception:
                continue
    return None


# ============================================================
# TEST SET CONSTRUCTION (corpus-aware)
# ============================================================

def build_test_set(df, num_cases, corpus_ids):
    """
    For each eligible user, hold out their MOST RECENT in-corpus product as ground truth.
    Users with no in-corpus products are skipped (a hit would be impossible).
    """
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    user_counts = df["user_id"].value_counts()
    eligible_users = user_counts[user_counts >= MIN_USER_REVIEWS].index.tolist()
    print(f"Found {len(eligible_users)} users with >= {MIN_USER_REVIEWS} reviews")

    test_cases = []
    skipped_no_corpus = 0

    for uid in eligible_users:
        user_reviews = df[df["user_id"] == uid].copy()
        in_corpus = user_reviews[
            user_reviews["product_id"].apply(to_bare_id).isin(corpus_ids)
        ]
        if len(in_corpus) == 0:
            skipped_no_corpus += 1
            continue

        ground_truth = in_corpus.iloc[-1]
        history = user_reviews.drop(ground_truth.name)
        if len(history) == 0:
            continue

        avg_rating = history["rating"].mean()
        products_reviewed = history["product_id"].tolist()
        product_titles = (history["product_title"].tolist()
                          if "product_title" in history.columns else [])

        persona = {
            "name": f"User_{uid[:8]}" if isinstance(uid, str) else f"User_{uid}",
            "user_id": uid,
            "avg_rating": round(float(avg_rating), 1),
            "review_count": int(len(history)),
            "recent_products": [to_bare_id(p) for p in products_reviewed[-5:]],
            "recent_product_titles": [t for t in product_titles[-5:] if t],
            "traits": _infer_traits(history),
        }

        held_out_review_text = ""
        if "review_text" in ground_truth and pd.notna(ground_truth["review_text"]):
            held_out_review_text = str(ground_truth["review_text"]).strip()

        test_cases.append({
            "persona": persona,
            "ground_truth_product": to_bare_id(ground_truth["product_id"]),
            "ground_truth_rating": float(ground_truth["rating"]),
            "ground_truth_review_text": held_out_review_text,
            "history_product_ids": [to_bare_id(p) for p in products_reviewed],
        })

    print(f"Skipped {skipped_no_corpus} users with no in-corpus products")
    print(f"Built {len(test_cases)} usable test cases (held-out product is in corpus)")

    if len(test_cases) == 0:
        raise ValueError(
            "No test cases have an in-corpus ground-truth product. "
            "The interactions file and the ChromaDB corpus may not overlap. "
            "Check that the corpus was built from the same product universe."
        )

    if len(test_cases) > num_cases:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(test_cases), size=num_cases, replace=False)
        test_cases = [test_cases[i] for i in idx]

    return test_cases


def _infer_traits(history):
    traits = []
    avg = history["rating"].mean()
    if avg >= 4.0:
        traits.append("generally positive reviewer")
    elif avg <= 2.5:
        traits.append("critical reviewer")
    else:
        traits.append("balanced reviewer")
    if len(history) >= 10:
        traits.append("experienced buyer")
    std = history["rating"].std()
    if pd.notna(std):
        if std < 0.5:
            traits.append("consistent in ratings")
        elif std > 1.5:
            traits.append("varied opinions")
    return traits


# ============================================================
# API CALLS
# ============================================================

def call_recommend(persona, message, session_id=None):
    payload = {
        "persona_description": persona,
        "message": message,
        "session_id": session_id or f"eval_{int(time.time())}",
    }
    try:
        resp = requests.post(f"{API_BASE_URL}/recommend", json=payload, timeout=150)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  API call failed: {e}")
        return None


def call_simulate(persona, product_id):
    payload = {
        "persona_description": persona,
        "product_id": product_id,
    }
    try:
        resp = requests.post(f"{API_BASE_URL}/simulate", json=payload, timeout=150)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  /simulate call failed: {e}")
        return None


def extract_generated_review(sim_response):
    if not sim_response:
        return ""
    review = sim_response.get("review")
    if isinstance(review, str):
        return review.strip()
    review_text = sim_response.get("review_text")
    if isinstance(review_text, str):
        return review_text.strip()
    return ""


def extract_predicted_rating(sim_response):
    if not sim_response:
        return None
    try:
        rating = float(sim_response.get("rating"))
    except (TypeError, ValueError, AttributeError):
        return None
    if rating < 1 or rating > 5:
        return None
    return rating


def extract_recommended_products(response):
    if not response:
        return []
    products = response.get("debug", {}).get("products", [])
    return [to_bare_id(p) for p in products]


def extract_recommendation_text(response):
    if not response:
        return ""
    recommendation = response.get("recommendation")
    if isinstance(recommendation, str):
        return recommendation.strip()
    return ""


# ============================================================
# METRICS
# ============================================================

def compute_hit_rate(test_results):
    hits = total = 0
    for r in test_results:
        if r["recommended_products"] is not None:
            total += 1
            if r["ground_truth_product"] in r["recommended_products"]:
                hits += 1
    return hits / total if total else 0.0


def compute_ndcg(test_results, k=TOP_K):
    scores = []
    for r in test_results:
        rec = r["recommended_products"]
        if rec is None:
            continue
        gt = r["ground_truth_product"]
        relevance = [1.0 if pid == gt else 0.0 for pid in rec[:k]]
        if sum(relevance) == 0:
            scores.append(0.0)
            continue
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
        idcg = 1.0 / np.log2(2)
        scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else 0.0


def compute_category_match(test_results, id_to_category):
    """Category-Match@K: did ANY recommended product share the GT product's category?"""
    matches = total = 0
    for r in test_results:
        gt_cat = id_to_category.get(r["ground_truth_product"])
        if gt_cat is None:
            continue
        rec_cats = {id_to_category.get(p) for p in (r["recommended_products"] or [])}
        rec_cats.discard(None)
        total += 1
        if gt_cat in rec_cats:
            matches += 1
    return matches / total if total else 0.0


def compute_rmse(test_results):
    errors = []
    for r in test_results:
        pred = r.get("predicted_rating")
        actual = r.get("actual_rating")
        if pred is None or actual is None:
            continue
        errors.append((pred - actual) ** 2)
    if not errors:
        return 0.0
    return float(np.sqrt(np.mean(errors)))


def compute_text_metrics(test_results):
    rouge_available = rouge_scorer is not None
    bert_available = bert_score is not None
    rouge_scores = []
    bert_f1_scores = []

    scorer = None
    if rouge_available:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    valid_examples = []
    for r in test_results:
        generated = (r.get("generated_text") or "").strip()
        reference = (r.get("reference_text") or "").strip()
        if not generated or not reference:
            r["rouge"] = None
            r["bertscore"] = None
            continue

        valid_examples.append(r)

        if scorer is not None:
            scores = scorer.score(reference, generated)
            rouge_entry = {
                "rouge1_f": round(float(scores["rouge1"].fmeasure), 4),
                "rouge2_f": round(float(scores["rouge2"].fmeasure), 4),
                "rougeL_f": round(float(scores["rougeL"].fmeasure), 4),
            }
            rouge_scores.append(rouge_entry)
            r["rouge"] = rouge_entry
        else:
            r["rouge"] = None

    if bert_available and valid_examples:
        try:
            predictions = [r["generated_text"] for r in valid_examples]
            references = [r["reference_text"] for r in valid_examples]
            _precision, _recall, f1 = bert_score(
                predictions,
                references,
                lang="en",
                model_type="distilbert-base-uncased",
                verbose=False,
                rescale_with_baseline=True,
            )
            bert_f1_scores = [float(value) for value in f1]
            for r, score in zip(valid_examples, bert_f1_scores):
                r["bertscore"] = {
                    "f1": round(score, 4),
                    "model_type": "distilbert-base-uncased",
                }
        except Exception as exc:
            print(f"  BERTScore unavailable: {exc}")
            for r in valid_examples:
                r["bertscore"] = None

    aggregate = {
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
    return aggregate


# ============================================================
# TASK A EVALUATION (REVIEW SIMULATION)
# ============================================================

def run_review_simulation_evaluation(test_cases):
    print("\n" + "=" * 60)
    print("TASK A EVALUATION: Review Simulation (RMSE)")
    print("=" * 60)

    test_results = []
    for i, case in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {case['persona']['name']}")
        gt_product = case["ground_truth_product"]
        actual_rating = float(case["ground_truth_rating"])

        response = call_simulate(persona=case["persona"], product_id=gt_product)
        predicted_rating = extract_predicted_rating(response)

        if predicted_rating is None:
            print(f"  GT product={gt_product}  actual={actual_rating:.1f}  pred=invalid")
            test_results.append({
                "user": case["persona"]["name"],
                "ground_truth_product": gt_product,
                "actual_rating": actual_rating,
                "predicted_rating": None,
                "squared_error": None,
                "valid_prediction": False,
            })
            time.sleep(0.3)
            continue

        sq_err = (predicted_rating - actual_rating) ** 2
        print(
            f"  GT product={gt_product}  actual={actual_rating:.1f}  "
            f"pred={predicted_rating:.1f}  sq_err={sq_err:.3f}"
        )
        test_results.append({
            "user": case["persona"]["name"],
            "ground_truth_product": gt_product,
            "actual_rating": actual_rating,
            "predicted_rating": predicted_rating,
            "squared_error": round(float(sq_err), 6),
            "valid_prediction": True,
        })
        time.sleep(0.3)

    rmse = compute_rmse(test_results)
    num_valid = sum(1 for r in test_results if r["valid_prediction"])
    num_total = len(test_results)

    print("\n" + "-" * 40)
    print(f"RMSE:              {rmse:.4f}")
    print(f"Valid predictions: {num_valid}/{num_total}")
    print("-" * 40)

    return {
        "rmse": round(rmse, 4),
        "num_valid_predictions": num_valid,
        "num_total_cases": num_total,
        "details": test_results,
    }


def run_task_a_evaluation(test_cases, num_cases=30):
    print("\n" + "=" * 60)
    print("TASK A EVALUATION: Review Generation Quality")
    print("=" * 60)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    test_results = []

    cases_to_run = test_cases[:num_cases]
    for i, case in enumerate(cases_to_run):
        persona = case["persona"]
        gt_product = case["ground_truth_product"]
        reference_review = (case.get("ground_truth_review_text") or "").strip()

        print(f"\n[{i + 1}/{len(cases_to_run)}] {persona['name']}")

        if not reference_review:
            print(f"  Skipping case: missing held-out review text for product={gt_product}")
            test_results.append({
                "user": persona["name"],
                "ground_truth_product": gt_product,
                "reference_review": "",
                "generated_review": None,
                "rouge1_f1": None,
                "rouge2_f1": None,
                "rougeL_f1": None,
                "bertscore_f1": None,
                "valid_case": False,
                "skip_reason": "missing_reference_review",
            })
            continue

        response = call_simulate(persona=persona, product_id=gt_product)
        generated_review = extract_generated_review(response)

        if not generated_review:
            print(f"  Skipping case: /simulate failed or returned no review for product={gt_product}")
            test_results.append({
                "user": persona["name"],
                "ground_truth_product": gt_product,
                "reference_review": reference_review,
                "generated_review": None,
                "rouge1_f1": None,
                "rouge2_f1": None,
                "rougeL_f1": None,
                "bertscore_f1": None,
                "valid_case": False,
                "skip_reason": "simulate_failed_or_empty_review",
            })
            time.sleep(0.3)
            continue

        try:
            rouge_scores = scorer.score(reference_review, generated_review)
            rouge1_f1 = float(rouge_scores["rouge1"].fmeasure)
            rouge2_f1 = float(rouge_scores["rouge2"].fmeasure)
            rougeL_f1 = float(rouge_scores["rougeL"].fmeasure)

            _precision, _recall, f1 = bert_score(
                [generated_review],
                [reference_review],
                lang="en",
                model_type="microsoft/deberta-xlarge-mnli",
            )
            bertscore_f1 = float(f1.mean().item())
        except Exception as exc:
            print(f"  Skipping case: metric computation failed for product={gt_product} ({exc})")
            test_results.append({
                "user": persona["name"],
                "ground_truth_product": gt_product,
                "reference_review": reference_review,
                "generated_review": generated_review,
                "rouge1_f1": None,
                "rouge2_f1": None,
                "rougeL_f1": None,
                "bertscore_f1": None,
                "valid_case": False,
                "skip_reason": f"metric_computation_failed: {exc}",
            })
            time.sleep(0.3)
            continue

        print(
            f"  GT product={gt_product}  "
            f"ROUGE-1={rouge1_f1:.4f}  ROUGE-2={rouge2_f1:.4f}  ROUGE-L={rougeL_f1:.4f}  BERTScore={bertscore_f1:.4f}"
        )
        test_results.append({
            "user": persona["name"],
            "ground_truth_product": gt_product,
            "reference_review": reference_review,
            "generated_review": generated_review,
            "rouge1_f1": round(rouge1_f1, 4),
            "rouge2_f1": round(rouge2_f1, 4),
            "rougeL_f1": round(rougeL_f1, 4),
            "bertscore_f1": round(bertscore_f1, 4),
            "valid_case": True,
            "skip_reason": None,
        })
        time.sleep(0.3)

    valid_results = [result for result in test_results if result["valid_case"]]
    rouge1_avg = float(np.mean([result["rouge1_f1"] for result in valid_results])) if valid_results else 0.0
    rouge2_avg = float(np.mean([result["rouge2_f1"] for result in valid_results])) if valid_results else 0.0
    rougeL_avg = float(np.mean([result["rougeL_f1"] for result in valid_results])) if valid_results else 0.0
    bertscore_avg = float(np.mean([result["bertscore_f1"] for result in valid_results])) if valid_results else 0.0

    print("\n" + "-" * 40)
    print(f"ROUGE-1 F1:   {rouge1_avg:.4f}")
    print(f"ROUGE-2 F1:   {rouge2_avg:.4f}")
    print(f"ROUGE-L F1:   {rougeL_avg:.4f}")
    print(f"BERTScore F1: {bertscore_avg:.4f}")
    print(f"Valid cases:  {len(valid_results)}/{len(test_results)}")
    print("-" * 40)

    return {
        "rouge1_f1": round(rouge1_avg, 4),
        "rouge2_f1": round(rouge2_avg, 4),
        "rougeL_f1": round(rougeL_avg, 4),
        "bertscore_f1": round(bertscore_avg, 4),
        "num_test_cases": len(valid_results),
        "num_attempted_cases": len(test_results),
        "details": test_results,
    }


# ============================================================
# RANKING EVALUATION
# ============================================================

def run_ranking_evaluation(test_cases, id_to_category, id_to_metadata):
    print("\n" + "=" * 60)
    print("TASK B EVALUATION: Ranking Quality")
    print("=" * 60)

    test_results = []
    for i, case in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {case['persona']['name']}")
        titles = case["persona"].get("recent_product_titles", [])
        if titles:
            message = (f"I recently bought: {', '.join(titles[-5:])}. "
                       f"What else would you recommend for me?")
        else:
            message = "Based on my purchase history, what products would you recommend for me?"

        response = call_recommend(persona=case["persona"], message=message)
        recommended = extract_recommended_products(response)
        recommendation_text = extract_recommendation_text(response)
        gt = case["ground_truth_product"]
        hit = gt in recommended

        gt_cat = id_to_category.get(gt)
        rec_cats = {id_to_category.get(p) for p in recommended}
        rec_cats.discard(None)
        cat_match = (gt_cat in rec_cats) if gt_cat else False
        reference_text = build_reference_text(id_to_metadata.get(gt)) or " ".join(
            part for part in [gt_cat, gt] if part
        )

        print(f"  GT: {gt} ({gt_cat})  hit={'Y' if hit else 'n'}  cat_match={'Y' if cat_match else 'n'}")
        print(f"  Recommended: {recommended[:5]}")

        test_results.append({
            "user": case["persona"]["name"],
            "ground_truth_product": gt,
            "ground_truth_category": gt_cat,
            "recommended_products": recommended,
            "generated_text": recommendation_text,
            "reference_text": reference_text,
            "hit": hit,
            "category_match": cat_match,
        })
        time.sleep(0.5)

    hit_rate = compute_hit_rate(test_results)
    ndcg = compute_ndcg(test_results)
    cat_match_rate = compute_category_match(test_results, id_to_category)
    text_metrics = compute_text_metrics(test_results)

    print("\n" + "-" * 40)
    print(f"Hit Rate@{TOP_K}:        {hit_rate:.4f} ({hit_rate*100:.1f}%)")
    print(f"NDCG@{TOP_K}:            {ndcg:.4f}")
    print(f"Category-Match@{TOP_K}:  {cat_match_rate:.4f} ({cat_match_rate*100:.1f}%)")
    print(f"ROUGE-1 F1:            {text_metrics['rouge']['rouge1_f']:.4f}")
    print(f"ROUGE-2 F1:            {text_metrics['rouge']['rouge2_f']:.4f}")
    print(f"ROUGE-L F1:            {text_metrics['rouge']['rougeL_f']:.4f}")
    bertscore_f1 = text_metrics["bertscore"]["f1"]
    print(f"BERTScore F1:          {bertscore_f1:.4f}" if bertscore_f1 is not None else "BERTScore F1:          n/a")
    print(f"Test cases:          {len(test_results)}")
    print("-" * 40)

    return {
        "hit_rate": round(hit_rate, 4),
        "ndcg": round(ndcg, 4),
        "category_match_rate": round(cat_match_rate, 4),
        "text_metrics": text_metrics,
        "num_test_cases": len(test_results),
        "num_hits": sum(1 for r in test_results if r["hit"]),
        "num_category_matches": sum(1 for r in test_results if r["category_match"]),
        "details": test_results,
    }


# ============================================================
# CROSS-DOMAIN EVALUATION
# ============================================================

CROSS_DOMAIN_TEST_CASES = [
    {"message": "I need to keep working during power outages", "expected_intent": "remote_work_setup", "should_bridge": True},
    {"message": "I want to start working out at home", "expected_intent": "fitness_journey", "should_bridge": True},
    {"message": "I'm preparing for my friend's owambe next month", "expected_intent": "owambe_prep", "should_bridge": True},
    {"message": "I want to get into serious cooking at home", "expected_intent": "cooking_exploration", "should_bridge": True},
    {"message": "I need a new phone case", "expected_intent": "general_browsing", "should_bridge": False},
    {"message": "Looking for a USB cable", "expected_intent": "general_browsing", "should_bridge": False},
    {"message": "I want to set up a cozy reading corner at home", "expected_intent": "reading_habit", "should_bridge": True},
    {"message": "We just had a baby and need to get everything ready", "expected_intent": "baby_family_prep", "should_bridge": True},
    {"message": "I need a new HDMI cable for my monitor", "expected_intent": "general_browsing", "should_bridge": False},
    {"message": "Setting up a complete home office since I just went fully remote", "expected_intent": "remote_work_setup", "should_bridge": True},
]

DEFAULT_CROSS_DOMAIN_PERSONA = {
    "name": "Tobi", "age": 27, "location": "Yaba, Lagos",
    "occupation": "freelance developer", "budget": "careful with money",
    "traits": ["budget-conscious", "likes practical gadgets", "works remotely"],
}


def run_cross_domain_evaluation():
    print("\n" + "=" * 60)
    print("CROSS-DOMAIN BRIDGE EVALUATION")
    print("=" * 60)

    results = []
    for i, case in enumerate(CROSS_DOMAIN_TEST_CASES):
        print(f"\n[{i+1}/{len(CROSS_DOMAIN_TEST_CASES)}] \"{case['message']}\"")
        response = call_recommend(persona=DEFAULT_CROSS_DOMAIN_PERSONA, message=case["message"])

        if not response:
            results.append({**case, "passed": None, "actual_intent": None, "actual_bridged": None})
            continue

        debug = response.get("debug", {})
        intent_data = debug.get("intent", {})
        actual_intent = intent_data.get("intent", "unknown")
        actual_conf = intent_data.get("confidence", 0)
        actual_bridged = debug.get("bridged", False)
        locations = debug.get("locations", [])

        bridge_correct = (case["should_bridge"] == actual_bridged)
        print(f"  intent={actual_intent} ({actual_conf:.2f})  bridge={'Y' if actual_bridged else 'N'}  -> {'PASS' if bridge_correct else 'FAIL'}")

        results.append({
            "message": case["message"],
            "expected_intent": case["expected_intent"],
            "actual_intent": actual_intent,
            "confidence": actual_conf,
            "should_bridge": case["should_bridge"],
            "actual_bridged": actual_bridged,
            "bridge_correct": bridge_correct,
            "locations": locations,
            "passed": bridge_correct,
        })
        time.sleep(0.5)

    valid = [r for r in results if r["passed"] is not None]
    num_passed = sum(1 for r in valid if r["passed"])
    total = len(valid)
    should_b = [r for r in valid if r["should_bridge"]]
    should_not = [r for r in valid if not r["should_bridge"]]
    precision = sum(1 for r in should_b if r["passed"]) / len(should_b) if should_b else 0
    restraint = sum(1 for r in should_not if r["passed"]) / len(should_not) if should_not else 0

    print("\n" + "-" * 40)
    print(f"Cross-Domain Accuracy:  {(num_passed/total if total else 0):.1%} ({num_passed}/{total})")
    print(f"Bridge Precision:       {precision:.1%}")
    print(f"Bridge Restraint:       {restraint:.1%}")
    print("-" * 40)

    return {
        "accuracy": round(num_passed / total, 4) if total else 0,
        "bridge_precision": round(precision, 4),
        "bridge_restraint": round(restraint, 4),
        "num_passed": num_passed,
        "num_total": total,
        "details": results,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="OmniNaija Lite Evaluation v2")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cross-domain", action="store_true")
    parser.add_argument("--review-sim", action="store_true",
                        help="Run Task A review simulation RMSE evaluation")
    parser.add_argument("--num-cases", type=int, default=DEFAULT_NUM_TEST_CASES)
    args = parser.parse_args()

    print("=" * 60)
    print("OmniNaija Lite — Evaluation Suite v2")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}  |  API: {API_BASE_URL}")
    print("=" * 60)

    try:
        requests.get(f"{API_BASE_URL}/docs", timeout=5)
        print("API connection: OK")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot reach API at {API_BASE_URL}. Start: uvicorn main:app")
        return

    all_results = {}

    if not args.cross_domain:
        try:
            corpus_ids, id_to_category, id_to_metadata = load_corpus(CHROMA_PATH, CHROMA_COLLECTION)
        except Exception as e:
            print(f"ERROR loading ChromaDB corpus: {e}")
            print("Check CHROMA_PATH and CHROMA_COLLECTION in CONFIG.")
            corpus_ids, id_to_category, id_to_metadata = None, {}, {}

        if corpus_ids:
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

            if df is not None:
                print(f"Loaded {len(df)} interactions")
                all_interaction_products = set(df["product_id"].apply(to_bare_id))
                overlap = all_interaction_products & corpus_ids
                print(f"Overlap: {len(overlap)} products appear in BOTH the "
                      f"interactions and the corpus (of {len(corpus_ids)} corpus items)")

                test_cases = build_test_set(df, num_cases, corpus_ids)
                all_results["task_a_review_quality"] = run_task_a_evaluation(test_cases, num_cases=num_cases)
                if args.review_sim:
                    all_results["review_simulation"] = run_review_simulation_evaluation(test_cases)
                all_results["ranking"] = run_ranking_evaluation(test_cases, id_to_category, id_to_metadata)

    all_results["cross_domain"] = run_cross_domain_evaluation()

    all_results["timestamp"] = datetime.now().isoformat()
    all_results["config"] = {"api_url": API_BASE_URL, "top_k": TOP_K,
                             "bridge_threshold": BRIDGE_CONFIDENCE_THRESHOLD,
                             "review_sim_enabled": bool(args.review_sim)}

    with open(RESULTS_OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {RESULTS_OUTPUT_PATH}")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    if "review_simulation" in all_results:
        rs = all_results["review_simulation"]
        print(f"  Task A RMSE:             {rs['rmse']:.4f}")
        print(f"  Task A Valid Predictions:{rs['num_valid_predictions']}/{rs['num_total_cases']}")
    if "task_a_review_quality" in all_results:
        ta = all_results["task_a_review_quality"]
        print(f"  Task A ROUGE-1 F1:       {ta['rouge1_f1']:.4f}")
        print(f"  Task A ROUGE-2 F1:       {ta['rouge2_f1']:.4f}")
        print(f"  Task A ROUGE-L F1:       {ta['rougeL_f1']:.4f}")
        print(f"  Task A BERTScore F1:     {ta['bertscore_f1']:.4f}")
    if "ranking" in all_results:
        r = all_results["ranking"]
        print(f"  Hit Rate@{TOP_K}:        {r['hit_rate']*100:.1f}%")
        print(f"  NDCG@{TOP_K}:            {r['ndcg']:.4f}")
        print(f"  Category-Match@{TOP_K}:  {r['category_match_rate']*100:.1f}%")
        text_metrics = r.get("text_metrics", {})
        rouge = text_metrics.get("rouge", {})
        bertscore = text_metrics.get("bertscore", {})
        print(f"  ROUGE-1 F1:              {rouge.get('rouge1_f', 0.0):.4f}")
        print(f"  ROUGE-2 F1:              {rouge.get('rouge2_f', 0.0):.4f}")
        print(f"  ROUGE-L F1:              {rouge.get('rougeL_f', 0.0):.4f}")
        bert_f1 = bertscore.get("f1")
        print(f"  BERTScore F1:            {bert_f1:.4f}" if bert_f1 is not None else "  BERTScore F1:            n/a")
    cd = all_results["cross_domain"]
    print(f"  Cross-Domain Accuracy:   {cd['accuracy']:.1%}")
    print(f"  Bridge Precision:        {cd['bridge_precision']:.1%}")
    print(f"  Bridge Restraint:        {cd['bridge_restraint']:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()