"""
OmniNaija Lite — Offline Evaluation Script
===========================================
Computes NDCG@10, Hit Rate@10, and Cross-Domain Relevance Accuracy.

BEFORE RUNNING:
1. pip install scikit-learn requests pandas --break-system-packages
2. Make sure your FastAPI server is running (uvicorn main:app)
3. Adjust the CONFIG section below to match your dataset paths and formats

USAGE:
    python evaluate.py                  # full eval
    python evaluate.py --quick          # quick run with 10 test cases (sanity check)
    python evaluate.py --cross-domain   # only run cross-domain bridge evaluation
"""

import json
import time
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIG — adjust these to match your project
# ============================================================

# Your FastAPI server URL
API_BASE_URL = "http://localhost:8000"

# Path to your Amazon reviews dataset (CSV or JSON)
# Expected columns/fields: user_id, product_id, rating, review_text, timestamp (or similar)
# Adjust the column names in load_amazon_data() if yours differ
# AMAZON_DATA_PATH = "data/processed/amazon_electronics.jsonl"  # <-- updated to Member 3's cleaned dataset
AMAZON_DATA_PATH = "data/processed/user_interactions.jsonl"
# Minimum number of reviews a user must have to be included in the test set
MIN_USER_REVIEWS = 5

# Number of test cases to evaluate (set lower for quick runs)
DEFAULT_NUM_TEST_CASES = 50

# Number of recommendations to request per test case
TOP_K = 10

# Intent confidence threshold (should match your agent's threshold)
BRIDGE_CONFIDENCE_THRESHOLD = 0.6

# Output path for results
RESULTS_OUTPUT_PATH = "evaluation_results.json"


# ============================================================
# DATA LOADING
# ============================================================

def load_amazon_data(path: str) -> pd.DataFrame:
    """
    Load the Amazon dataset. 
    ADJUST THIS FUNCTION to match your actual data format.
    """
    path = Path(path)
    
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".json":
        df = pd.read_json(path, lines=True)
    elif path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # -------------------------------------------------------
    # IMPORTANT: rename your columns to match these standard names
    # Adjust the mapping below to match your dataset's column names
    # -------------------------------------------------------
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
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in dataset. "
                f"Available columns: {list(df.columns)}. "
                f"Edit the rename mapping in load_amazon_data()."
            )

    return df


def find_review_dataset(search_dirs=None) -> str | None:
    """Search common data folders for a review-style Amazon dataset.
    Returns the first candidate path or None.
    """
    search_dirs = search_dirs or ["data/processed", "data/raw", "data"]
    review_indicators = {"user_id", "reviewerID", "reviewer_id", "reviewer", "overall", "rating", "asin", "product_id"}

    for d in search_dirs:
        p = Path(d)
        if not p.exists():
            continue
        for f in sorted(p.iterdir()):
            if not f.is_file():
                continue
            if f.suffix.lower() not in {".csv", ".json", ".jsonl"}:
                continue
            try:
                cols = []
                if f.suffix.lower() == ".csv":
                    cols = pd.read_csv(f, nrows=0).columns.tolist()
                else:
                    with open(f, "r", encoding="utf-8") as fh:
                        for ln in fh:
                            if ln.strip():
                                first = ln.strip()
                                break
                        else:
                            continue
                    try:
                        obj = json.loads(first)
                        if isinstance(obj, dict):
                            cols = list(obj.keys())
                        elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
                            cols = list(obj[0].keys())
                    except Exception:
                        # if not valid JSON, skip
                        continue

                cols_lower = {c for c in (cols or [])}
                if review_indicators & cols_lower:
                    return str(f)
            except Exception:
                continue
    return None


# ============================================================
# TEST SET CONSTRUCTION
# ============================================================

def build_test_set(df: pd.DataFrame, num_cases: int) -> list[dict]:
    """
    For each eligible user, hold out their most recent review as ground truth.
    The remaining reviews become the user's 'history' for persona construction.
    """
    # Sort by timestamp if available, otherwise by index
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    
    # Filter to users with enough reviews
    user_counts = df["user_id"].value_counts()
    eligible_users = user_counts[user_counts >= MIN_USER_REVIEWS].index.tolist()
    
    print(f"Found {len(eligible_users)} users with >= {MIN_USER_REVIEWS} reviews")
    
    if len(eligible_users) == 0:
        raise ValueError(
            f"No users found with >= {MIN_USER_REVIEWS} reviews. "
            f"Lower MIN_USER_REVIEWS in the CONFIG section."
        )
    
    # Sample users if we have more than needed
    if len(eligible_users) > num_cases:
        rng = np.random.RandomState(42)
        eligible_users = rng.choice(eligible_users, size=num_cases, replace=False).tolist()
    
    test_cases = []
    for uid in eligible_users:
        user_reviews = df[df["user_id"] == uid].copy()
        
        # Hold out the last review as ground truth
        ground_truth = user_reviews.iloc[-1]
        history = user_reviews.iloc[:-1]
        
        # Build a simple persona from their review history
        avg_rating = history["rating"].mean()
        num_reviews = len(history)
        products_reviewed = history["product_id"].tolist()
        product_titles = history["product_title"].tolist() if "product_title" in history.columns else []
        
        # Construct persona description from history
        persona = {
            "name": f"User_{uid[:8]}" if isinstance(uid, str) else f"User_{uid}",
            "user_id": uid,
            "avg_rating": round(float(avg_rating), 1),
            "review_count": int(num_reviews),
            "recent_products": products_reviewed[-5:],
            "recent_product_titles": product_titles[-5:],
            "traits": _infer_traits(history),
        }
        
        test_cases.append({
            "persona": persona,
            "ground_truth_product": ground_truth["product_id"],
            "ground_truth_rating": float(ground_truth["rating"]),
            "history_product_ids": products_reviewed,
        })
    
    print(f"Built {len(test_cases)} test cases")
    return test_cases


def _infer_traits(history: pd.DataFrame) -> list[str]:
    """Infer simple traits from review history for persona construction."""
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
    
    if history["rating"].std() < 0.5:
        traits.append("consistent in ratings")
    elif history["rating"].std() > 1.5:
        traits.append("varied opinions")
    
    return traits


# ============================================================
# API CALLS
# ============================================================

def call_recommend(persona: dict, message: str, session_id: str = None) -> dict:
    """Call the /recommend endpoint and return the full response."""
    payload = {
        "persona_description": persona,
        "message": message,
        "session_id": session_id or f"eval_{int(time.time())}",
    }
    
    try:
        resp = requests.post(
            f"{API_BASE_URL}/recommend",
            json=payload,
            timeout=150,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  API call failed: {e}")
        return None


def extract_recommended_products(response: dict) -> list[str]:
    """
    Extract product IDs from the /recommend response.
    Adjust this based on your actual response structure.
    """
    if not response:
        return []
    
    debug = response.get("debug", {})
    products = debug.get("products", [])
    
    # If your products are returned as full IDs like "Electronics:B079ZNSHDN",
    # strip the category prefix to match ground truth format (e.g. B079ZNSHDN).
    products = [p.split(":")[-1] if ":" in p else p for p in products]
    
    return products


# ============================================================
# METRIC COMPUTATION
# ============================================================

def compute_hit_rate(test_results: list[dict]) -> float:
    """
    Hit Rate@K: fraction of test cases where the ground truth item
    appears anywhere in the top-K recommendations.
    """
    hits = 0
    total = 0
    
    for result in test_results:
        if result["recommended_products"] is not None:
            total += 1
            if result["ground_truth_product"] in result["recommended_products"]:
                hits += 1
    
    if total == 0:
        return 0.0
    return hits / total


def compute_ndcg(test_results: list[dict], k: int = TOP_K) -> float:
    """
    NDCG@K: measures ranking quality — rewards placing the relevant
    item higher in the list.

    Computes DCG directly from the pre-ranked recommendation list
    instead of using sklearn's ndcg_score (which expects predicted
    scores and re-sorts, causing false positives on tied scores).
    """
    ndcg_scores = []

    for result in test_results:
        recommended = result["recommended_products"]
        if recommended is None:
            continue

        gt = result["ground_truth_product"]

        # Build relevance vector: 1 if ground truth, 0 otherwise
        relevance = [1.0 if pid == gt else 0.0 for pid in recommended[:k]]

        # If the ground-truth item is not in the list, NDCG is 0
        if sum(relevance) == 0:
            ndcg_scores.append(0.0)
            continue

        # DCG: sum of rel_i / log2(i + 2)  (i is 0-indexed, so rank = i+1)
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))

        # IDCG: best possible case — single relevant item at rank 1
        idcg = 1.0 / np.log2(2)  # = 1.0

        ndcg_scores.append(dcg / idcg)

    if not ndcg_scores:
        return 0.0
    return float(np.mean(ndcg_scores))


# ============================================================
# TASK A: RANKING QUALITY EVALUATION
# ============================================================

def run_ranking_evaluation(test_cases: list[dict]) -> dict:
    """Run the main ranking evaluation: Hit Rate@10 and NDCG@10."""
    print("\n" + "=" * 60)
    print("TASK B EVALUATION: Ranking Quality")
    print("=" * 60)
    
    test_results = []
    
    for i, case in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] Evaluating user: {case['persona']['name']}")
        
        # Generic recommendation prompt
        # Build a personalized prompt from the user's actual product history
        titles = case["persona"].get("recent_product_titles", [])
        if titles:
            titles_str = ", ".join(t for t in titles[-5:] if t)
            message = f"I recently bought: {titles_str}. What else would you recommend for me?"
        else:
            message = "Based on my purchase history, what products would you recommend for me?"
        
        response = call_recommend(
            persona=case["persona"],
            message=message,
        )
        
        recommended = extract_recommended_products(response)
        gt = case["ground_truth_product"]
        hit = gt in recommended
        
        print(f"  Ground truth: {gt}")
        print(f"  Recommended:  {recommended[:5]}{'...' if len(recommended) > 5 else ''}")
        print(f"  Hit: {'YES' if hit else 'no'}")
        
        test_results.append({
            "user": case["persona"]["name"],
            "ground_truth_product": gt,
            "recommended_products": recommended,
            "hit": hit,
        })
        
        # Small delay to avoid rate-limiting
        time.sleep(0.5)
    
    hit_rate = compute_hit_rate(test_results)
    ndcg = compute_ndcg(test_results)
    
    print("\n" + "-" * 40)
    print(f"Hit Rate@{TOP_K}:  {hit_rate:.4f} ({hit_rate*100:.1f}%)")
    print(f"NDCG@{TOP_K}:      {ndcg:.4f}")
    print(f"Test cases:       {len(test_results)}")
    print("-" * 40)
    
    return {
        "hit_rate": round(hit_rate, 4),
        "ndcg": round(ndcg, 4),
        "num_test_cases": len(test_results),
        "num_hits": sum(1 for r in test_results if r["hit"]),
        "details": test_results,
    }


# ============================================================
# CROSS-DOMAIN BRIDGE EVALUATION
# ============================================================

# Predefined cross-domain test prompts with expected bridge behavior
CROSS_DOMAIN_TEST_CASES = [
    {
        "message": "I need to keep working during power outages",
        "expected_intent": "remote_work_setup",
        "should_bridge": True,
        "bridge_description": "Should recommend cafes/workspaces with backup power",
    },
    {
        "message": "I want to start working out at home",
        "expected_intent": "fitness_journey",
        "should_bridge": True,
        "bridge_description": "Should recommend gyms, fitness studios, or parks",
    },
    {
        "message": "I'm preparing for my friend's owambe next month",
        "expected_intent": "owambe_prep",
        "should_bridge": True,
        "bridge_description": "Should recommend tailors, event venues, or photo studios",
    },
    {
        "message": "I want to get into serious cooking at home",
        "expected_intent": "cooking_exploration",
        "should_bridge": True,
        "bridge_description": "Should recommend cooking classes, food markets, or specialty restaurants",
    },
    {
        "message": "I need a new phone case",
        "expected_intent": "general_browsing",
        "should_bridge": False,
        "bridge_description": "Should NOT bridge — low-intent, single-domain purchase",
    },
    {
        "message": "Looking for a USB cable",
        "expected_intent": "general_browsing",
        "should_bridge": False,
        "bridge_description": "Should NOT bridge — commodity purchase, no lifestyle signal",
    },
    {
        "message": "I want to set up a cozy reading corner at home",
        "expected_intent": "reading_habit",
        "should_bridge": True,
        "bridge_description": "Should recommend quiet cafes, libraries, or book clubs",
    },
    {
        "message": "We just had a baby and need to get everything ready",
        "expected_intent": "baby_family_prep",
        "should_bridge": True,
        "bridge_description": "Should recommend family-friendly restaurants, paediatric clinics, or parks",
    },
    {
        "message": "I need a new HDMI cable for my monitor",
        "expected_intent": "general_browsing",
        "should_bridge": False,
        "bridge_description": "Should NOT bridge — utility purchase",
    },
    {
        "message": "Setting up a complete home office since I just went fully remote",
        "expected_intent": "remote_work_setup",
        "should_bridge": True,
        "bridge_description": "Should recommend co-working spaces or cafes with WiFi and power",
    },
]

# Default persona for cross-domain tests
DEFAULT_CROSS_DOMAIN_PERSONA = {
    "name": "Tobi",
    "age": 27,
    "location": "Yaba, Lagos",
    "occupation": "freelance developer",
    "budget": "careful with money",
    "traits": ["budget-conscious", "likes practical gadgets", "works remotely"],
}


def run_cross_domain_evaluation() -> dict:
    """
    Evaluate cross-domain bridge accuracy.
    Tests whether the bridge fires when it should and stays silent when it shouldn't.
    """
    print("\n" + "=" * 60)
    print("CROSS-DOMAIN BRIDGE EVALUATION")
    print("=" * 60)
    
    results = []
    
    for i, case in enumerate(CROSS_DOMAIN_TEST_CASES):
        print(f"\n[{i+1}/{len(CROSS_DOMAIN_TEST_CASES)}] \"{case['message']}\"")
        print(f"  Expected: intent={case['expected_intent']}, bridge={'YES' if case['should_bridge'] else 'NO'}")
        
        response = call_recommend(
            persona=DEFAULT_CROSS_DOMAIN_PERSONA,
            message=case["message"],
        )
        
        if not response:
            print("  SKIPPED — API call failed")
            results.append({**case, "passed": None, "actual_intent": None, "actual_bridged": None})
            continue
        
        debug = response.get("debug", {})
        intent_data = debug.get("intent", {})
        actual_intent = intent_data.get("intent", "unknown")
        actual_confidence = intent_data.get("confidence", 0)
        actual_bridged = debug.get("bridged", False)
        locations = debug.get("locations", [])
        
        # Check if bridge behavior matches expectation
        bridge_correct = (case["should_bridge"] == actual_bridged)
        
        # Check if intent category roughly matches (flexible — LLM may use different naming)
        intent_correct = (
            case["expected_intent"].lower().replace("_", " ")
            in actual_intent.lower().replace("_", " ")
        ) or actual_intent == case["expected_intent"]
        
        passed = bridge_correct  # Bridge behavior is the primary check
        
        print(f"  Actual:   intent={actual_intent} ({actual_confidence:.2f}), bridge={'YES' if actual_bridged else 'NO'}")
        print(f"  Bridge correct: {'PASS' if bridge_correct else 'FAIL'}")
        print(f"  Locations returned: {locations[:2] if locations else 'none'}")
        
        results.append({
            "message": case["message"],
            "expected_intent": case["expected_intent"],
            "actual_intent": actual_intent,
            "confidence": actual_confidence,
            "should_bridge": case["should_bridge"],
            "actual_bridged": actual_bridged,
            "bridge_correct": bridge_correct,
            "intent_correct": intent_correct,
            "locations": locations,
            "passed": passed,
        })
        
        time.sleep(0.5)
    
    # Compute summary metrics
    valid_results = [r for r in results if r["passed"] is not None]
    num_passed = sum(1 for r in valid_results if r["passed"])
    total = len(valid_results)
    accuracy = num_passed / total if total > 0 else 0
    
    # Breakdown: bridge-should-fire cases vs bridge-should-not-fire cases
    should_bridge = [r for r in valid_results if r["should_bridge"]]
    should_not_bridge = [r for r in valid_results if not r["should_bridge"]]
    
    bridge_precision = (
        sum(1 for r in should_bridge if r["passed"]) / len(should_bridge)
        if should_bridge else 0
    )
    bridge_restraint = (
        sum(1 for r in should_not_bridge if r["passed"]) / len(should_not_bridge)
        if should_not_bridge else 0
    )
    
    print("\n" + "-" * 40)
    print(f"Cross-Domain Accuracy:  {accuracy:.1%} ({num_passed}/{total})")
    print(f"Bridge Precision:       {bridge_precision:.1%} (fires when it should)")
    print(f"Bridge Restraint:       {bridge_restraint:.1%} (silent when it should be)")
    print("-" * 40)
    
    return {
        "accuracy": round(accuracy, 4),
        "bridge_precision": round(bridge_precision, 4),
        "bridge_restraint": round(bridge_restraint, 4),
        "num_passed": num_passed,
        "num_total": total,
        "details": results,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="OmniNaija Lite Evaluation")
    parser.add_argument("--quick", action="store_true", help="Quick run with 10 test cases")
    parser.add_argument("--cross-domain", action="store_true", help="Only run cross-domain eval")
    parser.add_argument("--num-cases", type=int, default=DEFAULT_NUM_TEST_CASES, help="Number of test cases")
    args = parser.parse_args()
    
    print("=" * 60)
    print("OmniNaija Lite — Evaluation Suite")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API: {API_BASE_URL}")
    print("=" * 60)
    
    # Check API is running
    try:
        requests.get(f"{API_BASE_URL}/docs", timeout=5)
        print("API connection: OK")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot reach API at {API_BASE_URL}")
        print("Start your FastAPI server first: uvicorn main:app --reload")
        return
    
    all_results = {}
    
    # --- Ranking evaluation (skip if --cross-domain only) ---
    if not args.cross_domain:
        num_cases = 10 if args.quick else args.num_cases
        print(f"\nLoading Amazon data from: {AMAZON_DATA_PATH}")
        
        try:
            df = load_amazon_data(AMAZON_DATA_PATH)
            print(f"Loaded {len(df)} reviews")
            
            test_cases = build_test_set(df, num_cases=num_cases)
            ranking_results = run_ranking_evaluation(test_cases)
            all_results["ranking"] = ranking_results
        except FileNotFoundError:
            print(f"\nWARNING: Dataset not found at '{AMAZON_DATA_PATH}'")
            # Try to auto-detect a review-style dataset in common folders
            candidate = find_review_dataset()
            if candidate:
                print(f"Found candidate dataset: {candidate} — retrying load")
                try:
                    df = load_amazon_data(candidate)
                    print(f"Loaded {len(df)} reviews from {candidate}")
                    test_cases = build_test_set(df, num_cases=num_cases)
                    ranking_results = run_ranking_evaluation(test_cases)
                    all_results["ranking"] = ranking_results
                except Exception as e:
                    print(f"Failed to load candidate dataset: {e}")
                    print("Skipping ranking evaluation, running cross-domain only.\n")
            else:
                print("Update AMAZON_DATA_PATH in the CONFIG section, then re-run.")
                print("Skipping ranking evaluation, running cross-domain only.\n")
        except ValueError as e:
            print(f"\nWARNING: {e}")
            # If columns are missing, see if we can auto-detect a review dataset
            candidate = find_review_dataset()
            if candidate:
                print(f"Found candidate dataset: {candidate} — retrying load")
                try:
                    df = load_amazon_data(candidate)
                    print(f"Loaded {len(df)} reviews from {candidate}")
                    test_cases = build_test_set(df, num_cases=num_cases)
                    ranking_results = run_ranking_evaluation(test_cases)
                    all_results["ranking"] = ranking_results
                except Exception as e2:
                    print(f"Failed to load candidate dataset: {e2}")
                    print("Skipping ranking evaluation.\n")
            else:
                print("Skipping ranking evaluation.\n")
    
    # --- Cross-domain evaluation (always runs) ---
    cross_domain_results = run_cross_domain_evaluation()
    all_results["cross_domain"] = cross_domain_results
    
    # --- Save results ---
    output_path = Path(RESULTS_OUTPUT_PATH)
    all_results["timestamp"] = datetime.now().isoformat()
    all_results["config"] = {
        "api_url": API_BASE_URL,
        "top_k": TOP_K,
        "bridge_threshold": BRIDGE_CONFIDENCE_THRESHOLD,
    }
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    # --- Final summary ---
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    if "ranking" in all_results:
        r = all_results["ranking"]
        print(f"  Hit Rate@{TOP_K}:           {r['hit_rate']:.4f} ({r['hit_rate']*100:.1f}%)")
        print(f"  NDCG@{TOP_K}:               {r['ndcg']:.4f}")
    
    cd = all_results["cross_domain"]
    print(f"  Cross-Domain Accuracy:   {cd['accuracy']:.1%}")
    print(f"  Bridge Precision:        {cd['bridge_precision']:.1%}")
    print(f"  Bridge Restraint:        {cd['bridge_restraint']:.1%}")
    print("=" * 60)
    print("\nPaste these numbers into Section 5 of your solution paper.")


if __name__ == "__main__":
    main()
    