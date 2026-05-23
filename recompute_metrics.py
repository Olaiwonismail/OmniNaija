# recompute_metrics.py
import json
import math
from pathlib import Path
from sklearn.metrics import ndcg_score
import numpy as np

def load(path="evaluation_results.json"):
    return json.loads(Path(path).read_text())

def compute_hit_rate(details, k=10):
    hits = 0
    total = 0
    for r in details:
        if r.get("recommended_products") is not None:
            total += 1
            if r["ground_truth_product"] in r.get("recommended_products", [])[:k]:
                hits += 1
    return hits, total, (hits / total if total else 0.0)

def compute_ndcg(details, k=10):
    scores = []
    for r in details:
        recs = r.get("recommended_products") or []
        gt = r["ground_truth_product"]
        rel = [1.0 if pid == gt else 0.0 for pid in recs[:k]]
        while len(rel) < k:
            rel.append(0.0)
        ideal = [1.0] + [0.0]*(k-1)
        try:
            sc = ndcg_score([ideal], [rel], k=k)
        except Exception:
            sc = 0.0
        scores.append(sc)
    return float(np.mean(scores)) if scores else 0.0

def main():
    obj = load()
    details = obj["ranking"]["details"]
    stored = obj["ranking"]
    hits, total, hit_rate = compute_hit_rate(details, k=10)
    ndcg = compute_ndcg(details, k=10)
    print("Stored (ranking):", {k: stored[k] for k in ("hit_rate","ndcg","num_hits","num_test_cases")})
    print(f"Recomputed: hit_rate={hit_rate:.4f} ({hits}/{total}), ndcg={ndcg:.4f}")
    if abs(hit_rate - stored.get("hit_rate",0)) > 1e-6 or abs(ndcg - stored.get("ndcg",0)) > 1e-6:
        print("DISCREPANCY DETECTED")
        for i,r in enumerate(details):
            if r.get("recommended_products"):
                if r["ground_truth_product"] in r["recommended_products"][:10]:
                    print("  HIT at case", i, r["user"], r["ground_truth_product"])
        print("If all hits are false, NDCG should be 0.0 for binary relevance. Stored NDCG looks incorrect.")
    else:
        print("Stored metrics match recomputed values.")

if __name__ == '__main__':
    main()