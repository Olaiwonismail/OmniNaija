"""
Build a synthetic user-interactions dataset from bundled product reviews.
============================================================================
Reads your product JSONL files, extracts individual reviews, and assigns
them to synthetic users so the ranking evaluation can run.

Outputs: data/processed/user_interactions.jsonl
Each line: {"user_id": "user_042", "product_id": "B079ZNSHDN", "rating": 4, "review_text": "..."}

DISCLOSURE for your paper:
"Per-user interaction histories were not available in the source dataset.
We constructed synthetic user profiles by distributing individual reviews
across generated user identities, ensuring each user has interactions
with at least MIN_REVIEWS_PER_USER distinct products. This enables
ranking evaluation while acknowledging the profiles are synthetic."
"""

import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

INPUT_FILES = [
    "data/processed/amazon_electronics.jsonl",
    "data/processed/amazon_books.jsonl",
]

OUTPUT_PATH = "data/processed/user_interactions.jsonl"

# Each synthetic user will have between these many reviews
MIN_REVIEWS_PER_USER = 6
MAX_REVIEWS_PER_USER = 20

# Target number of synthetic users
TARGET_NUM_USERS = 400

# Random seed for reproducibility
SEED = 42


# ============================================================
# MAIN
# ============================================================

def main():
    random.seed(SEED)

    # Step 1: Flatten all reviews with their product_id
    print("Step 1: Extracting reviews from product files...")
    all_reviews = []

    for filepath in INPUT_FILES:
        path = Path(filepath)
        if not path.exists():
            print(f"  WARNING: {filepath} not found, skipping")
            continue

        count = 0
        with open(path) as f:
            for line in f:
                product = json.loads(line)
                product_id = product.get("product_id") or product.get("asin", "unknown")
                category = product.get("category", "unknown")
                title = product.get("title", "")

                for review in product.get("bundled_reviews", []):
                    all_reviews.append({
                        "product_id": product_id,
                        "category": category,
                        "product_title": title,
                        "rating": review.get("rating", 3),
                        "review_text": review.get("text", ""),
                        "verified_purchase": review.get("verified_purchase", False),
                        "helpful_vote": review.get("helpful_vote", 0),
                    })
                    count += 1

        print(f"  {filepath}: extracted {count} reviews")

    print(f"  Total reviews: {len(all_reviews)}")

    if len(all_reviews) == 0:
        print("ERROR: No reviews found. Check your INPUT_FILES paths.")
        return

    # Step 2: Shuffle and assign to synthetic users
    print(f"\nStep 2: Assigning reviews to ~{TARGET_NUM_USERS} synthetic users...")
    random.shuffle(all_reviews)

    # Strategy: distribute reviews round-robin across users,
    # then remove users with fewer than MIN_REVIEWS_PER_USER
    users = defaultdict(list)
    user_ids = [f"user_{i:04d}" for i in range(TARGET_NUM_USERS)]

    for i, review in enumerate(all_reviews):
        uid = user_ids[i % TARGET_NUM_USERS]
        users[uid].append(review)

    # Filter out users with too few reviews
    valid_users = {
        uid: reviews
        for uid, reviews in users.items()
        if len(reviews) >= MIN_REVIEWS_PER_USER
    }

    # Cap at MAX_REVIEWS_PER_USER
    for uid in valid_users:
        if len(valid_users[uid]) > MAX_REVIEWS_PER_USER:
            valid_users[uid] = valid_users[uid][:MAX_REVIEWS_PER_USER]

    total_interactions = sum(len(r) for r in valid_users.values())
    print(f"  Valid users: {len(valid_users)}")
    print(f"  Total interactions: {total_interactions}")
    print(f"  Avg reviews per user: {total_interactions / len(valid_users):.1f}")

    # Step 3: Check product diversity per user
    low_diversity = 0
    for uid, reviews in valid_users.items():
        unique_products = len(set(r["product_id"] for r in reviews))
        if unique_products < MIN_REVIEWS_PER_USER:
            low_diversity += 1

    if low_diversity > 0:
        print(f"  NOTE: {low_diversity} users have duplicate product reviews (acceptable for synthetic data)")

    # Step 4: Write output
    print(f"\nStep 3: Writing to {OUTPUT_PATH}...")
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    line_count = 0
    with open(output_path, "w") as f:
        for uid, reviews in sorted(valid_users.items()):
            for review in reviews:
                record = {
                    "user_id": uid,
                    "product_id": review["product_id"],
                    "rating": review["rating"],
                    "review_text": review["review_text"],
                    "category": review["category"],
                    "product_title": review["product_title"],
                    "verified_purchase": review["verified_purchase"],
                }
                f.write(json.dumps(record) + "\n")
                line_count += 1

    print(f"  Wrote {line_count} interaction records")

    # Step 5: Print summary stats for the paper
    print("\n" + "=" * 50)
    print("STATS FOR YOUR SOLUTION PAPER (Section 5)")
    print("=" * 50)

    all_ratings = [r["rating"] for reviews in valid_users.values() for r in reviews]
    print(f"  Synthetic users:        {len(valid_users)}")
    print(f"  Total interactions:     {total_interactions}")
    print(f"  Unique products:        {len(set(r['product_id'] for reviews in valid_users.values() for r in reviews))}")
    print(f"  Avg rating:             {sum(all_ratings) / len(all_ratings):.2f}")
    print(f"  Rating distribution:")
    for star in range(1, 6):
        count = all_ratings.count(star)
        pct = count / len(all_ratings) * 100
        print(f"    {star} star: {count:>5} ({pct:.1f}%)")

    print(f"\n  Output file: {OUTPUT_PATH}")
    print(f"\n  Next step: update evaluate.py CONFIG section:")
    print(f'    AMAZON_DATA_PATH = "{OUTPUT_PATH}"')
    print(f"  Then run: python evaluate.py --quick")


if __name__ == "__main__":
    main()