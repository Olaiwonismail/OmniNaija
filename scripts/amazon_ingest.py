#!/usr/bin/env python3
"""
Amazon Reviews 2023 ingestion — REVIEWS-FIRST approach.

Instead of sampling products from meta then hunting for their reviews
(which requires scanning millions of rows), this script:

1. Streams reviews → groups by parent_asin → stops once we have enough
   products with ≥ min_reviews each.
2. Streams meta → keeps only products found in step 1.
3. Merges and writes bundled JSONL.

This is fast because we stop scanning as soon as we hit the target count.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import fsspec

HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"

CATEGORIES = {
    "Electronics": "Electronics",
    "Books": "Books",
    "Home_and_Kitchen": "Home_and_Kitchen",
}


def parse_int(value, default=0):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        value = value.replace(",", "").strip()
        try:
            return int(value)
        except ValueError:
            return default
    return default


def parse_float(value, default=None):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "").strip()
        try:
            return float(value)
        except ValueError:
            return default
    return default


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join([str(item) for item in value if item is not None])
    return str(value)


def iter_jsonl(path):
    """Stream JSONL from HuggingFace Hub."""
    with fsspec.open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def top_reviews(reviews, top_k=3):
    """Sort by helpful_vote desc, then rating desc. Return top K."""
    return sorted(
        reviews,
        key=lambda r: (r.get("helpful_vote", 0), r.get("rating", 0)),
        reverse=True,
    )[:top_k]


# ─── STEP 1: Stream reviews, collect products with enough reviews ────────────

def collect_reviews(file_key, min_reviews, target_products, max_reviews_per_product=10):
    """
    Stream reviews and group by parent_asin.
    Stop as soon as we have `target_products` products with >= min_reviews each.
    Returns: dict of {parent_asin: [review_entries]}
    """
    review_path = f"hf://datasets/{HF_REPO}/raw/review_categories/{file_key}.jsonl"
    print(f"  Streaming reviews from {file_key}...")

    all_reviews = defaultdict(list)   # parent_asin -> list of review dicts
    complete = set()                   # ASINs that hit min_reviews threshold
    scanned = 0

    for row in iter_jsonl(review_path):
        scanned += 1
        if scanned % 25000 == 0:
            print(f"  [{scanned:,} reviews scanned] → {len(complete)} complete products", end="\r")

        # Stop early once we have enough
        if len(complete) >= target_products:
            print(f"\n  Early stop at {scanned:,} reviews — got {len(complete)} products.")
            break

        parent_asin = row.get("parent_asin") or row.get("asin")
        if not parent_asin:
            continue

        # Skip products we've already completed (save memory)
        if parent_asin in complete and len(all_reviews[parent_asin]) >= max_reviews_per_product:
            continue

        text = row.get("text") or ""
        if not text.strip():
            continue

        entry = {
            "text": text,
            "rating": parse_int(row.get("rating"), default=0),
            "verified_purchase": bool(row.get("verified_purchase", False)),
            "helpful_vote": parse_int(row.get("helpful_vote"), default=0),
        }

        all_reviews[parent_asin].append(entry)

        if len(all_reviews[parent_asin]) >= min_reviews:
            complete.add(parent_asin)

    else:
        print(f"\n  Reached end of reviews at {scanned:,} rows — got {len(complete)} products.")

    # Only return products that met the threshold
    result = {}
    for asin in complete:
        reviews = all_reviews[asin]
        # Keep top reviews by helpfulness
        result[asin] = top_reviews(reviews, top_k=max_reviews_per_product)

    # Free memory
    del all_reviews
    return result


# ─── STEP 2: Stream meta, keep only products we found reviews for ────────────

def collect_meta(file_key, target_asins):
    """
    Stream meta and keep rows matching target_asins.
    """
    meta_path = f"hf://datasets/{HF_REPO}/raw/meta_categories/meta_{file_key}.jsonl"
    print(f"  Streaming meta for {len(target_asins)} products...")

    meta = {}
    remaining = set(target_asins)
    scanned = 0

    for row in iter_jsonl(meta_path):
        scanned += 1
        if scanned % 5000 == 0:
            print(f"  [{scanned:,} meta rows] → {len(meta)} matched", end="\r")

        parent_asin = row.get("parent_asin")
        if parent_asin in remaining:
            meta[parent_asin] = row
            remaining.discard(parent_asin)

        # Stop early if we found all
        if not remaining:
            print(f"\n  Found all {len(meta)} products in meta at row {scanned:,}.")
            break

    else:
        print(f"\n  Meta scan done at {scanned:,} rows. Matched {len(meta)}/{len(target_asins)}.")

    return meta


# ─── STEP 3: Merge and write ─────────────────────────────────────────────────

def build_product_row(parent_asin, category, meta, review_entries):
    title = normalize_text(meta.get("title"))
    description = normalize_text(meta.get("description"))
    features = ensure_list(meta.get("features"))
    details = meta.get("details") or {}
    brand = meta.get("brand") or ""
    store = meta.get("store") or ""
    price = parse_float(meta.get("price"))
    images = ensure_list(meta.get("images"))

    ratings = [r["rating"] for r in review_entries if r.get("rating")]
    avg_rating = sum(ratings) / max(len(ratings), 1)

    return {
        "asin": parent_asin,
        "parent_asin": parent_asin,
        "title": title,
        "description": description,
        "features": features,
        "details": details,
        "brand": brand,
        "store": store,
        "price": price,
        "images": images,
        "category": category,
        "avg_rating": round(avg_rating, 3),
        "review_count": len(review_entries),
        "bundled_reviews": top_reviews(review_entries, top_k=3),
    }


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Amazon Reviews 2023 → bundled product JSONL (reviews-first).")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--per-category", type=int, default=2000, help="Target products per category")
    parser.add_argument("--min-reviews", type=int, default=5, help="Min reviews per product")
    parser.add_argument("--categories", default=",".join(CATEGORIES.keys()),
                        help="Comma-separated category keys (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = [c.strip() for c in args.categories.split(",") if c.strip()]

    for category, file_key in CATEGORIES.items():
        if category not in selected:
            continue

        print(f"\n{'='*60}")
        print(f"  {category}")
        print(f"{'='*60}")

        # Step 1: Reviews first — collect products that have enough reviews
        reviews_data = collect_reviews(
            file_key=file_key,
            min_reviews=args.min_reviews,
            target_products=args.per_category,
        )

        if not reviews_data:
            print(f"  WARNING: No products found with {args.min_reviews}+ reviews. Skipping {category}.")
            continue

        # Step 2: Get meta for those products only
        meta_data = collect_meta(file_key=file_key, target_asins=set(reviews_data.keys()))

        # Step 3: Merge and write
        rows = []
        for parent_asin, review_entries in reviews_data.items():
            meta = meta_data.get(parent_asin)
            if not meta:
                continue  # No meta found for this product
            rows.append(build_product_row(parent_asin, category, meta, review_entries))

        safe_name = category.lower().replace(" & ", "_").replace(" ", "_")
        output_path = output_dir / f"amazon_{safe_name}.jsonl"
        write_jsonl(output_path, rows)

        ratings = [r["avg_rating"] for r in rows]
        avg = sum(ratings) / max(len(ratings), 1)
        print(f"\n  Wrote {len(rows)} products to {output_path}")
        print(f"  Avg rating: {avg:.2f}")


if __name__ == "__main__":
    main()