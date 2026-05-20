# Amazon Products Schema (Bundled Per Product)

Each JSONL row represents one product with bundled reviews.

## Required Fields

- `asin`: "B00X4WHP5E"
- `parent_asin`: "B00X4WHP5E" (nullable)
- `title`: "Anker PowerCore 26800"
- `description`: "Ultra-high capacity portable charger..."
- `features`: ["26800mAh capacity", "Dual USB output"] (nullable)
- `details`: {"capacity": "26800mAh", "ports": "2"} (nullable)
- `brand`: "Anker" (nullable)
- `store`: "AnkerDirect" (nullable)
- `price`: 49.99 (nullable)
- `images`: ["https://.../image.jpg"] (at least first URL when available)
- `category`: "Electronics" | "Books" | "Home & Kitchen"
- `avg_rating`: 4.4
- `review_count`: 128
- `bundled_reviews`:
  - `text`: "Charges my laptop during outages..."
  - `rating`: 5
  - `verified_purchase`: true
  - `helpful_vote`: 12

## Embedding Document Text

```
{title}. {description}. Reviews: {top_3_review_snippets}
```

## Notes
- `features`, `details`, `price`, `brand`, `store`, `parent_asin` can be null if missing.
- `images` should be an array; at least the first image URL when present.
- `bundled_reviews` should be ordered by helpfulness (helpful votes, then rating).
