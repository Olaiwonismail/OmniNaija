#!/usr/bin/env python3
"""Build a persistent Chroma collection from processed Amazon JSONL files."""

import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


def build_document(row):
    review_snippets = "; ".join([r.get("text", "") for r in row.get("bundled_reviews", [])])
    title = row.get("title", "")
    description = row.get("description", "")
    return f"{title}. {description}. Reviews: {review_snippets}"


def build_metadata(row):
    """
    ChromaDB only accepts str, int, float, bool, or None as metadata values.
    Anything complex (dict, list) must be JSON-serialized to a string.
    """
    return {
        # Scalar fields — pass through directly
        "asin":         row.get("asin") or "",
        "parent_asin":  row.get("parent_asin") or "",
        "title":        row.get("title") or "",
        "brand":        row.get("brand") or "",
        "store":        row.get("store") or "",
        "category":     row.get("category") or "",
        "price":        float(row["price"]) if row.get("price") is not None else 0.0,
        "avg_rating":   float(row.get("avg_rating") or 0.0),
        "review_count": int(row.get("review_count") or 0),
        # Complex fields — serialize to JSON strings so M1 can json.loads() them
        "features":        json.dumps(row.get("features") or []),
        "images":          json.dumps(row.get("images") or []),
        "details":         json.dumps(row.get("details") or {}),
        "bundled_reviews": json.dumps(row.get("bundled_reviews") or []),
    }


def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB collection from Amazon products JSONL.")
    parser.add_argument("--input-dir", default="data/processed", help="Directory with JSONL files")
    parser.add_argument("--persist-dir", default="chroma_db", help="Persistent ChromaDB directory")
    parser.add_argument("--collection", default="amazon_products", help="Chroma collection name")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    jsonl_files = sorted(input_dir.glob("amazon_*.jsonl"))
    if not jsonl_files:
        raise SystemExit("No processed amazon_*.jsonl files found.")

    client = chromadb.PersistentClient(path=str(Path(args.persist_dir)))

    # Drop and recreate collection to avoid stale data on re-runs
    try:
        client.delete_collection(name=args.collection)
    except Exception:
        pass
    collection = client.create_collection(name=args.collection)

    model = SentenceTransformer(args.model)

    documents = []
    metadatas = []
    ids = []

    def flush():
        if not documents:
            return
        embeddings = model.encode(documents, batch_size=args.batch_size, show_progress_bar=False)
        collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
        documents.clear()
        metadatas.clear()
        ids.clear()

    total = 0
    for path in jsonl_files:
        print(f"Loading {path.name}...")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                asin = row.get("asin")
                category = row.get("category", "unknown")
                doc_id = f"{category}:{asin}"

                documents.append(build_document(row))
                metadatas.append(build_metadata(row))
                ids.append(doc_id)
                total += 1

                if len(documents) >= args.batch_size:
                    flush()

    flush()
    print(f"\nDone. Collection '{args.collection}' size: {collection.count()}")


if __name__ == "__main__":
    main()