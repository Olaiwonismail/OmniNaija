#!/usr/bin/env python3
"""Run sanity queries against the local Chroma collection."""

import argparse

import chromadb
from sentence_transformers import SentenceTransformer

QUERIES = [
    "power bank for working during outages",
    "yoga mat for home workouts",
    "Nigerian cookbook",
    "noise-cancelling headphones for coding",
    "baby formula",
]


def main():
    parser = argparse.ArgumentParser(description="Run sanity queries against ChromaDB.")
    parser.add_argument("--persist-dir", default="chroma_db", help="Persistent ChromaDB directory")
    parser.add_argument("--collection", default="amazon_products", help="Chroma collection name")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--top-k", type=int, default=5, help="Top K results")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.persist_dir)
    collection = client.get_collection(name=args.collection)
    model = SentenceTransformer(args.model)

    for query in QUERIES:
        embedding = model.encode(query)
        result = collection.query(query_embeddings=[embedding], n_results=args.top_k)
        print("\n---")
        print(f"Query: {query}")
        for doc_id, metadata in zip(result["ids"][0], result["metadatas"][0]):
            title = metadata.get("title", "")
            category = metadata.get("category", "")
            print(f"- {doc_id} | {category} | {title}")


if __name__ == "__main__":
    main()
