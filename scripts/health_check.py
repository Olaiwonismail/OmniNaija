import chromadb
from sentence_transformers import SentenceTransformer
import pathlib
import sys

def test_embeddings():
    print("--- Testing Embedding Model ---")
    try:
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        text = "This is a health check test."
        embedding = model.encode([text])
        dim = embedding.shape[1]
        if dim == 384:
            print(f"✅ Embeddings: SUCCESS (Dim: {dim})")
            return model
        else:
            print(f"❌ Embeddings: FAILED (Expected dim 384, got {dim})")
            return None
    except Exception as e:
        print(f"❌ Embeddings: FAILED (Error: {e})")
        return None

def main():
    chroma_dir = pathlib.Path("chroma_db")
    if not chroma_dir.exists():
        print(f"❌ ERROR: ChromaDB directory '{chroma_dir}' not found.")
        sys.exit(1)

    model = test_embeddings()
    if not model:
        print("\n🛑 HEALTH CHECK FAILED: Embedding model issue.")
        sys.exit(1)

    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize ChromaDB: {e}")
        sys.exit(1)

    collections_to_test = [
        {"name": "amazon_products", "query": "portable charger"},
        {"name": "venue_locations", "query": "cafe with wifi in Yaba"}
    ]

    print(f"\n--- Testing ChromaDB Collections ---")
    all_passed = True
    for item in collections_to_test:
        col_name = item["name"]
        query = item["query"]
        
        try:
            collection = client.get_collection(name=col_name)
            print(f"✅ Collection '{col_name}': LOADED")
            
            # Run query
            query_emb = model.encode([query]).tolist()
            results = collection.query(query_embeddings=query_emb, n_results=1)
            
            if results and results['documents'] and len(results['documents'][0]) > 0:
                # Print a snippet of the result to prove it's real
                snippet = results['documents'][0][0][:60].replace('\n', ' ')
                print(f"✅ Query '{query}': SUCCESS (Match: {snippet}...")
            else:
                print(f"⚠️ Query '{query}': WARNING (No results found)")
                all_passed = False
        except Exception as e:
            print(f"❌ Collection '{col_name}': FAILED (Error: {e})")
            all_passed = False

    print("\n==============================")
    if all_passed:
        print("🚀 HEALTH CHECK PASSED")
    else:
        print("🛑 HEALTH CHECK FAILED")
    print("==============================\n")

if __name__ == "__main__":
    main()