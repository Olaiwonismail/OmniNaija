import chromadb
from sentence_transformers import SentenceTransformer
import sys

def main():
    chroma_dir = "chroma_db"
    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection(name="venue_locations")
    except Exception as e:
        print(f"Error accessing collection: {e}")
        sys.exit(1)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    queries = [
        "cafe with generator and WiFi in Yaba",
        "gym in Lekki",
        "event venue in Victoria Island",
        "quiet family restaurant Lagos"
    ]

    print("--- Venue Sanity Queries ---")
    for query in queries:
        print(f"\nQuery: '{query}'")
        query_embedding = model.encode([query]).tolist()
        
        try:
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=5
            )
            
            if not results['documents'] or not results['documents'][0]:
                print("  No results found.")
                continue

            for i in range(len(results['documents'][0])):
                name = results['metadatas'][0][i]['name']
                area = results['metadatas'][0][i]['area']
                print(f"  {i+1}. {name} ({area})")
        except Exception as e:
            print(f"  Error during query: {e}")

if __name__ == "__main__":
    main()