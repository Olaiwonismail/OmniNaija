import json

import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

def build_description(v):
    """
    Creates a rich, plain-English description string for a venue to improve 
    semantic search retrieval.
    """
    categories = ", ".join(v.get('categories', []))
    
    # Convert booleans to searchable text
    gen = "has a generator" if v.get('has_generator') else "no generator"
    wifi = "has WiFi" if v.get('has_wifi') else "no WiFi"
    late = "is open late" if v.get('open_late') else "not open late"
    fam = "is family friendly" if v.get('family_friendly') else "not family friendly"
    park = "has parking" if v.get('parking') else "no parking"
    
    return (
        f"{v['name']} is a {categories} located in {v['area']}, {v['city']}. "
        f"Address: {v['address']}. Hours: {v['hours']}. "
        f"Amenities: {gen}, {wifi}, {late}, {fam}, {park}. "
        f"Rating: {v.get('stars', 0)} stars from {v.get('review_count', 0)} reviews."
    )

def main():
    # Paths
    input_file = Path("data/processed/nigerian_venues.jsonl")
    chroma_dir = Path("chroma_db")
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found.")
        return

    print(f"Loading model: all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"Initializing ChromaDB at {chroma_dir}...")
    client = chromadb.PersistentClient(path=str(chroma_dir))
    
    # Create or get the venue_locations collection
    collection = client.get_or_create_collection(name="venue_locations")

    ids = []
    documents = []
    metadatas = []

    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            venue = json.loads(line)
            
            # Create the searchable document
            desc = build_description(venue)
            
            # Use name or a combination as ID (assuming names are unique enough for this scale)
            # Or better, use a slugified name or just index
            venue_id = venue['name'].replace(" ", "_").lower() + "_" + str(venue['latitude'])
            
            ids.append(venue_id)
            documents.append(desc)
            
            # Metadata for filtering (Chroma supports basic types)
            # Note: Chroma metadata values must be str, int, float, or bool
            metadata = {
                "name": venue["name"],
                "city": venue["city"],
                "area": venue["area"],
                "has_generator": bool(venue.get("has_generator")),
                "has_wifi": bool(venue.get("has_wifi")),
                "family_friendly": bool(venue.get("family_friendly")),
                "stars": float(venue.get("stars", 0)),
                "categories": ", ".join(venue.get("categories", []))
            }
            metadatas.append(metadata)

    if not ids:
        print("No data found in JSONL.")
        return

    print(f"Generating embeddings for {len(ids)} venues...")
    embeddings = model.encode(documents).tolist()

    print("Upserting to ChromaDB...")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Successfully built 'venue_locations' collection with {len(ids)} entries.")

if __name__ == "__main__":
    main()