from agent.graph import retrieve_products

queries = [
    "power bank for working during blackouts",
    "phone case",
]

for q in queries:
    print("QUERY:", q)
    results = retrieve_products(q, top_k=3)
    for r in results:
        print(f"- {r.get('product_id')} | {r.get('category')} | {r.get('title')}")
    print()
