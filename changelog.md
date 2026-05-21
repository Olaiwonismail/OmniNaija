# Progress Changelog — OmniNaija Day 1

## Session Overview
**Mission:** Stand up the Amazon Vector Store to unblock Agent Core (M1) and Persona Development (M4).
**Status:** Day 1 complete — Amazon vector store built and validated.

## Quick Start (Instant Read)

### Data Flow (Day 1)
Reviews (stream) → group by `parent_asin` → filter ≥5 reviews & avg rating ≥3.0 → merge meta → JSONL → ChromaDB

### Key Scripts
- `scripts/amazon_ingest.py` — builds `data/processed/amazon_<category>.jsonl`
- `scripts/build_chroma.py` — embeds and persists to `chroma_db/`
- `scripts/sanity_query.py` — runs 5 retrieval sanity checks

### Run Commands
```bash
python scripts/amazon_ingest.py --per-category 2000
python scripts/build_chroma.py --persist-dir chroma_db
python scripts/sanity_query.py --persist-dir chroma_db
```

---

### Done So Far

#### 1. Repo Scaffolding
- [x] Created root-level folder structure (`data/raw`, `data/processed`, `scripts/`, `api/`, `ui/`, `paper/`, `personas/`, `chroma_db/`).
- [x] Configured `.gitignore` to prevent leaking large data or secrets.
- [x] Initialized `README.md` with complete Hackathon PRD and Day 1 setup instructions.
- [x] Added `docker-compose.yml` (stub) and `.env.example`.

#### 2. Data Infrastructure
- [x] **Requirements:** Created `requirements.txt` with `datasets`, `chromadb`, `sentence-transformers`, `fsspec`, and `pandas`.
- [x] **Schema:** Defined `SCHEMA.md` for bundled product rows (meta + top 3 reviews).
- [x] **Ingestion Script:**
    - Finalized a reviews-first ingestion flow with early stopping for speed.
    - Switched to `fsspec` streaming to avoid JSON schema casting errors.
    - Enforced quality filters (review count ≥ 5, avg rating ≥ 3.0).
    - Standardized on `parent_asin` for product grouping.
- [x] **Vector Store Stand-up:** Built a persistent ChromaDB collection using `all-MiniLM-L6-v2`.
- [x] **Sanity Testing:** Verified 5 Nigerian-context queries.

#### 3. Environment & Tools
- [x] Installed all dependencies in the Codespace environment.
- [x] Verified Hugging Face dataset layout for `McAuley-Lab/Amazon-Reviews-2023`.

---

### Day 1 Outcomes
- **Ingestion:** ~6,000 products across Electronics, Books, Home & Kitchen (2k each).
- **Performance:** Reviews-first flow scanned ~370k reviews total (early stopping).
- **Chroma:** Persistent collection built and embedded.
- **Retrieval Quality:** Strong matches for outage power, headphones, yoga, and baby care queries.

---

## Session: Data Schema Expansion (Day 2)

### Done So Far
#### 1. Venue Schema
- [x] Created `data/venue_schema.json` to define the structure for Nigerian venue data (name, address, coordinates, facilities, etc.).
