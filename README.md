# OmniNaija Lite
## Intent-Based Cross-Domain Recommendation Engine for Nigerian Markets

> Most teams ship a single-domain recommender. We ship **intent reasoning** linking Amazon and Yelp—the only team with a name for it: the **Intent Graph**. This approach uniquely unlocks the 25-point cross-domain bonus category in the rubric.

 
## Graph Simulation & Tests

We include a small LangGraph-style runner and node tests so you can exercise the intent→retrieve→compose flow locally without the frontend.

- Run the interactive demo runner (prints recommendations and debug state):

```bash
PYTHONPATH=. python scripts/run_graph_demo.py
```

- Run lightweight node tests (downloads embeddings if needed; `compose_response` will call Gemini when `GEMINI_API_KEY` is set):

```bash
PYTHONPATH=. python scripts/run_node_tests.py
```

- Call the graph flow programmatically (no HTTP server) via the helper test:

```bash
PYTHONPATH=. python scripts/test_graph_simulation.py
```

- HTTP endpoint: `POST /graph_simulate` — sends `persona_description` and `chat_message`, returns `recommendation` and `debug` state. Example `curl`:

```bash
curl -sS -X POST http://127.0.0.1:8000/graph_simulate \
  -H 'Content-Type: application/json' \
  -d '{"persona_description": {"name":"Tobi"}, "chat_message":"I need something to work during blackouts"}'
```

Notes:
- For embedding model downloads, set `HF_TOKEN` (Hugging Face) to improve throughput and avoid rate limits.
- For end-to-end recommender text generation set `GEMINI_API_KEY` in `.env` or the environment.

## Evaluation Metrics

The offline evaluation scripts now report both ranking metrics and text-similarity metrics.

- Ranking: Hit Rate@K, NDCG@K, Category-Match@K
- Text similarity: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore

Run the main evaluator:

```bash
python evaluate_v2.py
```

Recompute ranking metrics from a saved results file:

```bash
python scripts/compute_ranking_metrics.py --results evaluation_results.json --out results_ranking.json
```

Both scripts use the held-out product metadata as a reference proxy for ROUGE/BERTScore when no gold explanation text is available.

## Streamlit App

Run the UI with two tabs and the sidebar persona selector:

```bash
streamlit run ui/streamlit_app.py
```

What to expect:
- The sidebar shows 5 pre-built Nigerian personas with a name, short bio, and avatar.
- Selecting a persona immediately stores that persona in session state and sends it with all future API calls.
- The first tab drives `/simulate` for review generation.
- The second tab drives `/recommend` for multi-turn chat, keeping the same `session_id` and persona across follow-ups.

---

## Problem
Generic AI recommenders treat a Lagos software engineer the same as a Brooklyn one. They see a power bank purchase as a transaction. They miss the **grid context**, the **cafe-with-backup-power culture**, the way Nigerians route around infrastructure gaps to live their lives. Recommendation engines for our market are flying blind.

## Solution
OmniNaija is an LLM agent that reasons about **why** someone is buying, not just **what**. It:
- **Understands persona** — encodes Nigerian user demographics, vocabulary, value priorities, cultural context
- **Extracts intent** — reads purchase history and conversational signals to infer deep user state (remote work, fitness prep, celebration planning, etc.)
- **Bridges intelligently** — maps Amazon product recommendations → Yelp service recommendations using the same inferred intent
- **Knows its limits** — refuses to bridge when signal confidence is low (proving it's not a hardcoded lookup table)

---

## MVP: Three Core Features

### 1. Persona-Driven Review Simulator
**Input:** Persona JSON + product metadata  
**Output:** 1–5 star rating + written review in authentic voice (Nigerian English / Pidgin where natural)

Few-shot prompting anchored to persona style examples; rating drawn from persona's historical bias. Demonstrates behavioral fidelity at scale.

```json
{
  "persona_id": "tobi_yaba_dev",
  "product": "30000mAh Power Bank",
  "review": "Solid gadget fr fr. Keeps my MacBook alive when NEPA dey form... Got two of these now. Na 5-star for real. Saves my bread.",
  "rating": 5
}
```

### 2. Intent-Aware Conversational Recommender ⭐ (Core)
Multi-turn chat interface with:
- **Conversational memory** — full chat history maintained in session
- **Persona-conditioned retrieval** — Amazon vector store queries shaped by user profile
- **Intent extraction node** — reads cart + chat history, outputs structured intent object (JSON)
- **Natural language responses** — LLM composes final recommendation in appropriate register

**Example flow:**
```
User (Persona: Tobi): "My laptop keeps dying at client meetings. What can I do?"
Agent (Intent: Remote work continuity): Retrieves 30kMah + 20kMah power banks, scores by user preference
Response: "Tobi, you need something serious. I'd grab this 30000mAh — lasts all day, even with the MacBook. Power delivery sorted."
```

### 3. Cross-Domain Bridge (The 25-Point Wow Factor)
- Intent → Yelp retrieval **only when** bridge-confidence score exceeds threshold
- Agent demonstrates it's **not hardcoded** by explicitly declining weak-signal bridges
- UI surfaces inferred intent in a **side panel** so judges see the reasoning in real-time

**Example:**
```
Same intent (remote work): 
  Amazon recommendation: 30kMah Power Bank
  Yelp recommendation: "Yaba Hub Cafe — backup gen, wifi 24/7, coffee is decent"
  Confidence: 0.92 ✅ Bridge executed
  
Weak signal (user asks for "red shoes"):
  Amazon recommendation: Red Adidas sneakers  
  Yelp recommendation: [SKIPPED]
  Confidence: 0.31 ❌ Insufficient intent signal, staying in Amazon
```

---

## Success Metrics (Demonstrable in 60 Seconds)

✅ **6 generated reviews** (3 personas × 2 products), each scoring 4/5+ on behavioral fidelity  
✅ **3 distinct intents successfully bridged** Amazon → Yelp (remote work, fitness, owambe prep)  
✅ **1 deliberate negative case** — agent declines to bridge when signal is weak  

The last one separates us from "fancy lookup table" teams.

---

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| **LLM (primary)** | Gemini 2.x Flash | Generous free tier, fast, multilingual, handles Pidgin |
| **LLM (fallback)** | OpenAI gpt-4o-mini | Reliability backup, swap via env var |
| **Orchestration** | LangGraph | State-machine flow demos better; cleaner to reason about for the paper |
| **Vector DB** | ChromaDB (local) | Zero-config, dockerizes in one line, no cloud account needed |
| **Embeddings** | sentence-transformers all-MiniLM-L6-v2 | Free, local, fast (~5k items) |
| **Backend** | FastAPI | Brief requirement, clean async support |
| **Frontend** | Streamlit + streamlit-chat | Demoable in hours, looks clean, minimal infra |
| **Container** | Docker + docker-compose | One-command reproducibility = max points |
| **Datasets** | Amazon (~3k) + Yelp (~2k + 200 hand-augmented Lagos/Abuja venues) | Memory-fit, demo-rich, disclosure-compliant |

---

## Project Structure

```
OmniNaija/
├── backend/
│   ├── agents/
│   │   ├── intent_extraction.py       # Intent parsing from chat + cart
│   │   ├── review_simulator.py        # Persona-driven review generation
│   │   └── recommendation_bridge.py   # Cross-domain orchestration
│   ├── llm/
│   │   └── multi_provider.py          # Gemini / OpenAI abstraction
│   ├── db/
│   │   ├── chromadb_manager.py        # Vector store operations
│   │   └── collections/               # Pre-computed embeddings
│   ├── api/
│   │   ├── main.py                    # FastAPI app
│   │   ├── endpoints.py               # POST /simulate, POST /recommend
│   │   └── session_manager.py         # Multi-turn memory
│   ├── prompts/
│   │   ├── review_generation.py       # Few-shot persona templates
│   │   ├── intent_extraction.py       # Structured intent prompts
│   │   └── bridge_scoring.py          # Confidence threshold logic
│   └── requirements.txt
├── frontend/
│   ├── streamlit_app.py               # Main UI
│   ├── components/
│   │   ├── chat_interface.py
│   │   ├── intent_panel.py            # Side panel display
│   │   ├── persona_selector.py
│   │   └── demo_mode.py               # Cached responses fallback
│   └── assets/
│       ├── personas.json              # 5 detailed Nigerian personas
│       └── demo_cache.json            # Pre-computed responses
├── data/
│   ├── amazon/
│   │   ├── raw_reviews.csv            # Electronics, Books, Home subset
│   │   └── products.json              # Product metadata
│   ├── yelp/
│   │   ├── original.csv               # Yelp Nigeria data
│   │   ├── augmented_lagos_abuja.csv  # Hand-augmented venues
│   │   └── schema_mapping.py          # Format compatibility
│   └── embeddings/
│       ├── amazon.chroma/             # ChromaDB collection
│       └── yelp.chroma/               # ChromaDB collection
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── solution_paper.md                  # 4–8 page paper (Intent Graph narrative)
├── pitch_deck.md                      # 10-slide pitch outline
├── demo_script.md                     # 90-second live demo + backup talking points
├── REPRODUCIBILITY.md                 # Checklist for judges
└── README.md                          # This file
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (recommended for reproducibility)
- API keys: `GEMINI_API_KEY` (and optionally `OPENAI_API_KEY` for fallback)

### Quick Start (Docker)

```bash
# Clone repo
git clone https://github.com/Olaiwonismail/OmniNaija.git
cd OmniNaija

# Copy env template
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

# Build and run
docker-compose up --build

# Open browser
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000
```

### Local Development Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Initialize ChromaDB collections
python backend/db/chromadb_manager.py --init

# Start backend server
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal: start frontend
streamlit run frontend/streamlit_app.py
```

### Data Ingestion (Day 1)

Use streaming from Hugging Face to build bundled product JSONL, then create the Chroma collection.

```bash
# Install data dependencies
pip install datasets chromadb sentence-transformers

# Stream + sample Amazon Reviews 2023 into bundled JSONL
python scripts/amazon_ingest.py --per-category 2000

# Build persistent Chroma collection
python scripts/build_chroma.py --persist-dir chroma_db

# Sanity-check retrieval results
python scripts/sanity_query.py --persist-dir chroma_db
```

---

## API Endpoints

### POST `/simulate`
Generate persona-driven reviews.

**Request:**
```json
{
  "persona_id": "tobi_yaba_dev",
  "product_id": "30000mah_power_bank",
  "product_metadata": {
    "name": "30000mAh USB-C Power Bank",
    "category": "Electronics",
    "price": 15000
  }
}
```

**Response:**
```json
{
  "rating": 5,
  "review": "Solid gadget fr fr. Keeps my MacBook alive when NEPA dey form...",
  "behavioral_fidelity": 0.94,
  "model": "gemini-2.0-flash"
}
```

### POST `/recommend` (Multi-turn)
Get intent-aware recommendations with optional cross-domain bridging.

**Request:**
```json
{
  "session_id": "tobi_session_001",
  "user_message": "My laptop keeps dying at client meetings. What can I do?",
  "persona_id": "tobi_yaba_dev",
  "bridge_threshold": 0.85
}
```

**Response:**
```json
{
  "assistant_message": "Tobi, you need something serious. I'd grab this 30000mAh...",
  "inferred_intent": {
    "primary": "remote_work_continuity",
    "confidence": 0.92,
    "reasoning": "Mentions laptop dying + client meetings → work reliability concern"
  },
  "amazon_recommendations": [
    {
      "product_id": "30k_pb_001",
      "name": "30000mAh USB-C Power Bank",
      "relevance": 0.96
    }
  ],
  "yelp_recommendations": [
    {
      "venue_id": "yaba_hub_cafe",
      "name": "Yaba Hub Cafe",
      "reason": "Same intent (remote work), infrastructure (backup gen, wifi)",
      "bridge_confidence": 0.92
    }
  ],
  "bridge_executed": true,
  "model": "gemini-2.0-flash"
}
```

---

## Personas (5 Detailed Examples)

Stored in `frontend/assets/personas.json`. Each includes:
- Demographics (age, location, profession, income)
- Historical purchase patterns + cart items
- Communication style (vocabulary, code-switching, idioms)
- Value priorities (quality, price, infrastructure workarounds)

**Examples:**
1. **Tobi** — Freelance dev in Yaba, code-switches to Pidgin, prioritizes uptime
2. **Folake** — Product manager in Ikoyi, fitness enthusiast, seeks community + gear
3. **Kingsley** — Student in Calabar, budget-conscious, DIY culture, studies electrical engineering
4. **Chioma** — Owambe event organizer, Lagos mainland, high-context buying (aesthetics + logistics)
5. **Ahmed** — Kano businessman, e-commerce seller, bulk-buy optimization focus

---

## Demo Mode & Fallback Strategy

**Risk:** LLM API failure during live pitch  
**Mitigation:** Demo Mode toggle in Streamlit UI

- Pre-computed cached responses for the core offline scenarios in `demo_cache/`
- `DEMO_MODE=true` short-circuits `/simulate` and `/recommend` to those cached request/response pairs
- Visually identical to live calls (same delay, same streaming animation)
- Activates via UI toggle or env var `DEMO_MODE=true`
- Multi-provider abstraction: switch Gemini → OpenAI → Anthropic via `LLM_PROVIDER` env var

Offline verification:

```bash
PYTHONPATH=. python scripts/test_demo_mode.py
```

Cached demo files currently include:
- `demo_cache/simulate_review_backup.json`
- `demo_cache/scenario1_recommend_remote_work.json`
- `demo_cache/scenario2_recommend_bridge_spot.json`
- `demo_cache/scenario3_recommend_owambe_bridge.json`

**Additional safety net:**
- Pre-recorded 2-minute demo video (unlisted, backed up locally)
- If WiFi fails entirely, play video + pitch live over it

---

## Running the Live Demo

```bash
# Terminal 1: Backend
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
streamlit run frontend/streamlit_app.py

# Open browser: http://localhost:8501
```

**Demo Script (90 seconds):**
1. **[0–20s] Problem** — Generic AI misses Nigerian context
2. **[20–50s] Live Demo** — Chat with Tobi persona, see power bank recommendation
3. **[50–70s] Wow Factor** — Show same intent bridging to Yaba Hub Cafe (Yelp)
4. **[70–90s] Negative Case** — Show agent declining weak bridge signal
5. **[90s] Impact** — Next billion users, generalizable architecture

See `demo_script.md` for detailed talking points and backup scenarios.

---

## Solution Paper & Pitch Deck

- **Solution Paper** (`solution_paper.md`) — 4–8 pages, structured around Intent Graph concept
- **Pitch Deck** (`pitch_deck.md`) — 10 slides, narrative flow for live presentation
- **Personas Deep Dive** — Embedded in paper Appendix with dialect samples, purchase history, values

---

## Reproducibility Checklist

See `REPRODUCIBILITY.md` for judges:
- ✅ One-command Docker setup
- ✅ Pre-populated ChromaDB collections (no external API calls for embeddings)
- ✅ Cached demo responses for offline operation
- ✅ `.env.example` with all required keys
- ✅ All datasets versioned in repo (Amazon subset + Yelp + augmented venues)
- ✅ GitHub actions / CI placeholder for future deployments

---

## Team Roles

| Member | Responsibility | Deliverables |
|--------|-----------------|--------------|
| **Member 1** | Agent Core (Backend/ML) | LangGraph state machine, multi-provider LLM client, API endpoints, prompt library |
| **Member 2** | Frontend & Polish (Streamlit) | Chat UI, intent panel, demo mode toggle, visual design, loading states |
| **Member 3** | Data & Infrastructure | ChromaDB ingestion, Yelp augmentation, embeddings pipeline, Docker, .env strategy |
| **Member 4** | Paper, Pitch, Personas | Solution paper (Intent Graph narrative), 10-slide deck, 5 detailed personas, README, demo script |

---

## Development Timeline

| Day | Date | Goal | Status |
|-----|------|------|--------|
| **1** | 20 May | Lock architecture, write personas, subset datasets, scaffold repo, multi-provider client built | TBD |
| **2** | 21 May | Agent graph end-to-end (rough), basic Streamlit chat, paper outline approved | TBD |
| **3** | 22 May | Cross-domain bridge functioning, intent panel visible, first draft paper, Docker working | TBD |
| **4** | 23 May | Demo video recorded, paper finalized, repo cleaned, dress rehearsal | TBD |
| **Submit** | 24 May | **Noon deadline** (not midnight) | — |

---

## Key Differentiators

1. **Intent Graph Concept** — No other team will have a name for cross-domain reasoning
2. **Negative Case Demonstration** — Proves agent wisdom, not just a lookup table
3. **Intent Panel UI** — Judges see the reasoning in real-time
4. **Persona Depth** — Not generic; 5 deeply researched Nigerian profiles with dialect + purchase history
5. **Multilingual NLG** — Code-switches to Pidgin where culturally appropriate
6. **One-Command Reproducibility** — Docker + cached demo = zero-friction judge experience

---

## Environment Variables

Create `.env` file (copy from `.env.example`):

```env
# LLM Configuration
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_key_here  # Optional fallback
LLM_PROVIDER=gemini  # or openai

# Backend
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
CORS_ORIGINS=http://localhost:8501,http://localhost:3000

# Frontend
STREAMLIT_SERVER_PORT=8501

# Demo Mode
DEMO_MODE=false  # Set to true for cached responses

# ChromaDB
CHROMADB_HOST=localhost
CHROMADB_PORT=8000
```

---

## Testing

```bash
# Unit tests (TBD)
pytest backend/tests/

# Integration: intent extraction
python backend/tests/test_intent_extraction.py

# Integration: review simulator
python backend/tests/test_review_simulator.py

# Integration: cross-domain bridge
python backend/tests/test_bridge_scoring.py
```

---

## Known Limitations & Mitigations

| Issue | Mitigation |
|-------|-----------|
| LLM API throttling during demo | Multi-provider fallback + demo cache |
| Thin Yelp Nigeria coverage | Hand-augmented 200+ venues from Google Maps |
| Cold start (first embedding query slow) | Pre-computed ChromaDB collections in repo |
| Network failure at venue | Pre-recorded demo video backup |

---

## Future Roadmap

- WhatsApp interface for mass adoption
- Jumia / Konga partnership integration
- Goodreads as third domain (book clubs, reading cafes)
- Multi-language support (Hausa, Yoruba, Igbo)
- Real-time sentiment analysis on user reviews
- Feedback loop: user ratings → persona refinement

---

## References & Resources

- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **ChromaDB Docs:** https://docs.trychroma.com/
- **Gemini API:** https://ai.google.dev/
- **FastAPI:** https://fastapi.tiangolo.com/
- **Streamlit:** https://docs.streamlit.io/

---

## License

DSN x BCT LLM Agent Challenge Hackathon Project (May 2026)

---

**Questions?** See `solution_paper.md` for architectural deep-dive and `demo_script.md` for live demo guidance.

**Submit by 24 May noon.** Good luck! 🚀
