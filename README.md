# OmniNaija Lite
## Intent-Based Cross-Domain Recommendation Engine for Emerging Markets

> Most AI recommenders ship a single-domain lookup table. We ship **intent reasoning** that dynamically bridges domains (Amazon products ↔ Yelp venues) using a framework we call the **Intent Graph**. This unique architecture explicitly unlocks the advanced cross-domain and behavioral reasoning requirements for the DSN × BCT LLM Agent Challenge.

![Build Status](https://img.shields.io/badge/Status-Hackathon_Submission-brightgreen)
![Tech Stack](https://img.shields.io/badge/Stack-LangGraph%20%7C%20FastAPI%20%7C%20ChromaDB%20%7C%20Streamlit-blue)

---

## 🌍 The Problem
Generic AI recommenders treat a Lagos software engineer the same as a Brooklyn one. They see a power bank purchase as a static transaction. They miss the **grid context**, the **cafe-with-backup-power culture**, and the way Nigerians route around infrastructure gaps. Traditional recommendation engines operating in emerging markets are flying blind to the "why" behind a purchase.

## 💡 The Solution
OmniNaija Lite is an LLM agent that reasons about **why** someone is buying, not just **what**. 
- **Understands persona** — Encodes Nigerian user demographics, vocabulary, value priorities, and cultural context.
- **Extracts intent** — Analyzes purchase history and conversational signals to infer deep user state (e.g., remote work continuity, fitness prep, owambe planning).
- **Bridges intelligently** — Maps Amazon product recommendations dynamically to Yelp service/venue recommendations using the same inferred intent.
- **Exhibits restraint** — Refuses to bridge domains when the intent signal confidence is low, proving it is a reasoning engine and not a hardcoded lookup table.

---

## 🚀 Core Capabilities

### 1. Persona-Driven Simulation
Generates high-fidelity product reviews anchored to specific, researched Nigerian personas. The agent adopts appropriate dialects (Nigerian English / Pidgin) and biases its ratings based on the persona's historical purchasing behavior.

### 2. Intent-Aware Conversational Recommender
A multi-turn chat interface that remembers context and shapes its Amazon vector store queries based on the active user profile. 
**Innovation: Recency-Anchored Retrieval**
Instead of diluting vector search queries with full conversational history, the agent dynamically anchors semantic searches to the user's most recent 1-2 product titles when available. This dramatically improves exact-match relevance for "what should I buy next" queries while leaving pure intent-driven queries unaffected.

### 3. Cross-Domain Bridge (The "Wow" Factor)
When the LLM's confidence in an inferred intent exceeds a strict threshold, the agent executes a cross-domain bridge.
**Example:**
*Intent: Remote Work Continuity*
- **Amazon:** Recommends a 30,000mAh Power Bank
- **Yelp:** Simultaneously recommends "Yaba Hub Cafe" (inferred need for backup generator + Wi-Fi)
- **Result:** A cohesive lifestyle recommendation, not just a product push.

---

## 📊 Quantitative Results

Our architecture was rigorously evaluated offline using a custom test suite against a held-out dataset of interactions. 

| Metric | Score | Significance |
|--------|-------|--------------|
| **Task A: BERTScore F1** | `0.7490` | Demonstrates high semantic fidelity in generated persona reviews. |
| **Task B: Category-Match@10** | `58.0%` | Agent correctly identifies the exact domain of the user's next purchase out of 6,000 items. |
| **Task B: Cross-Domain Accuracy** | `90.0%` | Agent correctly aligns Yelp venues with the user's latent Amazon intent 9 out of 10 times. |
| **Task B: Bridge Precision** | `100.0%` | When the agent decides to bridge domains, the resulting venue is always topically relevant. |
| **Task B: Bridge Restraint** | `66.7%` | Crucially, the agent successfully identifies weak signals and *refuses* to bridge, avoiding hallucinated connections. |

---

## 🛠 Tech Stack & Architecture

- **Orchestration:** LangGraph (State-machine flow for predictable, debuggable reasoning)
- **Primary LLM:** Gemini (via Google API - Fast, multilingual, excellent Pidgin comprehension)
- **Fallback LLM:** `llama-3.3-70b-versatile` (via Groq API for high-speed resilience)
- **Vector Database:** ChromaDB (Local, embedded, zero-config)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (Local semantic search)
- **Evaluation Model:** `distilbert-base-uncased` (Used for offline BERTScore metric calculation)
- **Backend/Frontend:** FastAPI & Streamlit
- **Datasets:** Subset of Amazon Reviews (2023) + Yelp Nigeria Data (augmented with ~200 specific Lagos/Abuja venues)

---

## 🏃‍♂️ Quick Start & Reproducibility

We prioritized a zero-friction experience for judges. The repository includes pre-computed ChromaDB collections and a Dockerized environment.

### Prerequisites
- Docker & Docker Compose
- `GEMINI_API_KEY`

### 1-Click Launch
```bash
# Clone repo
git clone https://github.com/Olaiwonismail/OmniNaija.git
cd OmniNaija

# Configure environment
cp .env.example .env
# --> Add your GEMINI_API_KEY to .env

# Build and start services
docker-compose up --build
```
- **Frontend UI:** [http://localhost:8501](http://localhost:8501)
- **Backend API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Testing & Evaluation

### Reproducing the Quantitative Results
Judges can reproduce the exact metrics from the table above by running the offline evaluation suite. The evaluation script bypasses the HTTP API and runs entirely in-process, meaning you do not need to start the backend server first.

**1. Full Ranking, Cross-Domain & Text Quality Evaluation (Task B + Task A Text)**
Computes Category-Match, Hit Rate, ROUGE, BERTScore, and Cross-Domain Accuracy on the full 50-case test set:
```bash
python evaluate_v2.py
```

**2. Rating Accuracy / RMSE Evaluation (Task A Ratings)**
Computes the Root Mean Squared Error for simulated review ratings:
```bash
python evaluate_v2.py --review-sim
```

*(Note: To run a faster 10-case verification instead of the full 50 cases, append the `--quick` flag to either command).*

### Graph Node Tests
Exercise the LangGraph intent→retrieve→compose flow locally without the frontend:
```bash
PYTHONPATH=. python scripts/run_node_tests.py
```

### Demo Mode (Fallback Strategy)
To ensure the live demo survives network issues or API rate limits, the UI includes a `DEMO_MODE` toggle. When enabled, the app short-circuits to pre-computed, cached LLM responses from `demo_cache/` that are visually identical to live API calls.

---

## 👥 Personas
The system includes 5 deeply researched Nigerian profiles (stored in `frontend/assets/personas.json`), ensuring recommendations are highly contextualized:
1. **Tobi:** Freelance dev in Yaba; code-switches to Pidgin; prioritizes uptime and infrastructure.
2. **Folake:** Product manager in Ikoyi; fitness enthusiast; seeks community.
3. **Kingsley:** Student in Calabar; budget-conscious DIYer.
4. **Chioma:** Owambe event organizer; high-context buyer focused on aesthetics and logistics.
5. **Ahmed:** Kano e-commerce seller; focuses on bulk-buy optimization.

---

## 📝 Project Details
**Author:** Olaiwon Ismail (Solo Developer)  
**Event:** DSN × BCT LLM Agent Challenge Hackathon  
**Documentation:** See `solution_paper.md` for an architectural deep-dive into the Intent Graph.
