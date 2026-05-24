# OmniNaija Lite 🇳🇬
### Intent-Based Cross-Domain Recommender for Emerging Markets

> Most recommenders use static lookup tables. OmniNaija ships **intent reasoning** that dynamically bridges domains (Amazon products ↔ Yelp venues) using an **Intent Graph** to understand *why* users buy, not just *what*. Built for the **DSN × BCT LLM Agent Challenge**.

![Status](https://img.shields.io/badge/Status-Hackathon_Submission-brightgreen)
![Stack](https://img.shields.io/badge/Stack-LangGraph%20%7C%20FastAPI%20%7C%20ChromaDB%20%7C%20Streamlit-blue)

---

## 💡 The Core Innovation
Traditional engines are blind to emerging market context (e.g., power outages). OmniNaija solves this:
- **Nigerian Personas:** Custom profiles modeling local context (Pidgin vocabulary, grid reliance, Owambe logistics).
- **Recency-Anchored Retrieval:** Anchors product vector search to the user's latest 1-2 purchases to ensure highly relevant recommendations.
- **Dynamic Cross-Domain Bridge:** Connects Amazon products to Yelp venues (e.g., recommends a backup power bank and a Yaba coworking cafe with generators).
- **Strict Restraint:** Refuses to bridge domains when intent confidence is low (66.7% restraint).

---

## 📊 Quantitative Results
Offline evaluation against held-out datasets yields the following verified metrics:

| Metric | Score | Significance |
|--------|-------|--------------|
| **Task A: BERTScore F1** | `0.7490` | High semantic fidelity in persona review simulation. |
| **Task B: Category-Match@10** | `58.0%` | Accurately predicts purchase category out of 6k options. |
| **Task B: Cross-Domain Accuracy** | `90.0%` | Correctly maps Yelp venues to Amazon intent. |
| **Task B: Bridge Precision** | `100.0%` | Zero irrelevant venue bridge hallucinations. |
| **Task B: Bridge Restraint** | `66.7%` | High confidence refusal when signal is weak. |

---

## 🛠 Tech Stack & Models
*   **Orchestration:** LangGraph (Reasoning Flow)
*   **Primary LLM:** Gemini (Google API)
*   **Fallback LLM:** `llama-3.3-70b-versatile` (Groq API)
*   **Vector DB:** ChromaDB (Local, Persistent)
*   **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (Local)
*   **Evaluation:** `distilbert-base-uncased` (Offline Metrics)

---

## 🏃‍♂️ Zero-Friction Setup

Initialize databases, stream processed subsets, and launch the system in a clean environment:

```bash
# 1. Clone & enter repository
git clone https://github.com/Olaiwonismail/OmniNaija.git && cd OmniNaija

# 2. Add API keys
cp .env.example .env  # Add GEMINI_API_KEY to .env

# 3. Bootstrap data & build ChromaDB (takes < 1 min)
pip install -r requirements.txt
python scripts/bootstrap.py

# 4. Launch UI & API in Docker
docker-compose up --build
```
*   **Frontend App:** [http://localhost:8501](http://localhost:8501)
*   **FastAPI API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Testing & Offline Evaluation

Run the evaluation suite in-process (no server startup required):

```bash
# Task B + Task A Text (Hits, NDCG, ROUGE, BERTScore)
python evaluation/evaluate_v2.py

# Task A Ratings (RMSE Evaluation)
python evaluation/evaluate_v2.py --review-sim

# Quick 10-Case Verification
python evaluation/evaluate_v2.py --quick
```

---

## 📝 Details
*   **Author:** Olaiwon Ismail (Solo Developer)
*   **Event:** DSN × BCT LLM Agent Challenge Hackathon
*   **Deep Dive:** See `solution_paper.md` for architectural design and intent mappings.
