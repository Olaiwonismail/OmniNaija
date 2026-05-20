<!-- # OmniNaija
AI agent simulating Nigerian user behavior for cross-domain lifestyle recommendations. -->

🧠 OmniNaija (The Cross-Domain Lifestyle Agent)
A Context-Aware LLM Agent for the African Consumer

📖 About the Project
OmniNaija is a state-of-the-art LLM agent built for the DSN X BCT LLM Agent Challenge. Traditional AI recommendation systems often treat users as static profiles rather than dynamic, context-sensitive agents. They fail to capture the rich cultural context, behavioral nuances, and environmental realities of the Nigerian consumer.

OmniNaija solves this by utilizing hyper-specific Nigerian personas to drive a cross-domain reasoning engine. It doesn't just recommend products; it understands why a user needs them. By bridging e-commerce data (Amazon) with real-world lifestyle data (Yelp), OmniNaija anticipates what people behave like, what they want, and what they'll choose next.

🎯 Hackathon Objectives Addressed
This project directly tackles the two core tasks of the competition:

Task A (User Modeling): We built an agent that deeply understands users enough to simulate their reviews. It simulates star ratings and written reviews for unseen items while capturing exact tone, rating behaviour, and contextual nuance.

Task B (Recommendation): We built an agentic workflow that reasons before recommending. It delivers personalized recommendations by handling complex multi-turn, cold-start, and cross-domain scenarios (e.g., dynamically transitioning from an Amazon tech purchase to a Yelp workspace recommendation based on user context).

⚙️ Core Features
/simulate-review Endpoint: Takes a user persona and product details as input, and generates culturally accurate reviews and ratings as output.

/recommend Endpoint: Takes a user persona as input and engages in multi-turn conversational retrieval to produce personalized recommendations as output.

Containerized Delivery: The entire API and vector database are cleanly packaged using Docker for easy execution and code reproducibility.

🛠️ Technology Stack
Backend: Python / FastAPI

AI/Agent Orchestration: LangChain / OpenAI

Vector Database: ChromaDB (for fast semantic retrieval of subset Amazon & Yelp data)

Infrastructure: Docker & Docker Compose

🚀 Getting Started (Local Setup)
(Edit this section with your specific environment variables and run commands once the backend is finalized)

Clone the repository:

Bash
git clone https://github.com/Olaiwonismail/OmniNaija.git
cd OmniNaija
Environment Variables:
Create a .env file in the root directory and add your required LLM API keys:

Plaintext
OPENAI_API_KEY=your_key_here
Run the Container:

Bash
docker-compose up --build
Test the Endpoints:
Navigate to http://localhost:8000/docs to interact with the FastAPI Swagger UI and test the /simulate-review and /recommend endpoints.

📂 Deliverables Checklist
[ ] Clean, documented, reproducible codebase.

[ ] Containerized API endpoint.

[ ] 4-8 Page Solution Paper covering architecture decisions and ablation studies.
