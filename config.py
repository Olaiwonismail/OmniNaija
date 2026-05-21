import os

from dotenv import load_dotenv


load_dotenv()


class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemma4")
    DEMO_MODE = os.getenv("DEMO_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}