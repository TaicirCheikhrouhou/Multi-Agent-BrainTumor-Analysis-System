# config.py - Ollama Configuration for CrewAI
from crewai import LLM
import os

# Completely disable cloud APIs
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)

# Local LLM configuration with Ollama
llm = LLM(
    model="ollama/mistral:latest",
    base_url="http://localhost:11434",
    temperature=0.1
)

print("✅ Ollama LLM configured: mistral:latest")