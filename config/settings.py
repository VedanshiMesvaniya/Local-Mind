import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
PDF_DIR = BASE_DIR / "pdfs"
DB_DIR = BASE_DIR / "chroma_db_local"

# Models
MAIN_MODEL = os.getenv("MAIN_MODEL", "minimax-m3:cloud")
UTILITY_MODEL = "phi4-mini:latest"
EMBEDDING_MODEL = "nomic-embed-text"

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Mem0
USER_ID = "mihirmaru"