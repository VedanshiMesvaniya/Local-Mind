# localmind/run_ingest.py
import sys
import os

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.ingest import run_ingestion

if __name__ == "__main__":
    print(" Starting LocalMind Ingestion Pipeline...")
    print(" Scanning the /pdfs directory...")
    total_chunks = run_ingestion()
    print(f" Ingestion Complete! Indexed {total_chunks} chunks.")
    print(" You can now start the API with: uvicorn localmind.interfaces.api:app --reload")