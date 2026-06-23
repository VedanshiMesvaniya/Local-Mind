# retrieval/embeddings.py
from langchain_openai import OpenAIEmbeddings
from config.settings import EMBEDDING_MODEL, OLLAMA_HOST

def get_embeddings():
    """
    Uses Ollama's OpenAI-compatible endpoint to bypass LangChain's 
    buggy native OllamaEmbeddings payload and timeout issues.
    """
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,             # "qwen3-embedding:4b"
        base_url=f"{OLLAMA_HOST}/v1",      # Points to Ollama's /v1 endpoint
        openai_api_key="ollama",           # Required dummy key for Ollama
        check_embedding_ctx_length=False,  # Prevents context length errors
        chunk_size=5                       # CRITICAL: Prevents CPU timeouts
    )