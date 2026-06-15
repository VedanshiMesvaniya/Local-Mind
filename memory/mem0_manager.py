from mem0 import Memory
from config.settings import UTILITY_MODEL, EMBEDDING_MODEL, OLLAMA_HOST

mem0_config = {
    "llm": {"provider": "ollama", "config": {"model": UTILITY_MODEL, "temperature": 0}},
    "embedder": {"provider": "ollama", "config": {"model": EMBEDDING_MODEL}},
    "vector_store": {
        "provider": "qdrant",
        "config": {"collection_name": "mem0_ollama_v1", "path": "./mem0_qdrant_data", "embedding_model_dims": 768}
    }
}

# Initialize Mem0
m = Memory.from_config(mem0_config)

def save_memory_background(conversation_text: str, user_id: str):
    """Runs asynchronously via FastAPI BackgroundTasks."""
    try:
        m.add(conversation_text, user_id=user_id)
    except Exception as e:
        print(f"Memory save failed: {e}")

def search_memory(query: str, user_id: str):
    results = m.search(query, filters={"user_id": user_id})
    if isinstance(results, dict) and 'results' in results:
        return results['results']
    return results if isinstance(results, list) else []