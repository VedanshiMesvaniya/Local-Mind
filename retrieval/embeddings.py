from langchain_ollama import OllamaEmbeddings
from config.settings import EMBEDDING_MODEL, OLLAMA_HOST

def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)