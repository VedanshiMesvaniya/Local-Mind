from langchain_chroma import Chroma
from .embeddings import get_embeddings
from config.settings import DB_DIR

def get_vectorstore():
    return Chroma(persist_directory=str(DB_DIR), embedding_function=get_embeddings())