from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from .prompt import REWRITE_PROMPT
from config.settings import UTILITY_MODEL, OLLAMA_HOST

def get_rewrite_chain():
    rewrite_llm = ChatOllama(model=UTILITY_MODEL, temperature=0, base_url=OLLAMA_HOST)
    return REWRITE_PROMPT | rewrite_llm | StrOutputParser()