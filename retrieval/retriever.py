from .vectorstore import get_vectorstore

def get_retriever():
    vectorstore = get_vectorstore()
    # MMR ensures diversity in retrieved chunks
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 15})