import time
from .parser import hybrid_pdf_parser
from .chunker import get_text_splitter
from retrieval.vectorstore import get_vectorstore
from config.settings import PDF_DIR

def run_ingestion():
    print("Starting Batched Ingestion...")
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files: raise ValueError("No PDFs found.")
        
    vectorstore = get_vectorstore()
    text_splitter = get_text_splitter()
    total_chunks = 0
    
    for file_path in pdf_files:
        try:
            docs = hybrid_pdf_parser(file_path)
            processed_docs = []
            for doc in docs:
                if doc.metadata["type"] == "text":
                    processed_docs.extend(text_splitter.split_documents([doc]))
                else:
                    processed_docs.append(doc) 
                    
            vectorstore.add_documents(processed_docs)
            total_chunks += len(processed_docs)
            print(f"Indexed {file_path.name}")
        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")
            
    print(f"Ingestion Complete! Total chunks: {total_chunks}")
    return total_chunks