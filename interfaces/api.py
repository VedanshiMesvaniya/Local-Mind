import random
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from retrieval.retriever import get_retriever
from llm.query_rewriter import get_rewrite_chain
from llm.generator import get_rag_chain
from memory.mem0_manager import save_memory_background, search_memory
from observability.metrics import get_current_performance_metrics

app = FastAPI(title="LocalMind RAG API")

class QueryRequest(BaseModel):
    prompt: str
    chat_history: List[dict] = []
    user_id: str = "default_user"
    model_name: Optional[str] = None

def format_docs(docs):
    return "\n\n".join([f"<doc src='{d.metadata.get('source')}' p='{d.metadata.get('page')}'>{d.page_content}</doc>" for d in docs])

@app.post("/query")
async def stream_query(request: QueryRequest, background_tasks: BackgroundTasks):
    # === PHASE 1: PREPARE HISTORY ===
    history_str = "\n".join([
        f"{m['role']}: {m['content']}" 
        for m in request.chat_history[-6:]
    ]) if request.chat_history else "None"

    # === PHASE 2: STRICT BYPASS LOGIC ===
    # We ONLY rewrite if:
    # 1. It is a follow-up question (chat history exists)
    # 2. AND we hit the 10% random chance
    # For standalone questions, we ALWAYS bypass.
    is_follow_up = bool(request.chat_history)
    should_rewrite = is_follow_up and (random.random() < 0.10)

    if should_rewrite:
        # 10% of follow-ups: Use the utility model to resolve pronouns/context
        rewrite_chain = get_rewrite_chain()
        standalone_q = await rewrite_chain.ainvoke({
            "chat_history": history_str,
            "question": request.prompt
        })
    else:
        # 100% of standalone questions + 90% of follow-ups: 
        # Bypass the utility model entirely. Pass raw prompt to retriever.
        standalone_q = request.prompt 

    # === PHASE 3: RETRIEVAL ===
    retriever = get_retriever()
    docs = await retriever.ainvoke(standalone_q)
    context = format_docs(docs)

    # === PHASE 4: MEMORY SEARCH (DISABLED) ===
    memory_context = "No long-term memories active."

    # === PHASE 5: STREAMING GENERATION ===
    chain = get_rag_chain(request.model_name)

    async def generate_stream():
        full_response = ""
        async for token in chain.astream({
            "question": request.prompt,
            "context": context,
            "memory": memory_context,
            "chat_history": history_str
        }):
            full_response += token
            yield token

        # === PHASE 6: BACKGROUND MEMORY SAVE (DISABLED) ===
        background_tasks.add_task(
            save_memory_background,
            f"User: {request.prompt}\nAI: {full_response}",
            request.user_id
        )

    return StreamingResponse(generate_stream(), media_type="text/plain")

@app.get("/metrics")
async def system_metrics():
    return get_current_performance_metrics()

# Add this to interfaces/api.py

from config.settings import PDF_DIR
from pathlib import Path

@app.get("/documents")
async def list_documents():
    """Returns a list of all PDFs currently available in the knowledge base."""
    files = []
    if PDF_DIR.exists():
        for f in PDF_DIR.glob("*.pdf"):
            files.append({
                "filename": f.name, 
                "status": "Ingested",
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2)
            })
    return files