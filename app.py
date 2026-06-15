import streamlit as st
import time
import threading
import os
import ollama

# Import the new conversational components from rag_core
from rag_core import (
    retriever, rewrite_chain, m, USER_ID, MAIN_MODEL,
    get_current_performance_metrics, format_docs, get_rag_chain 
)

# --- BACKGROUND MEMORY SAVER ---
def save_memory_in_background(text, user_id):
    """Saves memory silently without blocking the Streamlit UI."""
    try:
        m.add(text, user_id=user_id)
    except Exception:
        pass

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Enterprise RAG HUD", layout="wide", page_icon="🏢")

# --- SIDEBAR: THE "HUD" ---
with st.sidebar:
    st.title(" LOCAL-MIND Control Center")
    st.divider()
    
    st.subheader(" MAIN GENERATION MODEL")
    st.caption("This model writes the final answer. Pick your heaviest model.")
    
    # Fetch local models dynamically from Ollama
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        client = ollama.Client(host=ollama_host)
        models_response = client.list()
        available_models = [m['model'] for m in models_response.get('models', [])]
    except Exception as e:
        st.error(f"Ollama connection failed: {e}")
        available_models = [MAIN_MODEL]

    if not available_models:
        st.warning("No local models found! Run `ollama pull <model>` in your terminal.")
        available_models = [MAIN_MODEL]

    # Default the dropdown to the MAIN_MODEL
    default_index = available_models.index(MAIN_MODEL) if MAIN_MODEL in available_models else 0
    
    selected_model = st.selectbox(
        "Select Main Model", 
        available_models, 
        index=default_index,
        key="main_gen_model"
    )
    
    st.divider()
    
    st.subheader(" System Telemetry")
    perf = get_current_performance_metrics()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="CPU Usage", value=f"{perf['cpu_percent']:.1f}%")
    with col2:
        st.metric(label="RAM Usage", value=f"{perf['memory_percent']:.1f}%")
        
    st.progress(perf['memory_percent'] / 100)
    st.caption(f"Used: {perf['memory_used_gb']:.2f} GB | Swap: {perf['swap_percent']:.1f}%")
    
    st.divider()
    
    show_debug = st.checkbox(" Show Retrieved Chunks", value=True, key="debug_chunks_checkbox")
    
    if st.button("🧹 Clear Chat History", key="clear_chat_btn"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title(" Enterprise Compliance Analyst")
st.caption("Powered by Streaming RAG, Local Rewriting & Mem0 Memory")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"] and show_debug:
            with st.expander(" View Retrieved Sources"):
                for i, src in enumerate(message["sources"]):
                    st.markdown(f"**[Rank {i+1}]** {src['source']} (Page {src['page']})")
                    st.code(src['preview'], language="markdown")

if prompt := st.chat_input("Ask a compliance or financial question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(" Thinking & Retrieving..."):
            start_time = time.time()
            
            try:
                # A. FORMAT HISTORY
                history_messages = st.session_state.messages[:-1]
                chat_history_str = "\n".join([
                    f"{msg['role'].capitalize()}: {msg['content']}" 
                    for msg in history_messages[-6:]
                ]) if history_messages else "No previous conversation."
                
                # B. REWRITE VAGUE QUESTIONS (Runs locally & instantly)
                if history_messages:
                    standalone_question = rewrite_chain.invoke({
                        "chat_history": chat_history_str,
                        "question": prompt
                    }).strip()
                else:
                    standalone_question = prompt

                # C. SEARCH MEMORY
                search_results = m.search(prompt, filters={"user_id": USER_ID})
                if isinstance(search_results, dict) and 'results' in search_results:
                    relevant_memories = search_results['results']
                elif isinstance(search_results, list):
                    relevant_memories = search_results
                else:
                    relevant_memories = []
                    
                memory_context = "\n".join([mem.get('memory', str(mem)) for mem in relevant_memories]) if relevant_memories else "No relevant past memories found."

                # D. RETRIEVE DOCUMENTS
                docs = retriever.invoke(standalone_question)
                context = format_docs(docs)

                # E. BUILD DYNAMIC CHAIN BASED ON UI DROPDOWN
                dynamic_rag_chain = get_rag_chain(selected_model)

                # F. GENERATE RESPONSE (STREAMING)
                response_stream = dynamic_rag_chain.stream({
                    "question": prompt,
                    "context": context,
                    "memory": memory_context,
                    "chat_history": chat_history_str
                })
                
                # Stream directly to the UI
                full_response = st.write_stream(response_stream)
                
                latency = time.time() - start_time
                
                # G. SAVE TO MEMORY IN BACKGROUND (Fixes the UI freeze)
                conversation_text = f"User: {prompt}\nAssistant: {full_response}"
                threading.Thread(
                    target=save_memory_in_background, 
                    args=(conversation_text, USER_ID), 
                    daemon=True
                ).start()

                # H. FETCH SOURCES FOR DEBUG
                sources = []
                if show_debug:
                    for doc in docs:
                        sources.append({
                            "source": doc.metadata.get("source", "Unknown"),
                            "page": doc.metadata.get("page", "N/A"),
                            "preview": doc.page_content[:200] + "..."
                        })

                st.caption(f"⏱️ Latency: {latency:.2f}s | 🔍 Search: '{standalone_question}' | 🧠 Model: `{selected_model}`")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response, 
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f" Pipeline Error: {e}")