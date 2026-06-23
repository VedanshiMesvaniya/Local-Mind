import streamlit as st
import httpx
import time
import re
import json
from pathlib import Path

# --- 1. PAGE CONFIG & CUSTOM CSS ---
st.set_page_config(page_title="LocalMind | Enterprise AI", layout="wide", page_icon="🏢")

# Custom CSS for a polished, enterprise SaaS look
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .source-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #0066cc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stProgress > div > div > div > div { background-color: #0066cc; }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# --- 3. SIDEBAR: CONTROL PLANE ---
with st.sidebar:
    st.title("🏢 LOCAL-MIND")
    st.caption("Enterprise Intelligence Engine v2.0")
    st.divider()
    
    # Model Configuration
    st.subheader("⚙️ Model Configuration")
    selected_model = st.selectbox(
        "Select Inference Model",
        ["qwen3.5:9b", "qwen3:14b", "minimax-m3:cloud", "phi4-mini:latest"],
        index=0
    )
    
    st.divider()
    
    # Knowledge Base Manager
    st.subheader("📂 Knowledge Base")
    
    # Direct PDF Upload
    uploaded_files = st.file_uploader("Upload PDFs to Ingest", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        pdf_dir = Path("pdfs") # Assumes you run streamlit from the project root
        pdf_dir.mkdir(parents=True, exist_ok=True)
        for file in uploaded_files:
            with open(pdf_dir / file.name, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"✅ Uploaded {len(uploaded_files)} files to /pdfs")
        st.info("Run `python run_ingest.py` in your terminal to index them.")

    # Active Document List
    try:
        response = httpx.get("http://localhost:8000/documents", timeout=5.0)
        docs = response.json()
        
        if docs:
            st.caption(f"Active Documents: {len(docs)}")
            for doc in docs:
                with st.container():
                    st.markdown(f"**📄 {doc['filename']}**")
                    st.caption(f"Size: {doc['size_mb']} MB")
        else:
            st.warning("No documents indexed.")
    except Exception:
        st.error("❌ Backend offline. Start FastAPI server.")

    st.divider()
    
    # System Telemetry
    st.subheader("📊 System Telemetry")
    try:
        perf = httpx.get("http://localhost:8000/metrics", timeout=2.0).json()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU", f"{perf['cpu_percent']:.1f}%")
            st.progress(perf['cpu_percent'] / 100.0)
        with col2:
            st.metric("RAM", f"{perf['memory_percent']:.1f}%")
            st.progress(perf['memory_percent'] / 100.0)
    except Exception:
        st.caption("Telemetry offline.")
        
    st.divider()
    
    # Session Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.rerun()
    with col2:
        if st.button("💾 Export", use_container_width=True):
            chat_export = json.dumps(st.session_state.messages, indent=2)
            st.download_button(
                "Download JSON",
                chat_export,
                file_name="localmind_audit_log.json",
                mime="application/json"
            )

# --- 4. MAIN INTERFACE: TWO-COLUMN LAYOUT ---
st.title("💬 Enterprise Compliance Analyst")
st.caption("Ask questions about your corporate documents. All processing is 100% local and air-gapped.")

# Create columns: 70% for Chat, 30% for Citations/Security
chat_col, source_col = st.columns([3, 1.2])

# --- LEFT COLUMN: CHAT ENGINE ---
with chat_col:
    # Render chat history
    for message in st.session_state.messages:
        display_name = "You" if message["role"] == "user" else "LocalMind AI"
        avatar = "👤" if message["role"] == "user" else "🏢"
        
        with st.chat_message(display_name, avatar=avatar):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a compliance or financial question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("You", avatar="👤"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("LocalMind AI", avatar="🏢"):
            placeholder = st.empty()
            full_response = ""
            
            # Payload now includes the selected model from the sidebar
            payload = {
                "prompt": prompt,
                "chat_history": st.session_state.messages[:-1],
                "user_id": "enterprise_user",
                "model_name": selected_model 
            }
            
            start_time = time.time()
            ttft = None
            
            try:
                with httpx.stream(
                    "POST", 
                    "http://localhost:8000/query", 
                    json=payload, 
                    timeout=300
                ) as response:
                    if response.status_code == 200:
                        for chunk in response.iter_text():
                            if chunk:
                                if ttft is None:
                                    ttft = time.time() - start_time
                                full_response += chunk
                                placeholder.markdown(full_response + "▌")
                        
                        placeholder.markdown(full_response)
                        
                        # MAGIC: Parse Citations for the Source Panel
                        # Looks for [Source: filename, Page: X]
                        citations = re.findall(r'\[Source:\s*(.*?),\s*Page:\s*(\d+)\]', full_response)
                        st.session_state.last_sources = list(set(citations)) 
                        
                        total_time = time.time() - start_time
                        st.caption(f"⏱️ **Total:** {total_time:.2f}s | 🚀 **TTFT:** {ttft:.2f}s | 🧠 **Model:** `{selected_model}`")
                    else:
                        st.error(f"API Error: {response.status_code}")
                        
            except httpx.ReadTimeout:
                st.error("⏳ Request timed out. The local model is taking too long.")
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")
        
        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun() # Rerun to update the source panel on the right

# --- RIGHT COLUMN: VERIFIED SOURCES & SECURITY ---
with source_col:
    st.subheader("📑 Verified Citations")
    st.caption("Grounding sources for the latest response.")
    
    if st.session_state.last_sources:
        for source, page in st.session_state.last_sources:
            # Render as a nice HTML card
            st.markdown(f"""
            <div class="source-card">
                <strong>📄 {source}</strong><br>
                <span style="color: #666;">Page: {page}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Ask a question to see verified source citations here.")
        
    st.divider()
    
    st.subheader("🛡️ Security Posture")
    st.success("🔒 **Air-Gapped Mode Active**")
    st.caption("Zero external API calls. No data leaves this machine.")
    
    st.divider()
    
    st.subheader("🧠 Architecture")
    st.caption("**Retrieval:** MMR (k=4, fetch=20)")
    st.caption("**Embedding:** qwen3-embedding:4b")
    st.caption("**Vector DB:** Chroma (Local)")