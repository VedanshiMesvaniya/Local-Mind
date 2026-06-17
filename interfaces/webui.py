import streamlit as st
import httpx
import time

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="LocalMind | Enterprise AI", layout="wide", page_icon="🏢")

# --- 2. SESSION STATE INITIALIZATION (CRITICAL: MUST BE FIRST) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🏢 LOCAL-MIND")
    st.caption("Enterprise Intelligence Engine")
    st.divider()
    
    # Knowledge Base Manager
    st.subheader("📂 Knowledge Base")
    try:
        response = httpx.get("http://localhost:8000/documents", timeout=5.0)
        docs = response.json()
        
        if docs:
            for doc in docs:
                st.success(f"📄 **{doc['filename']}**", icon=None)
                st.caption(f"Status: {doc['status']} | Size: {doc['size_mb']} MB")
        else:
            st.warning("No documents found. Run `python run_ingest.py`.")
    except Exception:
        st.error("❌ Cannot connect to API. Is the backend running?")

    st.divider()
    
    # System Telemetry
    st.subheader("📊 System Telemetry")
    try:
        perf = httpx.get("http://localhost:8000/metrics", timeout=2.0).json()
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="CPU", value=f"{perf['cpu_percent']:.1f}%")
        with col2:
            st.metric(label="RAM", value=f"{perf['memory_percent']:.1f}%")
    except Exception:
        st.caption("Telemetry offline.")
        
    st.divider()
    
    # Clear Chat Button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 4. MAIN CHAT INTERFACE ---
st.title("💬 Enterprise Compliance Analyst")
st.caption("Powered by LocalMind RAG Engine")

# Render chat history with custom names and avatars
for message in st.session_state.messages:
    display_name = "Mihir" if message["role"] == "user" else "Local Mind"
    avatar = "👤" if message["role"] == "user" else "🏢"
    
    with st.chat_message(display_name, avatar=avatar):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("Mihir", avatar="👤"):
        st.markdown(prompt)

    # 2. Generate and stream assistant response
    with st.chat_message("Local Mind", avatar="🏢"):
        placeholder = st.empty()
        full_response = ""
        
        payload = {
            "prompt": prompt,
            "chat_history": st.session_state.messages[:-1],
            "user_id": "enterprise_user" 
        }
        
        start_time = time.time()
        ttft = None
        
        try:
            with httpx.stream(
                "POST", 
                "http://localhost:8000/query", 
                json=payload, 
                timeout=500
            ) as response:
                if response.status_code == 200:
                    for chunk in response.iter_text():
                        if chunk:
                            # Capture Time to First Token (TTFT)
                            if ttft is None:
                                ttft = time.time() - start_time
                                
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")
                    
                    # Final render without the blinking cursor
                    placeholder.markdown(full_response)
                    
                    # Display Telemetry Footer
                    total_time = time.time() - start_time
                    if ttft:
                        st.caption(f"⏱️ **Total:** {total_time:.2f}s | 🚀 **First Token:** {ttft:.2f}s | 🧠 **Model:** `qwen3:14b`")
                    else:
                        st.caption(f"⏱️ **Total:** {total_time:.2f}s (Empty response)")
                        
                else:
                    st.error(f"API Error: {response.status_code}")
                    
        except httpx.ReadTimeout:
            st.error("⏳ Request timed out. The local model is taking too long to respond.")
        except Exception as e:
            st.error(f" Connection error: {str(e)}")
    
    # 3. Add assistant message to state
    st.session_state.messages.append({"role": "assistant", "content": full_response})