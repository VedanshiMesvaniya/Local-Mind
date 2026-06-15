import streamlit as st
import httpx

# 1. PAGE CONFIG
st.set_page_config(page_title="LocalMind HUD", layout="wide", page_icon="🏢")

# 2. INITIALIZE SESSION STATE (THIS FIXES YOUR CRASH)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. SIDEBAR: KNOWLEDGE BASE & TELEMETRY
with st.sidebar:
    st.title(" LOCAL-MIND Control Center")
    st.divider()
    
    # Knowledge Base Manager
    st.subheader(" Knowledge Base")
    try:
        response = httpx.get("http://localhost:8000/documents", timeout=5.0)
        docs = response.json()
        
        if docs:
            for doc in docs:
                st.success(f" **{doc['filename']}**", icon=None)
                st.caption(f"Status: {doc['status']} | Size: {doc['size_mb']} MB")
        else:
            st.warning("No documents found. Run `python run_ingest.py` to load PDFs.")
    except Exception as e:
        st.error(" Cannot connect to API. Is the backend running?")

    st.divider()
    
    # Telemetry
    st.subheader(" System Telemetry")
    try:
        perf = httpx.get("http://localhost:8000/metrics", timeout=2.0).json()
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="CPU Usage", value=f"{perf['cpu_percent']:.1f}%")
        with col2:
            st.metric(label="RAM Usage", value=f"{perf['memory_percent']:.1f}%")
    except Exception:
        st.caption("Telemetry offline.")

# 4. MAIN CHAT INTERFACE
st.title("💬 Enterprise Compliance Analyst")
st.caption("Powered by Streaming RAG, Local Rewriting & Mem0 Memory")

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new chat input
if prompt := st.chat_input("Ask a compliance question..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        payload = {
            "prompt": prompt,
            "chat_history": st.session_state.messages[:-1],
            "user_id": "mihirmaru"
        }
        
        try:
            # Stream from the FastAPI backend
            with httpx.stream(
                "POST", 
                "http://localhost:8000/query", 
                json=payload, 
                timeout=120.0  # Long timeout for heavy generation
            ) as response:
                if response.status_code == 200:
                    for chunk in response.iter_text():
                        if chunk:
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")
                    
                    # Final render without the blinking cursor
                    placeholder.markdown(full_response)
                else:
                    st.error(f"API Error: {response.status_code}")
                    
        except httpx.ReadTimeout:
            st.error("⏳ Request timed out. The local model is taking too long to respond.")
        except Exception as e:
            st.error(f" Connection error: {str(e)}")
    
    # Add assistant message to state
    st.session_state.messages.append({"role": "assistant", "content": full_response})