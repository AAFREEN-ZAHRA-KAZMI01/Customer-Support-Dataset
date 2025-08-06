import streamlit as st
from src.redis_memory import RedisMemory
from src.rag_chain import RAGPipeline
from src.qdrant_client import QdrantVectorStore
from src.embeddings import EmbeddingGenerator
import time
from dotenv import load_dotenv
import os
import uuid
from typing import Dict, Any

# # <style>
#     # .stChatMessage[data-testid="stChatMessage-user"] {
#         background: white;
#         color: black;
#         margin-left: auto;
#         border-radius: 16px 16px 4px 16px;
#         padding: 16px;
#     }

#     /* Assistant message bubble - black background, white text */
#     .stChatMessage[data-testid="stChatMessage-assistant"] {
#         background-color: white !important;
#         color: white !important;
#         margin-right: auto;
#         border-radius: 16px 16px 16px 4px;
#         padding: 16px;
#         border: 1px solid rgba(255,255,255,0.2);
#     }

#     /* Sidebar and layout - white background, black text */
#     .css-1d391kg, .css-1lcbmhc {
#         background-color: white !important;
#         color: black !important;
#     }

#     /* Entire background and main area */
#     body, .main, .block-container {
#         background-color: white;
#         color: black ;
#     }
#     .st-emotion-cache-9ajs8n{
#     background-color: black;
#         color: white ;
#     }
#     </style>

# # Load environment variables
load_dotenv()

# Minimal and clean CSS theme
def load_css():
    st.markdown("""
    
    """, unsafe_allow_html=True)

@st.cache_resource
def init_components(session_id: str) -> tuple[RAGPipeline, RedisMemory]:
    """Initialize and cache all components"""
    try:
        # Initialize with error handling
        vector_db = QdrantVectorStore()
        if not vector_db.collection_exists():
            vector_db.create_collection()
        
        return (
            RAGPipeline(),
            RedisMemory(session_id=session_id)
        )
    except Exception as e:
        st.error(f"System initialization failed: {str(e)}")
        st.stop()

def display_chat_history(memory: RedisMemory) -> None:
    """Display chat history in sidebar"""
    history = memory.get_history()
    if not history:
        st.caption("No conversations yet")
        return
    
    for i, msg in enumerate(history[-8:]):  # Show last 8 messages
        with st.expander(f"💬 Chat {i+1}", expanded=False):
            if msg['user']:
                st.markdown(f"**Q:** {msg['user'][:60]}...")
            if msg['assistant']:
                st.markdown(f"**A:** {msg['assistant'][:80]}...")

def display_main_chat(memory: RedisMemory) -> None:
    """Display main chat messages"""
    history = memory.get_history()
    for msg in history[-6:]:  # Show last 6 messages in main view
        if msg['user']:
            with st.chat_message("user"):
                st.write(msg['user'])
        if msg['assistant']:
            with st.chat_message("assistant"):
                st.write(msg['assistant'])

def generate_and_display_response(rag: RAGPipeline, prompt: str) -> Dict[str, Any]:
    """Generate response and display with smooth streaming effect"""
    message_placeholder = st.empty()
    full_response = ""
    
    # Generate response
    response = rag.generate_response(prompt)
    
    # Stream the response with typing effect
    words = response["answer"].split()
    for i, word in enumerate(words):
        full_response += word + " "
        time.sleep(0.05)  # Smooth typing speed
        
        # Add typing indicator
        if i < len(words) - 1:
            message_placeholder.markdown(full_response + "⏳")
        else:
            message_placeholder.markdown(full_response)
    
    return response

def main():
    # Page configuration
    st.set_page_config(
        page_title="E-Commerce Assistant",
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    load_css()
    
    # Session management
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Initialize components
    with st.spinner("🚀 Initializing your assistant..."):
        rag, memory = init_components(st.session_state.session_id)
    
    # Sidebar
    with st.sidebar:
        st.markdown("# 🛒 Shopping Assistant")
        st.markdown("---")
        
        # Session info
        st.subheader("📱 Session")
        st.code(memory.session_id[:8] + "...")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 New", use_container_width=True):
                st.session_state.session_id = str(uuid.uuid4())
                st.rerun()
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                memory.clear_history()
                st.rerun()
        
        # Chat history
        st.markdown("---")
        st.subheader("💭 History")
        display_chat_history(memory)
        
        # Stats
        st.markdown("---")
        st.subheader("📊 Stats")
        history_count = len(memory.get_history())
        st.metric("Messages", history_count)
    
    # Main chat interface
    st.title("E-Commerce Support")
    st.caption("Ask me anything about orders, payments, returns, or your account")
    
    # Display chat messages
    display_main_chat(memory)
    
    # Chat input and processing
    if prompt := st.chat_input("How can I help you today?"):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            try:
                response = generate_and_display_response(rag, prompt)
                
                # Show sources if available
                if response.get("sources"):
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(response["sources"], 1):
                            st.markdown(f"**Source {i}** | Category: `{source['metadata'].get('category', 'General')}` | Score: `{source['score']:.2f}`")
                            st.code(source['text'][:200] + "..." if len(source['text']) > 200 else source['text'])
                            if i < len(response["sources"]):
                                st.divider()
                
                # Save to memory
                memory.add_message(prompt, response["answer"])
                
            except Exception as e:
                st.error(f"⚠️ Something went wrong: {str(e)}")
                memory.add_message(prompt, f"Error: {str(e)}")

if __name__ == "__main__":
    main()