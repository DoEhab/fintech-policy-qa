# app.py
import streamlit as st
from ask_question import ask_rag_question

# Page configuration
st.set_page_config(page_title="Fintech Policy AI", page_icon="🏦", layout="wide")

st.title("🏦 Fintech Policy Q&A Assistant")
st.markdown("Ask questions about PCI DSS, open banking, financial data security, and compliance policies.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            st.markdown(f"**📚 Sources:** {', '.join(message['sources'])}")

# Chat input
if prompt := st.chat_input("Ask a question about financial policies..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents and generating answer..."):
            result = ask_rag_question(prompt)
            
            st.markdown(result["answer"])
            if result["sources"]:
                st.markdown(f"**📚 Sources:** {', '.join(result['sources'])}")
    
    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": result["answer"],
        "sources": result["sources"]
    })

# Sidebar
with st.sidebar:
    st.header("About this App")
    st.markdown("""
    This AI assistant is powered by:
    - **Embeddings**: Local Nomic/BGE models (No API limits!)
    - **Vector DB**: Qdrant
    - **LLM**: Cohere Command
    
    Upload your own policies to customize the knowledge base.
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()