import streamlit as st
from document_processor import process_and_store
from groq_chain import get_rag_chain

st.set_page_config(page_title="RAG Chatbot with Groq", layout="wide")
st.title("📚 AI Chatbot with Document Upload (Groq + RAG)")

# Upload document
uploaded_file = st.file_uploader("Upload a document (PDF/TXT/DOCX)", type=["pdf", "txt", "docx"])
if uploaded_file:
    with st.spinner("Processing document..."):
        process_and_store(uploaded_file)
    st.success("Document processed and indexed.")

# Chat interface
user_question = st.text_input("Ask a question from the document:")
if user_question:
    with st.spinner("Thinking..."):
        answer = get_rag_chain()(user_question)
    st.markdown("### 💬 Answer")
    st.write(answer)
