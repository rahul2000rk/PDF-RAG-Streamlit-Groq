import streamlit as st
import PyPDF2
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os

# --- Configuration ---
# Set the title of the Streamlit application
st.set_page_config(page_title="PDF RAG App with Groq", layout="wide")

# --- UI Elements ---
st.title("📄 PDF RAG App with Groq")
st.markdown(
    """
    Upload a PDF, and then ask questions about its content!
    This app uses Retrieval Augmented Generation (RAG) with Groq's LLMs.
    """
)

# Get Groq API Key from user input or environment variables
# Hardcoded Groq API Key - Replace "YOUR_GROQ_API_KEY_HERE" with your actual key
groq_api_key = "YOUR_GROQ_API_KEY_HERE" # It is recommended to use environment variables or Streamlit secrets for production apps.

# Initialize Groq LLM if API key is provided
llm = None
if groq_api_key:
    try:
        llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192") # You can change the model here
    except Exception as e:
        st.error(f"Error initializing Groq LLM: {e}. Please check your API key.")
else:
    st.warning("Please enter your Groq API Key to proceed.")

# --- Session State Management ---
# Initialize session state variables if they don't exist
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# --- PDF Processing Function ---
def process_pdf(pdf_file):
    """
    Processes the uploaded PDF: extracts text, splits it into chunks,
    creates embeddings, and stores them in a FAISS vector store.
    """
    st.info("Processing PDF... This might take a moment.")
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split the text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings and build the FAISS vector store
        # Using a local HuggingFace embedding model (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
        # This model will be downloaded on first run if not present.
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
        st.session_state.pdf_processed = True
        st.success("PDF processed successfully! You can now ask questions.")
        st.session_state.chat_history = [] # Clear chat history for new PDF
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        st.session_state.pdf_processed = False

# --- PDF Upload Section ---
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None and not st.session_state.pdf_processed:
    # Only process if a new file is uploaded or if the previous processing failed
    process_pdf(uploaded_file)
elif uploaded_file is None and st.session_state.pdf_processed:
    st.info("PDF already processed. Upload a new PDF to analyze a different document.")

# --- Chat Interface ---
if st.session_state.pdf_processed and llm:
    # Define the prompt template for the RAG chain
    # CORRECTED: Added {context} to the system prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use the following retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n\n{context}"),
        ("human", "{input}"),
    ])

    # Create the document combining chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retrieval chain
    retriever = st.session_state.vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Display chat messages from history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if user_query := st.chat_input("Ask a question about the PDF:"):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Invoke the RAG chain
                    response = retrieval_chain.invoke({"input": user_query})
                    assistant_response = response["answer"]
                    st.markdown(assistant_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, I encountered an error while trying to answer."})
elif not groq_api_key:
    st.info("Please enter your Groq API Key to enable the chat functionality.")
elif not st.session_state.pdf_processed:
    st.info("Please upload and process a PDF to start asking questions.")

