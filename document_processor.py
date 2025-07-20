from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os, tempfile

def load_file(uploaded_file):
    suffix = uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    if suffix == "pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix == "txt":
        loader = TextLoader(tmp_path)
    elif suffix == "docx":
        loader = Docx2txtLoader(tmp_path)
    else:
        raise ValueError("Unsupported file format.")
    return loader.load()

def process_and_store(uploaded_file):
    docs = load_file(uploaded_file)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="chroma_db"
    )
    vectordb.persist()
