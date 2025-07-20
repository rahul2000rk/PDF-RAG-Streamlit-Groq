from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

def get_rag_chain():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        groq_api_key="YOUR_GROQ_API_KEY",
        model_name="llama3-8b-8192"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    return qa_chain.run
