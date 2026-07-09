import streamlit as st
import os
import pickle
import time
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# Configuration from the notebook (simplified for app.py)
LLM_CONFIG = {
    "default_model": "llama-3.3-70b-versatile",
    "temperature": 0,
    "max_tokens": 1024,
    "timeout": None,
    "max_retries": 2
}

EMBEDDING_CONFIG = {
    "default_model": "sentence-transformers/all-MiniLM-L6-v2",
    "normalize_embeddings": True,
    "device": "cpu",
    "batch_size": 32,
    "cache_folder": "./cache/embeddings"
}

CHUNK_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 100,
    "separators": ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
}

RERANK_CONFIG = {
    "default_model": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    "top_k": 5,
    "batch_size": 16
}

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", page_icon=":robot:")
st.title(":robot: RAG Chatbot")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Check for GROQ_API_KEY
if "GROQ_API_KEY" not in os.environ:
    st.error("GROQ_API_KEY not found. Please set it in your environment variables.")
    st.stop()

# --- Helper Functions ---

@st.cache_resource
def load_processed_documents():
    if not os.path.exists("results/processed_documents.pkl"):
        return None
    with open("results/processed_documents.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def create_embedding_model(model_name):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": EMBEDDING_CONFIG["device"]
        },
        encode_kwargs={
            "normalize_embeddings": EMBEDDING_CONFIG["normalize_embeddings"],
            "batch_size": EMBEDDING_CONFIG["batch_size"]
        },
        cache_folder=EMBEDDING_CONFIG["cache_folder"]
    )

@st.cache_resource
def create_recursive_chunker(chunk_config):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_config["chunk_size"],
        chunk_overlap=chunk_config["chunk_overlap"],
        separators=chunk_config["separators"],
        keep_separator=True,
        add_start_index=True,
        strip_whitespace=True
    )

@st.cache_resource
def create_reranker(model_name):
    return CrossEncoder(model_name, max_length=512)

@st.cache_resource
def get_vectorstore(documents, _embedding_model, _chunker):
    st.info("Creating vector store...")
    # Chunk documents
    chunks = _chunker.split_documents(documents)
    
    # Add unique IDs to chunks if not present (important for RRF)
    for i, chunk in enumerate(chunks):
        if 'chunk_id' not in chunk.metadata:
            chunk.metadata['chunk_id'] = str(i) # Simple ID for demo

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(chunks, _embedding_model)
    st.success("Vector store created.")
    return vectorstore, chunks

@st.cache_resource
def create_llm():
    return ChatGroq(
        model=LLM_CONFIG["default_model"],
        temperature=LLM_CONFIG["temperature"],
        max_tokens=LLM_CONFIG["max_tokens"],
        timeout=LLM_CONFIG["timeout"],
        max_retries=LLM_CONFIG["max_retries"]
    )

# --- Pipeline Setup ---

# Sidebar Document Ingestion / Status
st.sidebar.title("Document Management")
processed_documents = load_processed_documents()

if processed_documents is not None:
    st.sidebar.success(f"Loaded {len(processed_documents)} processed pages/documents.")
    if st.sidebar.button("Clear Documents"):
        if os.path.exists("results/processed_documents.pkl"):
            os.remove("results/processed_documents.pkl")
        st.cache_resource.clear()
        st.rerun()
else:
    st.sidebar.warning("No documents loaded.")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF, TXT, or MD files to build knowledge base:",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    if st.sidebar.button("Process & Ingest"):
        if uploaded_files:
            all_docs = []
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                if filename.endswith(".pdf"):
                    import pypdf
                    reader = pypdf.PdfReader(uploaded_file)
                    for page_idx, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text and text.strip():
                            all_docs.append(Document(
                                page_content=text,
                                metadata={"filename": filename, "page_label": page_idx + 1}
                            ))
                else:
                    content = uploaded_file.read().decode("utf-8", errors="ignore")
                    all_docs.append(Document(
                        page_content=content,
                        metadata={"filename": filename, "page_label": 1}
                    ))
            if all_docs:
                os.makedirs("results", exist_ok=True)
                with open("results/processed_documents.pkl", "wb") as f:
                    pickle.dump(all_docs, f)
                st.sidebar.success(f"Ingested {len(all_docs)} pages/documents!")
                st.cache_resource.clear()
                st.rerun()
            else:
                st.sidebar.error("Could not extract text from the files.")
        else:
            st.sidebar.error("Please upload at least one file.")

if processed_documents is None:
    try:
        st.info("👈 Please upload PDF, TXT, or MD documents in the sidebar to build the knowledge base.")
        st.stop()
    except Exception:
        import sys
        sys.exit(0)

# Create Embedding Model
embedding_model = create_embedding_model(EMBEDDING_CONFIG["default_model"])

# Create Chunker
recursive_chunker = create_recursive_chunker(CHUNK_CONFIG)

# Create Vectorstore
vectorstore, all_chunks = get_vectorstore(processed_documents, embedding_model, recursive_chunker)

# Create MultiQuery Retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), # Base retriever for MultiQuery
    llm=create_llm()
)

# Create TinyBERT Reranker
tinybert_reranker = create_reranker(RERANK_CONFIG["default_model"])

# RAG Prompt Template
RAG_PROMPT = ChatPromptTemplate.from_template(
    """
    You are an expert research assistant.
    Use ONLY the context below to answer the question.
    If the answer is not found in the context, say you don't know.

    Context
    -------
    {context}

    Question
    --------
    {question}

    Answer:
    """
)

# --- RAG Chain Function ---

def get_rag_response(question, llm, retriever, reranker):
    # 1. Retrieve documents using MultiQuery
    retrieved_docs_multiquery = retriever.invoke(question)
    
    # 2. Rerank retrieved documents using TinyBERT
    pairs = [
        (question, doc.page_content)
        for doc in retrieved_docs_multiquery
    ]
    scores = reranker.predict(
        pairs,
        batch_size=RERANK_CONFIG["batch_size"],
        show_progress_bar=False
    )
    ranked_docs = sorted(
        zip(scores, retrieved_docs_multiquery),
        key=lambda x: x[0],
        reverse=True
    )
    
    # Take top K reranked documents for context
    top_k_reranked_docs = [doc for score, doc in ranked_docs[:RERANK_CONFIG["top_k"]]]

    context = "\n\n".join(
        [
            doc.page_content
            for doc in top_k_reranked_docs
        ]
    )

    # 3. Generate answer using LLM
    rag_chain = RAG_PROMPT | llm | StrOutputParser()
    answer = rag_chain.invoke({
        "context": context,
        "question": question
    })
    return answer, top_k_reranked_docs

# --- Streamlit Chat Interface ---

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            llm_instance = create_llm() # Create LLM instance for each request to avoid caching issues
            response, retrieved_reranked_docs = get_rag_response(prompt, llm_instance, multi_query_retriever, tinybert_reranker)
            st.markdown(response)
            
            with st.expander("Retrieved Documents (Reranked Top 5)"):
                for i, doc in enumerate(retrieved_reranked_docs):
                    st.write(f"**Document {i+1} (Source: {doc.metadata.get('filename', 'N/A')}, Page: {doc.metadata.get('page_label', 'N/A')})**")
                    st.text(doc.page_content[:500] + "...") # Display first 500 chars
                    st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": response})
