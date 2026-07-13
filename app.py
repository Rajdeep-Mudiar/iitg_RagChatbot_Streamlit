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

def get_ollama_models():
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models_data = response.json().get("models", [])
            return [m["name"] for m in models_data]
    except Exception:
        pass
    return ["llama3.2:1b", "qwen2.5-coder:7b", "gemma3:1b"]

@st.cache_resource
def create_llm(fallback_model=None):
    primary_llm = ChatGroq(
        model=LLM_CONFIG["default_model"],
        temperature=LLM_CONFIG["temperature"],
        max_tokens=LLM_CONFIG["max_tokens"],
        timeout=LLM_CONFIG["timeout"],
        max_retries=LLM_CONFIG["max_retries"]
    )
    if fallback_model:
        from langchain_community.llms import Ollama
        fallback_llm = Ollama(
            model=fallback_model,
            temperature=LLM_CONFIG["temperature"]
        )
        return primary_llm.with_fallbacks([fallback_llm])
    return primary_llm

# --- Pipeline Setup ---

# Sidebar LLM Settings
st.sidebar.title("LLM Settings")
enable_fallback = st.sidebar.checkbox("Enable Local Fallback", value=True, help="Use local Ollama model if Groq API fails")
fallback_model = None
if enable_fallback:
    ollama_models = get_ollama_models()
    default_index = 0
    for idx, model in enumerate(ollama_models):
        if "llama3.2:1b" in model or "gemma3:1b" in model:
            default_index = idx
            break
    fallback_model = st.sidebar.selectbox("Select Fallback Model", options=ollama_models, index=default_index)

st.sidebar.markdown("---")

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
                    import fitz
                    file_bytes = uploaded_file.read()
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    for page_idx, page in enumerate(doc):
                        text = page.get_text("text")
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
        st.info("Please upload PDF, TXT, or MD documents in the sidebar to build the knowledge base.")
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

# Optimized MultiQuery Prompt template for technical/mathematical QA
from langchain_core.prompts import PromptTemplate

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI assistant task solver and expert research assistant.
Your task is to analyze the user's input question and generate 3 different versions of the query.
These search queries should target different perspectives, sub-questions, terminology variations, abbreviations, or physical/mathematical formulations of the original question.
For instance:
- If the question contains acronyms or jargon, expand them to their full terms.
- If it involves math formulas or derivations, write the equations/formulas using standard symbols or describe them in clear text.
- If the query is complex, break it down into simpler search terms.
By generating multiple perspectives on the user query, your goal is to help the user retrieve the most relevant documents.

Original question: {question}

Provide these alternative queries separated by newlines. Do not add numbering, prefixes, introductory or concluding remarks. Just output the 3 alternative queries.
"""
)

# Create TinyBERT Reranker
tinybert_reranker = create_reranker(RERANK_CONFIG["default_model"])

# RAG Prompt Template
RAG_PROMPT = ChatPromptTemplate.from_template(
    """
    You are an expert research assistant.
    Use ONLY the context below to answer the question.
    If the answer is not found in the context, say you don't know.

    Formatting Instructions:
    - If your answer includes mathematical formulas, equations, symbols, or derivations, ALWAYS format them using LaTeX. Use double dollar signs `$$` for block equations (e.g. $$E = mc^2$$) and single dollar signs `$` for inline equations (e.g. $E = mc^2$).
    - If your answer includes code, programming blocks, or scripts, ALWAYS format them using markdown code block syntax with the appropriate language identifier (e.g. ```python ... ```).

    Context
    -------
    {context}

    Conversation History
    --------------------
    {chat_history}

    Question: {question}

    Answer:
    """
)

# --- RAG Chain Function ---

def condense_question(chat_history, question, llm):
    if not chat_history:
        return question, False
    
    # Format chat history as a string
    history_str = ""
    for msg in chat_history[-5:]: # Keep last 5 messages for context
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"
        
    condense_prompt = f"""You are an expert conversational analyzer. Your task is to analyze the conversation history and the follow-up question, then determine if the follow-up question is contextual (meaning it depends on the context of previous messages, references past topics, or uses pronouns like "it", "they", "this", "its", "that") or if it is a standalone/independent question.

Instructions:
1. If the question is contextual, you MUST rewrite it to be a fully independent standalone question. Replace all pronouns (like "it", "its", "this", "they", "their") or vague references with the actual names, concepts, equations, or protocols mentioned in the conversation history (e.g., if the history is about "BBM92 protocol" and the question is "How secure is it?", rewrite it to "How secure is the BBM92 protocol?"). Do NOT leave any pronouns unresolved.
2. If the question is already independent and does not refer to anything in the history, output the follow-up question exactly as-is.

Your output must be in the following exact format, with no other text, markdown, or explanation:
Is Contextual: <True/False>
Question: <the reformulated or original question>

Conversation History:
{history_str}

Follow-up Question: {question}
"""

    try:
        response = llm.invoke(condense_prompt)
        text = response.content.strip() if hasattr(response, "content") else str(response).strip()
        is_contextual = False
        standalone_q = question
        
        # Robust parsing of the output to handle varying LLM formats/markups/bullets
        has_labels = "is contextual" in text.lower() or "question:" in text.lower()
        if has_labels:
            for line in text.split("\n"):
                # Clean prefix bullet points, numbers, and common markdown syntax
                cleaned_line = re.sub(r"^[-*\d.\s#]+", "", line).strip()
                cleaned_line = cleaned_line.replace("*", "").replace("`", "").strip()
                
                if re.match(r"^Is\s+Contextual\s*:", cleaned_line, re.IGNORECASE):
                    val = re.sub(r"^Is\s+Contextual\s*:\s*", "", cleaned_line, flags=re.IGNORECASE).strip().lower()
                    is_contextual = "true" in val
                elif re.match(r"^Question\s*:", cleaned_line, re.IGNORECASE):
                    standalone_q = re.sub(r"^Question\s*:\s*", "", cleaned_line, flags=re.IGNORECASE).strip()
        else:
            # Fallback if the LLM output doesn't use the format labels at all
            standalone_q = text.replace("*", "").replace("`", "").strip()
            is_contextual = standalone_q.lower() != question.lower()
            
        return standalone_q, is_contextual
    except Exception:
        pass
    return question, False

def get_rag_response(question, chat_history, llm, vectorstore, reranker):
    # 1. Condense the question using chat history
    standalone_question, is_contextual = condense_question(chat_history, question, llm)
    
    # 2. Retrieve documents using MultiQuery with standalone question (created dynamically with current llm)
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        llm=llm,
        prompt=QUERY_PROMPT
    )
    retrieved_docs_multiquery = retriever.invoke(standalone_question)
    
    # 3. Rerank retrieved documents using TinyBERT
    pairs = [
        (standalone_question, doc.page_content)
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

    # Format history for prompt ONLY if it is contextual
    history_str = ""
    if is_contextual:
        for msg in chat_history[-5:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

    # 4. Generate answer using LLM
    rag_chain = RAG_PROMPT | llm | StrOutputParser()
    answer = rag_chain.invoke({
        "context": context,
        "chat_history": history_str,
        "question": standalone_question
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
            llm_instance = create_llm(fallback_model) # Create LLM instance for each request to avoid caching issues
            response, retrieved_reranked_docs = get_rag_response(prompt, st.session_state.messages[:-1], llm_instance, vectorstore, tinybert_reranker)
            st.markdown(response)
            
            with st.expander("Retrieved Documents (Reranked Top 5)"):
                for i, doc in enumerate(retrieved_reranked_docs):
                    st.write(f"**Document {i+1} (Source: {doc.metadata.get('filename', 'N/A')}, Page: {doc.metadata.get('page_label', 'N/A')})**")
                    st.text(doc.page_content[:500] + "...") # Display first 500 chars
                    st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": response})
