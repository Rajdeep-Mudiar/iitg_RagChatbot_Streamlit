# Tech Stack

This document outlines all the technologies and libraries used in the RAG Chatbot project.

---

## Core Framework & UI

- **Streamlit** - Web application framework for building the interactive chat interface
- **Python** - Primary programming language

---

## LLM & AI Stack

### Large Language Models

- **Groq (OpenAI GPT-OSS 120B / 20B, Qwen 3.6/3.8)** - Primary LLM for query condensation, multi-query expansion, and response generation
- **Ollama** - Local LLM fallback option for offline/alternative inference

### LLM Framework & Orchestration

- **LangChain** - Main framework for building RAG pipelines
- **langchain-core** - Core abstractions and components
- **langchain-groq** - Groq LLM integration
- **langchain-ollama** - Ollama LLM integration
- **langchain-community** - Community-provided integrations
- **langchain-huggingface** - HuggingFace embeddings integration
- **langchain-text-splitters** - Text chunking utilities
- **langchain-experimental** - Experimental features
- **langchain-classic** - Multi-query retriever implementation

### Embeddings & Vector Retrieval

- **sentence-transformers** - Pre-trained embedding model provider
  - Model: `sentence-transformers/all-MiniLM-L6-v2` for document/query embeddings
- **HuggingFace Embeddings** - Embedding generation from LangChain
- **FAISS** (Facebook AI Similarity Search) - Vector database for semantic search
  - **faiss-cpu** - CPU-optimized version for local vector indexing and retrieval

### Reranking & Ranking

- **Cross-Encoder** (TinyBERT)
  - Model: `cross-encoder/ms-marco-TinyBERT-L-2-v2` for context relevance ranking
  - Used to score and sort retrieved contexts before LLM generation

---

## Document Processing

- **PyMuPDF (fitz)** - PDF parsing and text extraction
- **pdfplumber** - Advanced PDF parsing and table extraction
- **pypdf** - Additional PDF manipulation utilities

---

## Data Processing & ML

- **pandas** - Data manipulation and analysis
- **PyTorch** - Deep learning framework (used by sentence-transformers and cross-encoder)
- **torchvision** - Computer vision utilities (PyTorch-dependent)

---

## Visualization

- **Matplotlib** - Plotting and graph generation for visualizations

---

## Environment & Configuration

- **python-dotenv** - Environment variable management from `.env` files

---

## Observability & Debugging

- **LangSmith** - Tracing and debugging service for LangChain applications

---

## Utilities

- **requests** - HTTP client for API calls

---

## Data Persistence

- **pickle** - Python object serialization for caching processed documents
- **JSON** - Configuration and data interchange format

---

## Summary by Category

| Category              | Technologies                                                |
| --------------------- | ----------------------------------------------------------- |
| **Framework**         | Streamlit, Python                                           |
| **LLM Models**        | Groq (Llama 3.3 70B), Ollama                                |
| **LLM Orchestration** | LangChain, langchain-core, langchain-groq, langchain-ollama |
| **Embeddings**        | sentence-transformers, HuggingFace, FAISS                   |
| **Reranking**         | Cross-Encoder (TinyBERT)                                    |
| **PDF Processing**    | PyMuPDF, pdfplumber, pypdf                                  |
| **ML/Data**           | PyTorch, torchvision, pandas                                |
| **Visualization**     | Matplotlib                                                  |
| **Configuration**     | python-dotenv                                               |
| **Observability**     | LangSmith                                                   |
