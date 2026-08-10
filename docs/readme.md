# Project Documentation Index

Welcome to the RAG Chatbot documentation. Here is a guide to the available documentation files:

---

## Documentation Directory

### 1. [Installation & Setup](setup.md)
Contains step-by-step instructions for:
- Creating the Python virtual environment via Anaconda/Miniconda.
- Installing the required pip dependencies.
- Configuring environment variables (including Groq and LangSmith keys).
- Setting up Ollama for local LLM fallbacks.
- Running the application locally.

### 2. [Operational Flows & Diagrams](flow.md)
Detailed visual guides of internal workflows using Mermaid diagrams, including:
- The **Document Ingestion Pipeline** (extracting text from uploaded files and serialization).
- The **Guardrails & Query Retrieval Pipeline** (guardrails screening, query condensation, multi-query expansion, and FAISS indexing).
- The **Reranking & Graphing/Response Pipeline** (evaluating chunk relevance, matplotlib plotting, and LLM grounding).

### 3. [System Architecture](architecture.md)
High-level overview of system components, including:
- Data tiers, logic layers, and presentation tiers.
- Complete breakdown of the tech stack.
- Design rationale behind choices like in-memory FAISS, multi-querying, cross-encoder reranking, and local model fallback routing.

### 4. [Features & Capabilities](features.md)
Deep dive into chatbot capabilities, describing:
- Multi-format ingestion support (PDF, TXT, MD) with source-tracking.
- Multi-Query Expansion.
- Cross-Encoder relevance scoring.
- Conversation memory processing.
- Input Guardrails and Graph Generation (Matplotlib).
- Support for complex formatting such as code and LaTeX.
- Automatic offline/fallback mechanics.

### 5. [Security & Relevance Guardrails](guardrails.md)
Complete breakdown of all guardrail mechanism layers within the chatbot:
- **Pre-Retrieval Guardrails**: LLM screening categorization of queries.
- **Context-Grounding Guardrails**: Prompt constraints preventing hallucinations.
- **Formatting Guardrails**: Forcing LaTeX and Markdown coding structure.
- **Visual/Graphing Guardrails**: Validating data dimensions before Matplotlib renders.

### 6. [Automated Graph Generation](graph.md)
Technical manual of the visualization workflow:
- **Request Detection**: Regex triggering patterns.
- **JSON Schema Extraction**: Structured data templates.
- **Data Validation**: pandas verification of numerical values and dimensions.
- **Plotting & Rendering**: Matplotlib chart configurations.
