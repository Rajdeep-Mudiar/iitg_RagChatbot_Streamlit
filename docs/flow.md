# Operational Flows & Sequence Diagrams

This document details the exact process flows within the RAG Chatbot, visualizing how documents are ingested, and how queries are processed, filtered, expanded, retrieved, reranked, and checked for graph generation.

---

## 1. Document Ingestion Flow

The ingestion pipeline handles the file uploading, parsing, metadata extraction, and pickle serialization.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as Streamlit Sidebar UI
    participant Parser as PyMuPDF / Text Parser
    participant Disk as Storage (results/processed_documents.pkl)

    User->>UI: Upload files (.pdf, .txt, .md)
    User->>UI: Click "Process & Ingest"
    Note over UI: Check file extension
    alt is PDF
        UI->>Parser: Extract text page-by-page via PyMuPDF (fitz)
        Parser-->>UI: Return page text & metadata (page_label, filename)
    else is TXT / MD
        UI->>Parser: Read text directly & decode UTF-8
        Parser-->>UI: Return full document text & metadata (filename)
    end
    UI->>Disk: Serialize & save Document list (pickle format)
    UI-->>User: Show ingestion success message
```

---

## 2. Guardrails & Query Processing Flow

When a user submits a query, it is first evaluated by a security and relevance guardrail. If it passes, it proceeds to condensation, expansion, and vector retrieval.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Chat as Streamlit Chat Interface
    participant GR as LLM Guardrail Filter
    participant LLM as Groq LLM (Llama 3.3 70B)
    participant VectorDB as FAISS Index

    User->>Chat: Type message and hit Enter
    
    alt Guardrails Enabled
        Chat->>GR: Evaluate Query Relevance & Safety
        GR-->>Chat: Return [Passed / Blocked]
    end

    alt Query Blocked
        Chat-->>User: Show safety block / off-topic message
    else Query Passed
        Chat->>LLM: Pass Query + Chat History for Condensation
        Note over LLM: Resolve pronouns / references<br/>into a standalone query
        LLM-->>Chat: Return standalone query

        Chat->>LLM: Send standalone query for Multi-Query Expansion
        Note over LLM: Generate 3 alternative formulations<br/>targeting different terminologies & perspectives
        LLM-->>Chat: Return 3 alternative queries

        par Query 1 Search
            Chat->>VectorDB: Query FAISS Index (Top 5)
            VectorDB-->>Chat: Return docs list 1
        and Query 2 Search
            Chat->>VectorDB: Query FAISS Index (Top 5)
            VectorDB-->>Chat: Return docs list 2
        and Query 3 Search
            Chat->>VectorDB: Query FAISS Index (Top 5)
            VectorDB-->>Chat: Return docs list 3
        end

        Note over Chat: Union and deduplicate<br/>documents by unique chunk_id
    end
```

---

## 3. Reranking, Graph Generation & Response Flow

After retrieving chunks, they are reranked. If the query asks for a graph, the chatbot extracts structured data from the context and plots it using Matplotlib. Otherwise, it generates a standard text response.

```mermaid
flowchart TD
    subgraph Rerank[Reranking Phase]
        A[Deduplicated retrieved chunks] --> B[Pair each chunk with original standalone query]
        B --> C[Compute scores using cross-encoder/ms-marco-TinyBERT-L-2-v2]
        C --> D[Sort chunks by relevance score descending]
        D --> E[Filter and keep Top 5 chunks]
    end

    subgraph Decision[Flow Decision]
        E --> F{Is Graph Requested?}
    end

    subgraph GraphGen[Graph Generation Branch]
        F -->|Yes| G[Send context to LLM for data extraction]
        G --> H[Extract structured JSON: labels, values, chart type]
        H --> I[Validate JSON data structure]
        I --> J[Plot chart using Matplotlib: Line, Bar, or Scatter]
        J --> K[Render figure in Streamlit UI]
    end

    subgraph StandardResponse[Response Generation Branch]
        F -->|No| L[Build grounded prompt with Top 5 context chunks]
        L --> M{Invoke Groq API}
        M -->|Success| N[Stream response to Streamlit UI]
        M -->|Failure / Timeout| O[Fallback to local Ollama model]
        O --> N
    end

    K --> P[Display source attributions in UI expander]
    N --> P
```
