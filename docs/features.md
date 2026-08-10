# Features & Capabilities

This chatbot is designed to provide highly accurate, contextual answers and visualizations based on user-supplied documents. Below are the key features and how they work.

---

## 1. Multi-Format Document Ingestion
- **Formats Supported**: `.pdf`, `.txt`, `.md`.
- **Page-Level Source Tracking**: For PDFs, page numbers are extracted and tracked in the metadata. This allows the chatbot to cite the exact page from which it retrieved its answers.
- **Robust Parsing**: Built on `PyMuPDF` (`fitz`), ensuring fast and accurate text extraction from complicated document layouts.

---

## 2. Advanced Contextual Retrieval

### Multi-Query Expansion
Standard retrieval can miss information if the user writes their question using different terminology than the document. The chatbot automatically generates **3 alternative formulations** of the query:
1. Explaining abbreviations or technical jargon.
2. Formulating search queries representing different perspectives.
3. Breaking complex requests down into simpler concepts.

All queries are executed in parallel against the FAISS index.

### Deduplication
The results from the multi-query retrieval are combined, and duplicate chunks are filtered out using unique `chunk_id` values, preserving space for fresh context.

---

## 3. TinyBERT Cross-Encoder Reranking
Retrieving more documents increases the chance of finding the answer, but feeding too much information to the LLM can clutter its context window and lead to "lost-in-the-middle" performance drop.
- **Reranker**: `cross-encoder/ms-marco-TinyBERT-L-2-v2`.
- **Operation**: The reranker scores the exact relevance of each retrieved chunk against the query. Only the top **5** highest-scoring chunks are kept.
- **Benefit**: Ensures the LLM receives the most relevant context, improving response quality and minimizing token usage.

---

## 4. Query Condensation & Conversation Memory
If you ask a follow-up question like *"How do I install it?"* after discussing the chatbot, the LLM will condense the history into a standalone query: *"How do I install the RAG chatbot?"* before querying the database. This allows natural, multi-turn conversations without losing context.

---

## 5. Security & Relevance Guardrails
A dedicated safety filter analyzes user prompts before execution. It prevents:
- Off-topic conversations unrelated to the uploaded documents.
- Prompt injection and jailbreaking attempts.
- Inappropriate or harmful queries.

If a query fails guardrail validation, the system responds with a helpful block message and skips the RAG pipeline to save tokens.

---

## 6. Automated Graph Generation
The chatbot can detect requests to plot data (e.g., *"create a graph of SNR vs distance"*).
- **Extraction**: The system extracts raw metrics and categories from the retrieved text context into structured JSON.
- **Validation**: Ensures labels and data arrays match in size and syntax.
- **Plotting**: Dynamically plots line graphs, bar charts, or scatter plots using Matplotlib and displays them inside the chat interface.

---

## 7. LangSmith Tracing & Observability
By configuring LangSmith, the system logs and traces all LLM calls, chain invocations, and retrievals. This provides developers with deep visibility into:
- Chain execution times and latency.
- LLM prompt inputs and raw generated outputs.
- Cost/Token counts and debugging traces.

---

## 8. Automatic Local Fallback (Ollama)
If the primary remote LLM (Groq) is unavailable due to rate limits or connection errors, the application automatically redirects the query to local models served by Ollama (e.g., `llama3.2:1b`), preventing downtime.
