# Security & Relevance Guardrails

This document describes the validation guardrails implemented in the RAG Chatbot, which protect the application from injections, off-topic conversations, hallucinations, and formatting/graphing errors.

---

## 1. Pre-Retrieval Input Guardrails

When a user submits a query, it is screened by a classification filter before invoking any retrieval steps or LLM chains. This preserves tokens and keeps the conversation aligned with the system's purpose.

### Logic
The guardrail evaluates the prompt and returns either `ALLOWED` or `BLOCKED`. 

```python
# app.py (run_guardrail)
def run_guardrail(question, llm):
    # Evaluates the classification prompt
    # Returns False if 'BLOCKED' is in the LLM response, otherwise True
```

### Allowed Categories
The guardrail permits:
- Questions directly asking about the uploaded research documents.
- Requests for summaries, explanations, comparisons, and technical details.
- Mathematical analyses and technical questions.
- Requests to generate charts or graphs based on document data.

### Blocked Categories
The guardrail rejects:
- Prompt injection or jailbreak attempts (e.g. "Ignore previous instructions...").
- Off-topic prompts (e.g., general advice, sports, coding assignments unrelated to papers).
- Unsafe, offensive, or inappropriate queries.

---

## 2. Context-Grounding Guardrails (Anti-Hallucination)

To ensure the chatbot only returns facts verified by the uploaded documents, the system employs strict instructions inside the `RAG_PROMPT`.

### Rules
1. **Strict Context Constraints**: The LLM must answer the user's question using **only** the supplied context chunks retrieved from the FAISS database.
2. **Anti-Fabrication**: The model is strictly prohibited from inventing facts or numerical values.
3. **Safe Handshake (I don't know)**: If the answer is not present in the retrieved chunks, the model is instructed to output the exact phrase:
   > *"I don't know based on the provided documents."*
4. **Attribution Separation**: The model must clearly separate facts found directly in the papers from its logical reasoning.

---

## 3. Formatting Guardrails

To maintain a clean and readable user interface, formatting guardrails are written into the prompt:
- **Math Equations**: All math equations and variables must be formatted in LaTeX math syntax (e.g., using `$E=mc^2$` or `$$\sum...$$`).
- **Code Blocks**: Any programming code must be formatted in standard Markdown code blocks with appropriate syntax highlighting.

---

## 4. Visual & Graphing Guardrails

When a user requests a chart or graph, the system uses validation guardrails to prevent python execution errors or rendering crashes:
- **Structured Data Extraction**: Graph queries are parsed, and the data is converted into a structured JSON string containing `title`, `graph_type`, `x_label`, `y_label`, `labels`, and `values`.
- **Dimensions Validation**: The graphing module verifies that arrays (e.g., labels and values) have matching sizes and contain valid numerical types.
- **Exception Catching**: If data extraction fails or is invalid, the chart plotter safely falls back to a standard text response and displays a validation warning instead of crashing.
