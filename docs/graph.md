# Automated Graph Generation

This document describes the automated graph generation capability implemented in the RAG Chatbot, which allows users to ask for visual data representations directly from their documents.

---

## 1. Graph Request Detection

The chatbot automatically scans incoming user prompts using a regular expression pattern to detect visual requests. 

### Trigger Patterns
The request is classified as a graph request if the prompt contains words or phrases matching any of these regex boundaries:
- `generate graph` / `create graph` / `make graph`
- `plot` / `graph` / `chart`
- `visualize` / `visualise`
- `line chart` / `bar chart` / `scatter plot`
- `histogram` / `pie chart`

---

## 2. Graph Data Extraction

Once a graph request is detected, the system extracts numeric details from the retrieved documents using the `GRAPH_EXTRACTION_PROMPT`.

### Rules Enforced
1. **Context Adherence**: Only numeric facts explicitly present in the retrieved chunks can be used.
2. **Anti-Hallucination**: The model is forbidden from inventing or guessing numeric points.
3. **No Estimations**: If values are missing, the system must not attempt to fill or interpolate them.
4. **Structured Format**: The model must respond *only* with a valid JSON representation matching the schema.

### JSON Schema
The LLM formats data into the following JSON structure:
```json
{
    "title": "Graph title",
    "x_label": "X axis label",
    "y_label": "Y axis label",
    "graph_type": "line | bar | scatter",
    "data": [
        {
            "x": 1,
            "y": 10
        },
        {
            "x": 2,
            "y": 20
        }
    ]
}
```
If no numerical data is found in the documents, the system returns:
```json
{
    "error": "No numerical data found"
}
```

---

## 3. Data Validation

Before plotting, the data passes through strict validation steps to prevent python plotting exceptions:
- **Type Checking**: Verifies the response is a dictionary/object and is not empty.
- **Error Checking**: Inspects if the JSON contains an `"error"` key.
- **Fields Presence**: Checks that `"data"`, `"x"`, and `"y"` columns exist.
- **Numeric Casting**: Converts all X and Y coordinates to float/numeric values using Pandas `to_numeric`. Non-numeric coordinates raise an error.
- **Size Checking**: Verifies the resulting dataframe contains at least 1 row of values.

---

## 4. Plotting & Rendering

If the validation passes, the system initializes `matplotlib.pyplot` and plots the chart:

- **Bar Chart (`bar`)**: Plots Y-values against X-values cast as strings.
- **Scatter Plot (`scatter`)**: Plots individual X and Y points.
- **Line Chart (`line` - Default)**: Sorts values by X-axis coordinate and plots a continuous line with circular markers (`o`).

The chart is styled with customized axis labels, title, tight layout padding, and a grid overlay (opacity `0.3`) before being rendered directly in Streamlit via `st.pyplot(fig)`.
