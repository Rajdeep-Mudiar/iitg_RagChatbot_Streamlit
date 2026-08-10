# Installation & Setup Guide

This guide provides step-by-step instructions for setting up the RAG Chatbot locally.

---

## 1. Prerequisites

Before starting, ensure you have the following installed on your machine:
- **Python**: Version 3.9 to 3.13 (Python 3.11 is recommended).
- **Miniconda / Anaconda**: Recommended for environment management.
- **Git**: For cloning the repository.
- **Ollama** (Optional): For running local fallback LLMs. Download it from [ollama.com](https://ollama.com/).

---

## 2. Environment Setup

### Step 1: Create a Conda Environment
Open your terminal (or Anaconda Prompt on Windows) and run:
```bash
conda create -n iit_streamlit_chatbot python=3.11 -y
```

### Step 2: Activate the Environment
```bash
conda activate iit_streamlit_chatbot
```

---

## 3. Installing Dependencies

Install the required Python packages using `requirements.txt`:
```bash
pip install -r requirements.txt
```

> [!NOTE]  
> If you have a GPU (NVIDIA), you can install the GPU version of PyTorch for faster local embedding and cross-encoder reranking:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 4. Environment Configuration (`.env`)

Create a `.env` file in the root of the project and configure the following variables:

### Required Config
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

### LangSmith Tracing (Optional)
To enable LangSmith tracing and observability:
```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_your_langsmith_api_key_here
LANGSMITH_PROJECT=IIT_Streamlit_RAG_Chatbot
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

---

## 5. Local Fallback Configuration (Optional)

If the Groq API limit is exceeded or offline, the chatbot can fall back to local models using Ollama:
1. Ensure Ollama is installed and running in the background.
2. Pull one of the supported small models:
   ```bash
   ollama pull llama3.2:1b
   # or
   ollama pull gemma2:2b
   # or
   ollama pull qwen2.5-coder:7b
   ```
3. The chatbot will automatically detect and fall back to these models if the primary Groq API call fails.

---

## 6. Running the Application

To run the chatbot:
```bash
streamlit run app.py
```

This will spin up a local development server. A browser window should automatically open at `http://localhost:8501`.
