import os
import re
import json
import pickle
import time
import requests

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM

from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate
)
from langchain_core.output_parsers import StrOutputParser

from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from sentence_transformers import CrossEncoder


# LOAD ENVIRONMENT & SECRETS

load_dotenv()


def get_config_value(key: str, default: str = None) -> str:
    """Safely get config from os.environ or st.secrets."""
    val = os.getenv(key)
    if val:
        return val
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default


GROQ_API_KEY = get_config_value("GROQ_API_KEY")
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# LANGSMITH CONFIGURATION

LANGSMITH_TRACING = get_config_value(
    "LANGSMITH_TRACING",
    "false"
)

LANGSMITH_API_KEY = get_config_value(
    "LANGSMITH_API_KEY"
)

LANGSMITH_ENDPOINT = get_config_value(
    "LANGSMITH_ENDPOINT",
    "https://api.smith.langchain.com"
)

LANGSMITH_PROJECT = get_config_value(
    "LANGSMITH_PROJECT",
    "IIT_Streamlit_RAG_Chatbot"
)

if LANGSMITH_TRACING and LANGSMITH_TRACING.lower() == "true":

    os.environ["LANGSMITH_TRACING"] = "true"

    if LANGSMITH_API_KEY:
        os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY

    os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT



# APPLICATION CONFIGURATION

LLM_CONFIG = {

    "default_model":
        "openai/gpt-oss-120b",

    "temperature":
        0,

    "max_tokens":
        1024,

    "timeout":
        None,

    "max_retries":
        2
}


EMBEDDING_CONFIG = {

    "default_model":
        "sentence-transformers/all-MiniLM-L6-v2",

    "normalize_embeddings":
        True,

    "device":
        "cpu",

    "batch_size":
        32,

    "cache_folder":
        "./cache/embeddings"
}


CHUNK_CONFIG = {

    "chunk_size":
        500,

    "chunk_overlap":
        100,

    "separators": [
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        ", ",
        " ",
        ""
    ]
}


RERANK_CONFIG = {

    "default_model":
        "cross-encoder/ms-marco-TinyBERT-L-2-v2",

    "top_k":
        5,

    "batch_size":
        16
}



# STREAMLIT CONFIG

st.set_page_config(
    page_title="IIT RAG Chatbot",
    page_icon=None,
    layout="wide"
)


st.title("IIT Research RAG Chatbot")

st.caption(
    "Multi Query RAG + TinyBERT Reranking + Guardrails "
    "+ Graph Generation + LangSmith"
)



# API KEY CHECK

if not GROQ_API_KEY:

    st.error(
        "GROQ_API_KEY not found. Please provide it in your .env file or Streamlit Secrets."
    )

    st.stop()



# SIDEBAR

st.sidebar.title("Settings")


# ------------------------------------------------------------
# LANGSMITH STATUS
# ------------------------------------------------------------

st.sidebar.subheader("LangSmith")

if (
    LANGSMITH_TRACING
    and LANGSMITH_TRACING.lower() == "true"
    and LANGSMITH_API_KEY
):

    st.sidebar.success(
        "Tracing Enabled"
    )

    st.sidebar.caption(
        f"Project: {LANGSMITH_PROJECT}"
    )

else:

    st.sidebar.warning(
        "Tracing Disabled"
    )


st.sidebar.markdown("---")


# ============================================================
# GUARDRAIL SETTINGS
# ============================================================

st.sidebar.subheader("Guardrails")

enable_guardrails = st.sidebar.checkbox(
    "Enable Guardrails",
    value=True
)


if enable_guardrails:

    st.sidebar.success(
        "Guardrails Enabled"
    )

else:

    st.sidebar.warning(
        "Guardrails Disabled"
    )



# ============================================================
# GROQ & OLLAMA MODEL DISCOVERY
# ============================================================

def get_groq_models(api_key=None):

    fallback_models = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b",
        "qwen/qwen3.8-27b"
    ]

    key = api_key or os.getenv("GROQ_API_KEY")

    if not key:
        return fallback_models

    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=3
        )

        if response.status_code == 200:
            data = response.json().get("data", [])
            excluded = [
                "whisper",
                "guard",
                "audio",
                "orpheus",
                "vision",
                "embed"
            ]

            valid_models = []
            for item in data:
                if item.get("active", True):
                    model_id = item.get("id", "")
                    if not any(ex in model_id.lower() for ex in excluded):
                        valid_models.append(model_id)

            if valid_models:
                preferred = "openai/gpt-oss-120b"
                if preferred in valid_models:
                    valid_models.remove(preferred)
                    valid_models.insert(0, preferred)
                return valid_models

    except Exception:
        pass

    return fallback_models


def is_ollama_running():

    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=1
        )
        return response.status_code == 200
    except Exception:
        return False


def get_ollama_models():

    try:

        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=2
        )

        if response.status_code == 200:

            models_data = response.json().get(
                "models",
                []
            )

            models = [
                model["name"]
                for model in models_data
            ]

            if models:
                return models

    except Exception:
        pass


    return [
        "llama3.2:1b",
        "qwen2.5-coder:7b",
        "gemma3:1b"
    ]


# ============================================================
# LLM SETTINGS
# ============================================================

st.sidebar.subheader("LLM Settings")

available_groq_models = get_groq_models(GROQ_API_KEY)

selected_groq_model = st.sidebar.selectbox(
    "Primary Groq Model",
    options=available_groq_models,
    index=0,
    help="Select the active LLM hosted on Groq."
)

ollama_available = is_ollama_running()

enable_fallback = st.sidebar.checkbox(
    "Enable Ollama Local Fallback",
    value=ollama_available,
    help="Fallback to local Ollama if Groq is offline."
)

fallback_model = None

if enable_fallback:

    if not ollama_available:
        st.sidebar.caption("⚠️ Ollama is offline (localhost:11434 not reachable).")

    ollama_models = get_ollama_models()

    default_index = 0

    for index, model in enumerate(
        ollama_models
    ):

        if (
            "llama3.2:1b" in model
            or
            "gemma3:1b" in model
        ):

            default_index = index
            break


    fallback_model = st.sidebar.selectbox(
        "Fallback Model",
        options=ollama_models,
        index=default_index
    )


# ============================================================
# LOAD PROCESSED DOCUMENTS
# ============================================================

@st.cache_resource
def load_processed_documents():

    path = (
        "results/"
        "processed_documents.pkl"
    )

    if not os.path.exists(path):

        return None


    with open(
        path,
        "rb"
    ) as file:

        return pickle.load(file)


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def create_embedding_model(
    model_name
):

    return HuggingFaceEmbeddings(

        model_name=model_name,

        model_kwargs={
            "device":
                EMBEDDING_CONFIG["device"]
        },

        encode_kwargs={

            "normalize_embeddings":
                EMBEDDING_CONFIG[
                    "normalize_embeddings"
                ],

            "batch_size":
                EMBEDDING_CONFIG[
                    "batch_size"
                ]
        },

        cache_folder=
            EMBEDDING_CONFIG[
                "cache_folder"
            ]
    )


# ============================================================
# CHUNKER
# ============================================================

@st.cache_resource
def create_recursive_chunker(
    chunk_config
):

    return RecursiveCharacterTextSplitter(

        chunk_size=
            chunk_config["chunk_size"],

        chunk_overlap=
            chunk_config["chunk_overlap"],

        separators=
            chunk_config["separators"],

        keep_separator=True,

        add_start_index=True,

        strip_whitespace=True
    )


# ============================================================
# RERANKER
# ============================================================

@st.cache_resource
def create_reranker(
    model_name
):

    return CrossEncoder(
        model_name,
        max_length=512
    )


# ============================================================
# VECTOR STORE
# ============================================================

@st.cache_resource
def get_vectorstore(
    documents,
    _embedding_model,
    _chunker
):

    chunks = _chunker.split_documents(
        documents
    )


    for index, chunk in enumerate(
        chunks
    ):

        if "chunk_id" not in chunk.metadata:

            chunk.metadata[
                "chunk_id"
            ] = str(index)


    vectorstore = FAISS.from_documents(
        chunks,
        _embedding_model
    )


    return vectorstore, chunks


# ============================================================
# CREATE LLM
# ============================================================

@st.cache_resource
def create_llm(
    model_name=None,
    fallback_model=None,
    groq_api_key=None
):

    target_model = (
        model_name
        or LLM_CONFIG["default_model"]
    )

    target_key = (
        groq_api_key
        or os.getenv("GROQ_API_KEY")
    )

    primary_llm = ChatGroq(
        model=target_model,
        groq_api_key=target_key,
        temperature=
            LLM_CONFIG[
                "temperature"
            ],
        max_tokens=
            LLM_CONFIG[
                "max_tokens"
            ],
        timeout=
            LLM_CONFIG[
                "timeout"
            ],
        max_retries=
            LLM_CONFIG[
                "max_retries"
            ]
    )


    if fallback_model:

        try:

            fallback_llm = OllamaLLM(
                model=fallback_model,
                temperature=
                    LLM_CONFIG[
                        "temperature"
                    ]
            )

            return primary_llm.with_fallbacks(
                [fallback_llm]
            )

        except Exception:
            pass


    return primary_llm


# ============================================================
# DOCUMENT INGESTION
# ============================================================

processed_documents = (
    load_processed_documents()
)


if processed_documents is not None:

    st.sidebar.success(
        f"Loaded "
        f"{len(processed_documents)} "
        f"pages/documents"
    )


    if st.sidebar.button(
        "Clear Documents"
    ):

        path = (
            "results/"
            "processed_documents.pkl"
        )


        if os.path.exists(path):

            os.remove(path)


        st.cache_resource.clear()

        st.rerun()


else:

    st.sidebar.warning(
        "No documents loaded"
    )


    uploaded_files = (
        st.sidebar.file_uploader(
            "Upload PDF, TXT or MD",
            type=[
                "pdf",
                "txt",
                "md"
            ],
            accept_multiple_files=True
        )
    )


    if st.sidebar.button(
        "Process & Ingest"
    ):

        if not uploaded_files:

            st.sidebar.error(
                "Upload at least one file."
            )

        else:

            all_docs = []


            for uploaded_file in (
                uploaded_files
            ):

                filename = (
                    uploaded_file.name
                )


                # ------------------------------------------------
                # PDF
                # ------------------------------------------------

                if filename.lower().endswith(
                    ".pdf"
                ):

                    import fitz


                    file_bytes = (
                        uploaded_file.read()
                    )


                    pdf = fitz.open(
                        stream=file_bytes,
                        filetype="pdf"
                    )


                    for page_index, page in enumerate(
                        pdf
                    ):

                        text = page.get_text(
                            "text"
                        )


                        if (
                            text
                            and
                            text.strip()
                        ):

                            all_docs.append(

                                Document(

                                    page_content=text,

                                    metadata={

                                        "filename":
                                            filename,

                                        "page_label":
                                            page_index + 1
                                    }
                                )
                            )


                # ------------------------------------------------
                # TXT / MD
                # ------------------------------------------------

                else:

                    content = (
                        uploaded_file
                        .read()
                        .decode(
                            "utf-8",
                            errors="ignore"
                        )
                    )


                    if content.strip():

                        all_docs.append(

                            Document(

                                page_content=content,

                                metadata={

                                    "filename":
                                        filename,

                                    "page_label":
                                        1
                                }
                            )
                        )


            if all_docs:

                os.makedirs(
                    "results",
                    exist_ok=True
                )


                with open(
                    "results/"
                    "processed_documents.pkl",
                    "wb"
                ) as file:

                    pickle.dump(
                        all_docs,
                        file
                    )


                st.sidebar.success(
                    f"Ingested "
                    f"{len(all_docs)} "
                    f"pages/documents"
                )


                st.cache_resource.clear()

                st.rerun()


            else:

                st.sidebar.error(
                    "Could not extract text."
                )


if processed_documents is None:

    st.info(
        "Upload a PDF, TXT or MD document "
        "from the sidebar."
    )

    st.stop()


# ============================================================
# CREATE RAG COMPONENTS
# ============================================================

embedding_model = (
    create_embedding_model(
        EMBEDDING_CONFIG[
            "default_model"
        ]
    )
)


recursive_chunker = (
    create_recursive_chunker(
        CHUNK_CONFIG
    )
)


vectorstore, all_chunks = (
    get_vectorstore(
        processed_documents,
        embedding_model,
        recursive_chunker
    )
)


tinybert_reranker = (
    create_reranker(
        RERANK_CONFIG[
            "default_model"
        ]
    )
)


# ============================================================
# MULTI QUERY PROMPT
# ============================================================

QUERY_PROMPT = PromptTemplate(

    input_variables=[
        "question"
    ],

    template="""
You are an expert research assistant.

Generate exactly 3 alternative search queries
for the user's question.

The queries should explore:

1. Different terminology
2. Technical or mathematical formulation
3. A decomposed version of the question

Expand abbreviations when useful.

If the question contains equations,
include mathematical terminology.

Return ONLY the 3 queries.
One query per line.

Original question:

{question}
"""
)


# ============================================================
# RAG PROMPT
# ============================================================

RAG_PROMPT = ChatPromptTemplate.from_template(
"""
You are an expert research assistant.

Answer the user's question using ONLY
the supplied document context.

Rules:

1. Do not invent facts.
2. Do not invent numerical values.
3. If the answer cannot be found in the
   context, say:

   "I don't know based on the provided documents."

4. Mathematical equations must use LaTeX.

5. Code must use markdown code blocks.

6. Clearly distinguish information from
   the documents from reasoning.

Context
-------
{context}

Conversation History
--------------------
{chat_history}

Question
--------
{question}

Answer:
"""
)


# ============================================================
# GUARDRAIL
# ============================================================

def run_guardrail(
    question,
    llm
):

    guardrail_prompt = ChatPromptTemplate.from_template(
"""
You are a security and relevance guardrail
for a research document chatbot.

Classify the user's request.

Allowed requests:

- Questions about uploaded documents
- Summaries
- Explanations
- Mathematical analysis
- Technical questions
- Comparisons based on documents
- Data analysis
- Graph generation based on document data

Potentially unsafe or inappropriate requests
should be rejected.

Return ONLY:

ALLOWED

or

BLOCKED

User request:

{question}
"""
    )


    try:

        chain = (
            guardrail_prompt
            | llm
            | StrOutputParser()
        )


        result = chain.invoke({

            "question":
                question
        })


        result = result.strip().upper()


        if "BLOCKED" in result:

            return False


        return True


    except Exception:

        # Fail open for normal research use
        return True


# ============================================================
# QUESTION CONDENSATION
# ============================================================

def condense_question(
    chat_history,
    question,
    llm
):

    if not chat_history:

        return question, False


    history_str = ""


    for message in chat_history[-5:]:

        role = (
            "User"
            if message["role"] == "user"
            else "Assistant"
        )


        history_str += (
            f"{role}: "
            f"{message['content']}\n"
        )


    prompt = f"""
You are a conversational question analyzer.

Determine whether the new question depends
on the previous conversation.

If contextual, rewrite it as a fully
standalone question.

Resolve pronouns such as:

it
this
that
they
their
its

Return EXACTLY:

Is Contextual: True/False
Question: standalone question

Conversation:

{history_str}

New question:

{question}
"""


    try:

        response = llm.invoke(
            prompt
        )


        text = (
            response.content
            if hasattr(
                response,
                "content"
            )
            else str(response)
        )


        text = text.strip()


        is_contextual = False

        standalone_question = question


        for line in text.splitlines():

            cleaned = (
                line
                .replace(
                    "`",
                    ""
                )
                .strip()
            )


            if re.match(
                r"^Is\s+Contextual\s*:",
                cleaned,
                re.I
            ):

                value = re.sub(
                    r"^Is\s+Contextual\s*:\s*",
                    "",
                    cleaned,
                    flags=re.I
                )


                is_contextual = (
                    value.strip().lower()
                    == "true"
                )


            elif re.match(
                r"^Question\s*:",
                cleaned,
                re.I
            ):

                standalone_question = re.sub(
                    r"^Question\s*:\s*",
                    "",
                    cleaned,
                    flags=re.I
                ).strip()


        return (
            standalone_question,
            is_contextual
        )


    except Exception:

        return (
            question,
            False
        )


# ============================================================
# GRAPH REQUEST DETECTION
# ============================================================

def is_graph_request(
    question
):

    patterns = [

        r"\bgenerate\s+(a\s+)?graph\b",

        r"\bcreate\s+(a\s+)?graph\b",

        r"\bmake\s+(a\s+)?graph\b",

        r"\bplot\b",

        r"\bgraph\b",

        r"\bchart\b",

        r"\bvisualize\b",

        r"\bvisualise\b",

        r"\bline\s+chart\b",

        r"\bbar\s+chart\b",

        r"\bscatter\s+plot\b",

        r"\bhistogram\b",

        r"\bpie\s+chart\b"
    ]


    question = question.lower()


    return any(
        re.search(
            pattern,
            question
        )
        for pattern in patterns
    )


# ============================================================
# GRAPH EXTRACTION PROMPT
# ============================================================

GRAPH_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
[
    (
        "system",
        """
You are a numerical data extraction assistant.

The user wants to create a graph from
numerical information contained in documents.

IMPORTANT RULES:

1. ONLY use numerical data that appears
   in the provided context.

2. NEVER invent numerical values.

3. Do not estimate missing values.

4. Do not use outside knowledge.

5. Extract values exactly as they appear.

6. Return ONLY valid JSON.

Allowed graph types:

line
bar
scatter

Required JSON structure:

{{
    "title": "Graph title",
    "x_label": "X axis",
    "y_label": "Y axis",
    "graph_type": "line",
    "data": [
        {{
            "x": 1,
            "y": 10
        }}
    ]
}}

If numerical data cannot be found,
return:

{{
    "error": "No numerical data found"
}}

DOCUMENT CONTEXT:

{context}

USER REQUEST:

{question}
"""
    )
]
)


# ============================================================
# EXTRACT GRAPH DATA
# ============================================================

def extract_graph_data(
    question,
    context,
    llm
):

    try:

        prompt_value = (
            GRAPH_EXTRACTION_PROMPT.invoke(
                {
                    "context":
                        context,

                    "question":
                        question
                }
            )
        )


        response = llm.invoke(
            prompt_value
        )


        text = (

            response.content

            if hasattr(
                response,
                "content"
            )

            else str(response)
        )


        text = text.strip()


        # Remove markdown fences

        text = re.sub(
            r"^```json\s*",
            "",
            text,
            flags=re.I
        )


        text = re.sub(
            r"^```\s*",
            "",
            text
        )


        text = re.sub(
            r"\s*```$",
            "",
            text
        )


        text = text.strip()


        # First attempt (direct load)

        try:

            return json.loads(
                text
            )


        except json.JSONDecodeError:

            pass


        # Brace balancing attempt to extract the first complete JSON object

        start_idx = text.find("{")

        if start_idx != -1:

            balance = 0

            for i in range(start_idx, len(text)):

                if text[i] == "{":

                    balance += 1

                elif text[i] == "}":

                    balance -= 1

                    if balance == 0:

                        candidate = text[
                            start_idx : i + 1
                        ]


                        try:

                            return json.loads(
                                candidate
                            )

                        except json.JSONDecodeError:

                            pass

                        break


        return {

            "error":
                "LLM returned invalid JSON."
        }


    except Exception as e:

        return {

            "error":
                f"Graph extraction failed: {str(e)}"
        }


# ============================================================
# VALIDATE GRAPH DATA
# ============================================================

def validate_graph_data(
    graph_data
):

    if not isinstance(
        graph_data,
        dict
    ):

        return False, None, (
            "Invalid graph data."
        )


    if "error" in graph_data:

        return False, None, (
            graph_data["error"]
        )


    data = graph_data.get(
        "data",
        []
    )


    if not data:

        return False, None, (
            "No numerical data found."
        )


    try:

        df = pd.DataFrame(
            data
        )


    except Exception as e:

        return False, None, (
            f"Could not create dataframe: {e}"
        )


    if "x" not in df.columns:

        return False, None, (
            "X-axis data is missing."
        )


    if "y" not in df.columns:

        return False, None, (
            "Y-axis data is missing."
        )


    try:

        df["x"] = pd.to_numeric(
            df["x"],
            errors="raise"
        )

        df["y"] = pd.to_numeric(
            df["y"],
            errors="raise"
        )


    except Exception:

        return False, None, (
            "X and Y values must be numerical."
        )


    if len(df) == 0:

        return False, None, (
            "No valid numerical rows."
        )


    return True, df, None


# ============================================================
# GENERATE GRAPH
# ============================================================

def generate_graph(
    graph_data
):

    valid, df, error = (
        validate_graph_data(
            graph_data
        )
    )


    if not valid:

        return None, error


    graph_type = graph_data.get(
        "graph_type",
        "line"
    )


    title = graph_data.get(
        "title",
        "Document Data"
    )


    x_label = graph_data.get(
        "x_label",
        "X"
    )


    y_label = graph_data.get(
        "y_label",
        "Y"
    )


    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    if graph_type == "bar":

        ax.bar(
            df["x"].astype(str),
            df["y"]
        )


    elif graph_type == "scatter":

        ax.scatter(
            df["x"],
            df["y"]
        )


    else:

        df = df.sort_values(
            "x"
        )


        ax.plot(
            df["x"],
            df["y"],
            marker="o"
        )


    ax.set_title(
        title
    )


    ax.set_xlabel(
        x_label
    )


    ax.set_ylabel(
        y_label
    )


    ax.grid(
        True,
        alpha=0.3
    )


    fig.tight_layout()


    return fig, None


# ============================================================
# RAG PIPELINE
# ============================================================

def get_rag_response(
    question,
    chat_history,
    llm,
    vectorstore,
    reranker
):

    # --------------------------------------------------------
    # 1. QUESTION CONDENSATION
    # --------------------------------------------------------

    standalone_question, is_contextual = (
        condense_question(
            chat_history,
            question,
            llm
        )
    )


    # --------------------------------------------------------
    # 2. MULTI QUERY RETRIEVAL
    # --------------------------------------------------------

    try:
        retriever = (
            MultiQueryRetriever.from_llm(
                retriever=
                    vectorstore.as_retriever(
                        search_kwargs={
                            "k": 5
                        }
                    ),
                llm=llm,
                prompt=QUERY_PROMPT
            )
        )
        retrieved_docs = (
            retriever.invoke(
                standalone_question
            )
        )
    except Exception:
        # Fallback to direct similarity search if MultiQuery LLM call fails
        base_retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 5
            }
        )
        retrieved_docs = base_retriever.invoke(
            standalone_question
        )


    # --------------------------------------------------------
    # 3. RERANK
    # --------------------------------------------------------

    if not retrieved_docs:

        return (
            "I don't know based on the "
            "provided documents.",
            [],
            standalone_question,
            is_contextual
        )


    pairs = [

        (
            standalone_question,
            doc.page_content
        )

        for doc in retrieved_docs
    ]


    scores = reranker.predict(
        pairs,

        batch_size=
            RERANK_CONFIG[
                "batch_size"
            ],

        show_progress_bar=False
    )


    ranked_docs = sorted(

        zip(
            scores,
            retrieved_docs
        ),

        key=lambda x: x[0],

        reverse=True
    )


    top_docs = [

        doc

        for score, doc
        in ranked_docs[
            :RERANK_CONFIG[
                "top_k"
            ]
        ]
    ]


    # --------------------------------------------------------
    # 4. CONTEXT
    # --------------------------------------------------------

    context = "\n\n".join(

        doc.page_content

        for doc in top_docs
    )


    # --------------------------------------------------------
    # 5. CONVERSATION HISTORY
    # --------------------------------------------------------

    history_str = ""


    if is_contextual:

        for message in chat_history[-5:]:

            role = (

                "User"

                if message["role"] == "user"

                else "Assistant"
            )


            history_str += (
                f"{role}: "
                f"{message['content']}\n"
            )


    # --------------------------------------------------------
    # 6. GENERATE ANSWER
    # --------------------------------------------------------

    rag_chain = (
        RAG_PROMPT
        | llm
        | StrOutputParser()
    )


    answer = rag_chain.invoke({

        "context":
            context,

        "chat_history":
            history_str,

        "question":
            standalone_question
    })


    return (
        answer,
        top_docs,
        standalone_question,
        is_contextual
    )



# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask a question about the documents..."
)


if prompt:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role":
            "user",

        "content":
            prompt
    })


    with st.chat_message(
        "user"
    ):

        st.markdown(
            prompt
        )


    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Processing..."
        ):

            start_time = time.time()


            # ------------------------------------------------
            # CREATE LLM
            # ------------------------------------------------

            llm_instance = (
                create_llm(
                    model_name=selected_groq_model,
                    fallback_model=fallback_model,
                    groq_api_key=GROQ_API_KEY
                )
            )


            # ------------------------------------------------
            # GUARDRAIL
            # ------------------------------------------------

            if enable_guardrails:

                allowed = (
                    run_guardrail(
                        prompt,
                        llm_instance
                    )
                )


                if not allowed:

                    response = (
                        "I can't help with that "
                        "request. Please ask a question "
                        "related to the uploaded research "
                        "documents."
                    )


                    st.warning(
                        response
                    )


                    st.session_state.messages.append({

                        "role":
                            "assistant",

                        "content":
                            response
                    })


                    st.stop()


            # ------------------------------------------------
            # RAG
            # ------------------------------------------------

            try:

                (
                    response,
                    retrieved_docs,
                    standalone_question,
                    is_contextual
                ) = get_rag_response(

                    prompt,

                    st.session_state.messages[
                        :-1
                    ],

                    llm_instance,

                    vectorstore,

                    tinybert_reranker
                )

            except Exception as e:

                st.error(
                    f"⚠️ Error generating response from Groq: {e}\n\n"
                    "Tip: You can switch to another Groq model (e.g. `openai/gpt-oss-20b` or `qwen/qwen3.6-27b`) from the sidebar."
                )

                st.stop()


            elapsed_time = (
                time.time()
                - start_time
            )


            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            st.markdown(
                response
            )


            st.caption(
                f"Response time: "
                f"{elapsed_time:.2f} seconds"
            )


            # =================================================
            # GRAPH GENERATION
            # =================================================

            if is_graph_request(
                prompt
            ):

                st.markdown(
                    "### Graph Generation"
                )


                graph_context = "\n\n".join(

                    doc.page_content

                    for doc
                    in retrieved_docs
                )


                with st.spinner(
                    "Extracting numerical data..."
                ):

                    graph_data = (
                        extract_graph_data(

                            question=
                                standalone_question,

                            context=
                                graph_context,

                            llm=
                                llm_instance
                        )
                    )


                if "error" in graph_data:

                    st.error(
                        graph_data["error"]
                    )


                else:

                    valid, df, error = (
                        validate_graph_data(
                            graph_data
                        )
                    )


                    if not valid:

                        st.error(
                            error
                        )

                    else:

                        st.write(
                            "**Extracted Data**"
                        )


                        st.dataframe(
                            df,
                            width="stretch"
                        )


                        fig, graph_error = (
                            generate_graph(
                                graph_data
                            )
                        )


                        if graph_error:

                            st.error(
                                graph_error
                            )

                        else:

                            st.pyplot(
                                fig,
                                width="stretch"
                            )


                            st.download_button(

                                "Download Graph Data CSV",

                                data=df.to_csv(
                                    index=False
                                ),

                                file_name=
                                    "graph_data.csv",

                                mime=
                                    "text/csv"
                            )


            # =================================================
            # RETRIEVED DOCUMENTS
            # =================================================

            with st.expander(
                "Retrieved Documents"
            ):

                if not retrieved_docs:

                    st.write(
                        "No documents retrieved."
                    )

                else:

                    for index, doc in enumerate(
                        retrieved_docs
                    ):

                        st.markdown(
                            f"### Document {index + 1}"
                        )


                        st.write(
                            "**Source:** "
                            f"{doc.metadata.get('filename', 'N/A')}"
                        )


                        st.write(
                            "**Page:** "
                            f"{doc.metadata.get('page_label', 'N/A')}"
                        )


                        st.text(
                            doc.page_content[
                                :1000
                            ]
                        )


                        st.markdown(
                            "---"
                        )


            # =================================================
            # QUESTION INFORMATION
            # =================================================

            with st.expander(
                "RAG Execution Details"
            ):

                st.write(
                    "**Original Question:**"
                )

                st.write(
                    prompt
                )


                st.write(
                    "**Standalone Question:**"
                )

                st.write(
                    standalone_question
                )


                st.write(
                    "**Contextual Question:**"
                )

                st.write(
                    is_contextual
                )


                st.write(
                    "**Retrieved Chunks:**"
                )

                st.write(
                    len(retrieved_docs)
                )



    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role":
            "assistant",

        "content":
            response
    })