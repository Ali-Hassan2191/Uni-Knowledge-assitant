import os
import json
from pathlib import Path

import streamlit as st
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="University Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAG_DIR = BASE_DIR / "rag_database"
FAISS_DIR = RAG_DIR / "faiss_db"
CONFIG_FILE = RAG_DIR / "config" / "rag_config.json"

MODEL_NAME = "openai/gpt-oss-120b"

DEFAULT_TOP_K = 5
MAX_HISTORY_MESSAGES = 8


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        /* Main application background */
        .stApp {
            background:
                radial-gradient(
                    circle at 10% 0%,
                    rgba(255, 165, 0, 0.12),
                    transparent 28%
                ),
                radial-gradient(
                    circle at 90% 10%,
                    rgba(255, 255, 255, 0.05),
                    transparent 25%
                ),
                #0b0f14;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        /* Hero */
        .hero {
            padding: 2rem 2.2rem;
            border-radius: 24px;
            background:
                linear-gradient(
                    135deg,
                    rgba(255, 165, 0, 0.18),
                    rgba(255, 255, 255, 0.04)
                );
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.25);
            margin-bottom: 1.4rem;
        }

        .hero-title {
            margin: 0;
            font-size: 2.25rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: white;
        }

        .hero-text {
            margin-top: 0.6rem;
            color: #b9c2cc;
            font-size: 1rem;
            line-height: 1.6;
        }

        /* Sidebar status */
        .status-card {
            padding: 0.9rem 1rem;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.045);
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 0.8rem;
        }

        /* Metrics */
        .metric-card {
            padding: 0.9rem;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.045);
            border: 1px solid rgba(255, 255, 255, 0.08);
            text-align: center;
        }

        .metric-value {
            font-size: 1.25rem;
            font-weight: 800;
            color: #ffa500;
        }

        .metric-label {
            color: #9da8b3;
            font-size: 0.78rem;
        }

        /* Source cards */
        .source-card {
            padding: 0.9rem 1rem;
            margin: 0.55rem 0;
            border-left: 4px solid #ffa500;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.045);
            border-top: 1px solid rgba(255, 255, 255, 0.06);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }

        .source-title {
            font-weight: 700;
            color: white;
            overflow-wrap: anywhere;
        }

        .source-meta {
            color: #ffa500;
            font-size: 0.86rem;
            margin-top: 0.25rem;
            overflow-wrap: anywhere;
        }

        .source-snippet {
            color: #b9c2cc;
            font-size: 0.86rem;
            margin-top: 0.5rem;
            line-height: 1.5;
            overflow-wrap: anywhere;
        }

        .footer {
            text-align: center;
            color: #6f7a85;
            font-size: 0.78rem;
            padding: 1.5rem 0 0.5rem;
        }

        /* Chat spacing */
        [data-testid="stChatMessage"] {
            margin-bottom: 0.75rem;
        }

        /* Sidebar width / readability */
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD RAG DATABASE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_rag_database():
    """Load the FAISS vector database and its configuration."""

    if not FAISS_DIR.exists():
        raise FileNotFoundError(
            f"FAISS directory not found:\n{FAISS_DIR}"
        )

    if not (FAISS_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            f"index.faiss was not found inside:\n{FAISS_DIR}"
        )

    if not (FAISS_DIR / "index.pkl").exists():
        raise FileNotFoundError(
            f"index.pkl was not found inside:\n{FAISS_DIR}"
        )

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"rag_config.json was not found:\n{CONFIG_FILE}"
        )

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = json.load(file)

    embedding_model = config.get(
        "embedding_model",
        "BAAI/bge-small-en-v1.5",
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore, config


# ============================================================
# GROQ CLIENT
# ============================================================

@st.cache_resource(show_spinner=False)
def get_groq_client():
    """Create and cache the Groq client."""

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your environment "
            "variables or Streamlit secrets."
        )

    return Groq(api_key=api_key)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(vectorstore, query, top_k):
    """Retrieve the most relevant chunks from FAISS."""

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = DEFAULT_TOP_K

    top_k = max(1, min(top_k, 20))

    return vectorstore.similarity_search_with_score(
        str(query),
        k=top_k,
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text, limit=600):
    """Normalize whitespace and limit preview length."""

    text = " ".join((text or "").split())

    if len(text) > limit:
        return text[:limit].rstrip() + "..."

    return text


# ============================================================
# BUILD LLM CONTEXT
# ============================================================

def build_context(retrieved_documents):
    """Convert retrieved LangChain Documents into LLM context."""

    context_parts = []

    for index, (document, distance) in enumerate(
        retrieved_documents,
        start=1,
    ):
        metadata = document.metadata or {}

        source_file = (
            metadata.get("source_file")
            or metadata.get("source_path")
            or metadata.get("file")
            or metadata.get("filename")
            or "Unknown document"
        )

        page_number = (
            metadata.get("page_number")
            or metadata.get("page")
            or metadata.get("original_page_index")
            or metadata.get("page_index")
            or "Unknown"
        )

        citation = (
            metadata.get("citation")
            or f"{source_file}, Page {page_number}"
        )

        context_parts.append(
            f"""
[SOURCE {index}]

Citation:
{citation}

Document:
{source_file}

Page:
{page_number}

Content:
{document.page_content}
"""
        )

    return "\n\n".join(context_parts)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(client, question, context, history):
    """Generate a grounded answer using Groq."""

    recent_history = history[-MAX_HISTORY_MESSAGES:]

    messages = [
        {
            "role": "system",
            "content": """
You are a University Student & Academic Knowledge Assistant.

Your job is to answer student questions using ONLY the
university knowledge supplied in the context.

Rules:

1. Do not invent university policies.
2. Do not invent fees.
3. Do not invent deadlines.
4. Do not invent GPA requirements.
5. Do not invent examination rules.
6. Do not use outside knowledge as university policy.
7. If the answer is not supported by the context,
   clearly say that the information was not found
   in the available university documents.
8. Keep answers clear and student-friendly.
9. When useful, mention the relevant document and page.
10. Never create a fake source.
""",
        }
    ]

    for message in recent_history:
        role = message.get("role")
        content = message.get("content")

        if role in {"user", "assistant"} and content:
            messages.append(
                {
                    "role": role,
                    "content": str(content),
                }
            )

    messages.append(
        {
            "role": "user",
            "content": f"""
UNIVERSITY KNOWLEDGE CONTEXT:

{context}

STUDENT QUESTION:

{question}

Answer the student's question using only the
university knowledge context above.
""",
        }
    )

    completion = client.chat.completions.create(
        messages=messages,
        model=MODEL_NAME,
        temperature=0.2,
        max_completion_tokens=900,
    )

    if not completion.choices:
        return (
            "I could not generate an answer from the "
            "available university documents."
        )

    answer = completion.choices[0].message.content

    if not answer:
        return (
            "I could not generate an answer from the "
            "available university documents."
        )

    return str(answer).strip()


# ============================================================
# SOURCE FORMATTER
# ============================================================

def create_sources(retrieved_documents):
    """Create serializable source dictionaries for session state."""

    sources = []

    for document, distance in retrieved_documents:
        metadata = document.metadata or {}

        source_file = (
            metadata.get("source_file")
            or metadata.get("source_path")
            or metadata.get("file")
            or metadata.get("filename")
            or "Unknown document"
        )

        page_number = (
            metadata.get("page_number")
            or metadata.get("page")
            or metadata.get("original_page_index")
            or metadata.get("page_index")
            or "Unknown"
        )

        citation = (
            metadata.get("citation")
            or f"{source_file}, Page {page_number}"
        )

        try:
            numeric_distance = float(distance)
        except (TypeError, ValueError):
            numeric_distance = None

        sources.append(
            {
                "file": str(source_file),
                "page": str(page_number),
                "citation": str(citation),
                "distance": numeric_distance,
                "snippet": clean_text(
                    str(document.page_content),
                    500,
                ),
            }
        )

    return sources


# ============================================================
# SOURCE UI
# ============================================================

def display_sources(sources):
    """Display retrieved sources safely.

    Important:
    Dynamic document text is escaped before being inserted into
    HTML. This prevents source content containing '<', '>', '&',
    etc. from breaking the Streamlit UI.
    """

    with st.expander("📚 View Retrieved Sources"):
        if not sources:
            st.info("No source information available.")
            return

        for index, source in enumerate(sources, start=1):
            if not isinstance(source, dict):
                continue

            source_file = (
                source.get("file")
                or source.get("source_file")
                or source.get("source_path")
                or source.get("filename")
                or "Unknown document"
            )

            page_number = (
                source.get("page")
                or source.get("page_number")
                or source.get("original_page_index")
                or "Unknown"
            )

            citation = (
                source.get("citation")
                or f"{source_file}, Page {page_number}"
            )

            distance = source.get("distance")

            snippet = (
                source.get("snippet")
                or source.get("content")
                or source.get("text")
                or "No preview available."
            )

            # Escape dynamic values before inserting them into HTML.
            import html

            source_file = html.escape(str(source_file))
            page_number = html.escape(str(page_number))
            citation = html.escape(str(citation))
            snippet = html.escape(clean_text(str(snippet), 500))

            distance_text = ""

            if distance is not None:
                try:
                    distance_text = (
                        f" • 🔎 Distance {float(distance):.4f}"
                    )
                except (TypeError, ValueError):
                    distance_text = ""

            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">
                        📄 Source {index}: {source_file}
                    </div>

                    <div class="source-meta">
                        📑 Page {page_number}
                        &nbsp; • &nbsp;
                        🔗 {citation}
                        {distance_text}
                    </div>

                    <div class="source-snippet">
                        {snippet}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🎓 University Knowledge Assistant</div>
        <div class="hero-text">
            Ask questions about academic policies, examinations,
            fees, scholarships, admissions, student services
            and university rules.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INITIALIZE SYSTEM
# ============================================================

try:
    with st.spinner("Loading university knowledge base..."):
        vectorstore, rag_config = load_rag_database()
        groq_client = get_groq_client()

    system_ready = True

except Exception as error:
    system_ready = False
    rag_config = {}

    st.error("⚠️ System initialization failed.")
    st.code(str(error))


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🎓 Assistant")

    if system_ready:
        st.success("Knowledge Base Ready")
    else:
        st.error("Knowledge Base Unavailable")

    st.markdown("---")

    if system_ready:
        document_count = rag_config.get(
            "document_count",
            "—",
        )

        chunk_count = rag_config.get(
            "chunk_count",
            "—",
        )

        embedding_model = rag_config.get(
            "embedding_model",
            "BAAI/bge-small-en-v1.5",
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                        {document_count}
                    </div>
                    <div class="metric-label">
                        Documents
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                        {chunk_count}
                    </div>
                    <div class="metric-label">
                        Chunks
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        st.caption(f"Embedding: `{embedding_model}`")
        st.caption(f"LLM: `{MODEL_NAME}`")

        top_k = st.slider(
            "🔎 Retrieved sources",
            min_value=3,
            max_value=8,
            value=DEFAULT_TOP_K,
        )

    else:
        top_k = DEFAULT_TOP_K

    st.markdown("---")

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        """
        <div class="status-card">
            <b>🔎 Traceable RAG</b><br>
            <span style="color:#9da8b3;">
                Every retrieved answer is connected
                to document and page metadata.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    role = message.get("role")

    if role not in {"user", "assistant"}:
        continue

    with st.chat_message(role):
        content = message.get("content", "")

        if content:
            st.markdown(content)

        if role == "assistant":
            sources = message.get("sources", [])

            if sources:
                display_sources(sources)


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask about university policies, fees, exams, scholarships..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:
    question = question.strip()

    if not question:
        st.warning("Please enter a question.")
        st.stop()

    if not system_ready:
        st.warning(
            "The knowledge base is not ready. "
            "Please fix the initialization error first."
        )
        st.stop()

    # Store user message.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("🔎 Searching university knowledge..."):
                retrieved_documents = retrieve_documents(
                    vectorstore,
                    question,
                    top_k,
                )

                if not retrieved_documents:
                    answer = (
                        "I could not find relevant information "
                        "in the available university documents."
                    )
                    sources = []
                else:
                    context = build_context(
                        retrieved_documents
                    )

                    with st.spinner("🤖 Generating answer..."):
                        answer = generate_answer(
                            groq_client,
                            question,
                            context,
                            st.session_state.messages[:-1],
                        )

                    sources = create_sources(
                        retrieved_documents
                    )

            st.markdown(answer)

            if sources:
                display_sources(sources)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        except Exception as error:
            error_message = (
                "I couldn't process the question. "
                "Please check the RAG database, API key, "
                "and application configuration."
            )

            st.error(error_message)
            st.caption(f"Technical detail: {error}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                }
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Built with Streamlit · FAISS · Hugging Face Embeddings · Groq
    </div>
    """,
    unsafe_allow_html=True,
)
