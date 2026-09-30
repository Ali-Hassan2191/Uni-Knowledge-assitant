import os
import json
import re
import html
from pathlib import Path

import streamlit as st
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="University Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# APP CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAG_DIR = BASE_DIR / "rag_database"
FAISS_DIR = RAG_DIR / "faiss_db"
CONFIG_FILE = RAG_DIR / "config" / "rag_config.json"

MODEL_NAME = "openai/gpt-oss-120b"
DEFAULT_TOP_K = 5
MAX_HISTORY_MESSAGES = 8


# ============================================================
# LIGHT PROFESSIONAL UI
# ============================================================

st.markdown(
    """
    <style>
        /* ---------- Global ---------- */
        .stApp {
            background: #f6f8fb;
            color: #172033;
        }

        [data-testid="stHeader"] {
            background: rgba(246, 248, 251, 0.92);
        }

        [data-testid="stAppViewContainer"] {
            background: #f6f8fb;
        }

        .main .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 7rem;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e5eaf0;
        }

        section[data-testid="stSidebar"] * {
            color: #243047;
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 9px;
            font-size: 1.18rem;
            font-weight: 750;
            color: #172033;
            margin-bottom: 18px;
        }

        .sidebar-status {
            background: #ecfdf5;
            border: 1px solid #ccefe0;
            color: #13795b;
            padding: 12px 14px;
            border-radius: 12px;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 20px;
        }

        .sidebar-divider {
            height: 1px;
            background: #e7ebf0;
            margin: 18px 0;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 14px 0;
        }

        .metric-card {
            background: #f8fafc;
            border: 1px solid #e6ebf1;
            border-radius: 12px;
            padding: 13px 8px;
            text-align: center;
        }

        .metric-value {
            color: #1d4ed8;
            font-size: 1.2rem;
            font-weight: 800;
        }

        .metric-label {
            color: #7a8798;
            font-size: 0.72rem;
            margin-top: 3px;
        }

        .model-info {
            background: #f8fafc;
            border: 1px solid #e6ebf1;
            border-radius: 11px;
            padding: 10px 12px;
            margin-top: 8px;
            color: #64748b;
            font-size: 0.76rem;
            line-height: 1.6;
            overflow-wrap: anywhere;
        }

        .trace-card {
            background: #f8fafc;
            border: 1px solid #e6ebf1;
            border-radius: 12px;
            padding: 13px;
            color: #64748b;
            font-size: 0.8rem;
            line-height: 1.5;
        }

        .trace-title {
            color: #243047;
            font-weight: 700;
            margin-bottom: 4px;
        }

        /* ---------- Hero ---------- */
        .hero {
            background: linear-gradient(135deg, #ffffff 0%, #eef5ff 100%);
            border: 1px solid #dfe8f5;
            border-radius: 22px;
            padding: 28px 32px;
            margin-bottom: 22px;
            box-shadow: 0 8px 28px rgba(30, 64, 175, 0.07);
        }

        .hero-title {
            display: flex;
            align-items: center;
            gap: 12px;
            color: #172033;
            font-size: 2rem;
            line-height: 1.2;
            font-weight: 800;
            letter-spacing: -0.025em;
        }

        .hero-icon {
            font-size: 1.85rem;
        }

        .hero-text {
            color: #64748b;
            margin-top: 9px;
            font-size: 0.98rem;
            line-height: 1.65;
            max-width: 850px;
        }

        /* ---------- Chat area ---------- */
        [data-testid="stChatMessage"] {
            border-radius: 16px;
            margin-bottom: 10px;
        }

        [data-testid="stChatMessageContent"] {
            color: #243047;
        }

        /* User message */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: #eaf2ff;
            border: 1px solid #d7e6ff;
        }

        /* Assistant message */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            background: #ffffff;
            border: 1px solid #e5eaf0;
            box-shadow: 0 3px 14px rgba(15, 23, 42, 0.035);
        }

        /* ---------- Chat input ---------- */
        [data-testid="stChatInput"] {
            background: #ffffff;
            border-top: 1px solid #e2e8f0;
            padding-top: 12px;
        }

        [data-testid="stChatInput"] > div {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 15px;
            box-shadow: 0 6px 24px rgba(15, 23, 42, 0.07);
        }

        [data-testid="stChatInput"] textarea {
            color: #172033 !important;
            background: #ffffff !important;
            font-size: 0.94rem !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #94a3b8 !important;
        }

        [data-testid="stChatInput"] button {
            color: #1d4ed8 !important;
        }

        /* ---------- Source cards ---------- */
        .source-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #3b82f6;
            border-radius: 11px;
            padding: 13px 15px;
            margin: 9px 0;
        }

        .source-title {
            color: #1e293b;
            font-weight: 750;
            font-size: 0.9rem;
            overflow-wrap: anywhere;
        }

        .source-meta {
            color: #2563eb;
            font-size: 0.77rem;
            margin-top: 4px;
            overflow-wrap: anywhere;
        }

        .source-snippet {
            color: #64748b;
            font-size: 0.8rem;
            line-height: 1.55;
            margin-top: 8px;
            overflow-wrap: anywhere;
        }

        /* ---------- Markdown / tables ---------- */
        .stMarkdown,
        [data-testid="stMarkdownContainer"] {
            color: #243047;
        }

        [data-testid="stMarkdownContainer"] table {
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0;
            font-size: 0.86rem;
            background: #ffffff;
        }

        [data-testid="stMarkdownContainer"] th {
            background: #eff6ff;
            color: #1e3a8a;
            font-weight: 700;
            text-align: left;
        }

        [data-testid="stMarkdownContainer"] th,
        [data-testid="stMarkdownContainer"] td {
            border: 1px solid #dbe3ed;
            padding: 9px 10px;
            vertical-align: top;
        }

        [data-testid="stMarkdownContainer"] tr:nth-child(even) td {
            background: #f8fafc;
        }

        /* ---------- Buttons ---------- */
        .stButton > button {
            border-radius: 10px;
            border: 1px solid #d7dee8;
            background: #ffffff;
            color: #334155;
            font-weight: 600;
        }

        .stButton > button:hover {
            border-color: #93b4e8;
            color: #1d4ed8;
            background: #f8fbff;
        }

        /* ---------- Slider ---------- */
        [data-testid="stSlider"] {
            padding-top: 4px;
        }

        /* ---------- Footer ---------- */
        .footer {
            text-align: center;
            color: #94a3b8;
            font-size: 0.75rem;
            padding: 20px 0 5px;
        }

        /* ---------- Mobile ---------- */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem 0.8rem 6rem;
            }

            .hero {
                padding: 22px 20px;
                border-radius: 17px;
            }

            .hero-title {
                font-size: 1.55rem;
            }

            .hero-text {
                font-size: 0.88rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RAG DATABASE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_rag_database():
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
# GROQ
# ============================================================

@st.cache_resource(show_spinner=False)
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to environment "
            "variables or Streamlit secrets."
        )

    return Groq(api_key=api_key)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(vectorstore, query, top_k):
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
# TEXT HELPERS
# ============================================================

def clean_text(text, limit=600):
    text = " ".join((text or "").split())

    if len(text) > limit:
        return text[:limit].rstrip() + "..."

    return text


def clean_model_answer(answer):
    """
    Clean accidental HTML produced by the model while preserving
    normal Markdown such as headings, bullets and tables.

    In the previous UI, the model could return literal <br>
    inside Markdown tables. That was appearing as raw text.
    """
    if not answer:
        return ""

    text = html.unescape(str(answer))

    # Convert common HTML line breaks to Markdown-friendly newlines.
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<p\s*>", "", text, flags=re.IGNORECASE)

    # Remove remaining simple HTML tags only if present.
    text = re.sub(r"<(?!https?://)[^>]+>", "", text)

    # Clean excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# CONTEXT
# ============================================================

def build_context(retrieved_documents):
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
# LLM ANSWER
# ============================================================

def generate_answer(client, question, context, history):
    recent_history = history[-MAX_HISTORY_MESSAGES:]

    messages = [
        {
            "role": "system",
            "content": """
You are a University Student & Academic Knowledge Assistant.

Answer questions using ONLY the university knowledge supplied
in the context.

Rules:
1. Do not invent university policies.
2. Do not invent fees.
3. Do not invent deadlines.
4. Do not invent GPA requirements.
5. Do not invent examination rules.
6. Do not use outside knowledge as university policy.
7. If the answer is not supported by the context, clearly say
   that the information was not found in the available
   university documents.
8. Keep answers clear, accurate and student-friendly.
9. When useful, mention the relevant document and page.
10. Never create a fake source.
11. Use Markdown for formatting.
12. For tables, use Markdown tables.
13. Do NOT use HTML tags such as <br>, <table>, <p>, etc.
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

Answer the student's question using only the university
knowledge context above.
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

    return clean_model_answer(answer)


# ============================================================
# SOURCE FORMATTER
# ============================================================

def create_sources(retrieved_documents):
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
    with st.expander("📚 Retrieved Sources", expanded=False):
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

            try:
                distance_text = (
                    f" • Similarity score: {float(distance):.4f}"
                ) if distance is not None else ""
            except (TypeError, ValueError):
                distance_text = ""

            # Dynamic source values are shown with Streamlit native
            # components instead of raw HTML.
            with st.container(border=True):
                st.markdown(
                    f"**📄 Source {index}: {source_file}**"
                )

                st.caption(
                    f"📑 Page {page_number}  •  "
                    f"🔗 {citation}{distance_text}"
                )

                st.write(
                    clean_text(str(snippet), 500)
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
        <div class="hero-title">
            <span class="hero-icon">🎓</span>
            <span>University Knowledge Assistant</span>
        </div>

        <div class="hero-text">
            Ask questions about academic policies, examinations,
            fees, scholarships, admissions, student services and
            university rules. Answers are grounded in your
            university knowledge base.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INITIALIZE
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
    st.markdown(
        """
        <div class="sidebar-brand">
            🎓 <span>Assistant</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if system_ready:
        st.markdown(
            '<div class="sidebar-status">✓ Knowledge Base Ready</div>',
            unsafe_allow_html=True,
        )
    else:
        st.error("Knowledge Base Unavailable")

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True,
    )

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

        st.markdown(
            f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{document_count}</div>
                    <div class="metric-label">Documents</div>
                </div>

                <div class="metric-card">
                    <div class="metric-value">{chunk_count}</div>
                    <div class="metric-label">Chunks</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="model-info">
                <b>Embedding</b><br>
                {embedding_model}<br><br>
                <b>LLM</b><br>
                {MODEL_NAME}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")

        top_k = st.slider(
            "🔎 Retrieved sources",
            min_value=3,
            max_value=8,
            value=DEFAULT_TOP_K,
            help="Number of relevant document chunks sent to the LLM.",
        )

    else:
        top_k = DEFAULT_TOP_K

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.markdown("")

    st.markdown(
        """
        <div class="trace-card">
            <div class="trace-title">🔎 Traceable RAG</div>
            Each answer can be inspected through its
            retrieved document sources and page metadata.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CHAT HISTORY
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
        University Knowledge Assistant ·
        Streamlit · FAISS · Hugging Face Embeddings · Groq
    </div>
    """,
    unsafe_allow_html=True,
)
