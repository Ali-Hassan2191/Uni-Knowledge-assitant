"""University Student & Academic Knowledge Assistant
RAG app: pre-built FAISS index + HuggingFace embeddings + Groq (gpt-oss-120b)."""

import html
import json
import os
from pathlib import Path

import streamlit as st
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
APP_TITLE = "University Knowledge Assistant"
GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

DB_DIR = Path(__file__).parent / "rag_database"
FAISS_DIR = DB_DIR / "faiss_db"
CONFIG_PATH = DB_DIR / "config" / "rag_config.json"
MANIFEST_PATH = DB_DIR / "metadata" / "document_manifest.json"

SUGGESTIONS = [
    "What are the undergraduate admission requirements?",
    "What is the minimum attendance for final exams?",
    "What GPA is needed for the merit scholarship?",
    "When is tuition due for the next semester?",
]

SYSTEM_PROMPT = """You are the University Knowledge Assistant for students.
Answer ONLY from the numbered context excerpts provided.
Rules:
- Cite every fact with its source tag, e.g. [S1] or [S2][S3].
- If the context does not contain the answer, say you could not find it in the
  university documents and suggest contacting the relevant university office.
- Never invent policies, dates, fees or GPA thresholds.
- Be clear and concise. Use short paragraphs or bullet points."""

st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="wide")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
:root{--pine:#0E3B43;--moss:#2E8B6F;--gold:#E2A93B;--ink:#1B2B30;--mist:#F4F7F8;--line:#DCE5E8;}
html, body, [class*="css"], .stMarkdown, .stChatInput textarea{font-family:'DM Sans',sans-serif;color:var(--ink);}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:4.5rem;max-width:920px;padding-bottom:2rem;}
.hero{display:flex;align-items:center;gap:1.2rem;background:linear-gradient(120deg,var(--pine) 0%,#155A5E 60%,var(--moss) 130%);
  border-radius:22px;padding:1.6rem 1.9rem;margin-bottom:1.4rem;color:#fff;box-shadow:0 10px 30px rgba(14,59,67,.18);}
.hero .badge{flex:0 0 64px;height:64px;border-radius:18px;background:rgba(255,255,255,.14);
  border:1px solid rgba(255,255,255,.28);display:flex;align-items:center;justify-content:center;}
.hero h1{font-family:'Fraunces',serif;font-size:2rem;margin:0 0 .25rem 0;padding:0;color:#fff;letter-spacing:-.01em;line-height:1.15;}
.hero p{margin:0;color:#CFE6E3;font-size:.98rem;line-height:1.5;}
@media (max-width:640px){.hero{flex-direction:column;align-items:flex-start;}.hero h1{font-size:1.6rem;}}
section[data-testid="stSidebar"]{background:var(--mist);border-right:1px solid var(--line);}
.stat{display:flex;justify-content:space-between;padding:.55rem .8rem;margin-bottom:.4rem;
  background:#fff;border:1px solid var(--line);border-radius:12px;font-size:.92rem;}
.stat b{color:var(--pine);}
[data-testid="stChatMessage"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:1rem 1.1rem;}
.src{border-left:4px solid var(--moss);background:var(--mist);border-radius:10px;padding:.7rem .9rem;margin:.5rem 0;}
.src .tag{display:inline-block;background:var(--pine);color:#fff;border-radius:6px;padding:0 .45rem;
  font-size:.78rem;font-weight:600;margin-right:.4rem;}
.src .cite{font-weight:600;color:var(--pine);}
.src .meta{font-size:.78rem;color:#5D7278;margin:.2rem 0 .35rem 0;}
.src .snip{font-size:.86rem;color:#33474D;line-height:1.5;}
.pill{display:inline-block;background:#FFF4DC;color:#8A5F0A;border:1px solid #F1D9A0;border-radius:999px;
  padding:0 .6rem;font-size:.75rem;font-weight:600;}
.stButton>button{border-radius:12px;border:1px solid var(--line);text-align:left;}
.stButton>button:hover{border-color:var(--moss);color:var(--pine);}

/* chat bubbles */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){background:#EEF6F3;border-color:#CFE5DC;}
[data-testid="stChatMessageAvatarUser"]{background:var(--pine)!important;color:#fff!important;}
[data-testid="stChatMessageAvatarAssistant"]{background:var(--moss)!important;color:#fff!important;}
/* chat input */
[data-testid="stChatInput"]{border:1.5px solid var(--line);border-radius:16px;background:#fff;}
[data-testid="stChatInput"]:focus-within{border-color:var(--moss);box-shadow:0 0 0 3px rgba(46,139,111,.18);}
[data-testid="stChatInput"] textarea{caret-color:var(--moss);}
/* send button: green background, white icon */
[data-testid="stChatInputSubmitButton"], [data-testid="stChatInput"] button{
  background:var(--moss)!important;color:#fff!important;border:none!important;border-radius:12px!important;}
[data-testid="stChatInputSubmitButton"] svg, [data-testid="stChatInput"] button svg{color:#fff!important;fill:#fff!important;}
[data-testid="stChatInputSubmitButton"]:hover, [data-testid="stChatInput"] button:hover{background:var(--pine)!important;}
[data-testid="stChatInputSubmitButton"]:disabled, [data-testid="stChatInput"] button:disabled{opacity:.55;}
.disclaimer{font-size:.78rem;color:#5D7278;line-height:1.45;margin-top:.8rem;}
#MainMenu, footer{visibility:hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_api_key() -> str | None:
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        key = None
    return key or os.environ.get("GROQ_API_KEY")


@st.cache_resource(show_spinner="Loading knowledge base…")
def load_vectorstore(model_name: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    # index.pkl is our own pickle created in Colab, so this is safe.
    return FAISS.load_local(
        str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True
    )


@st.cache_resource
def get_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def retrieve(vs: FAISS, query: str, k: int, files: list[str]) -> list[dict]:
    flt = (lambda m: m.get("source_file") in files) if files else None
    hits = vs.similarity_search_with_score(
        query, k=k, filter=flt, fetch_k=max(k * 6, 30)
    )
    results = []
    for doc, dist in hits:
        # Vectors are normalized, FAISS returns squared L2 -> cosine = 1 - d/2
        relevance = max(0.0, min(1.0, 1 - float(dist) / 2))
        results.append({"text": doc.page_content, "meta": doc.metadata, "score": relevance})
    return results


def build_messages(query: str, sources: list[dict], history: list[dict]) -> list[dict]:
    context = "\n\n".join(
        f"[S{i}] ({s['meta'].get('source_file')}, page {s['meta'].get('page_number')})\n{s['text']}"
        for i, s in enumerate(sources, 1)
    )
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    msgs.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})
    return msgs


def stream_answer(client: Groq, messages: list[dict], temperature: float):
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=1500,
        reasoning_effort="low",
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            m = s["meta"]
            snippet = html.escape(s["text"][:380].replace("\n", " ")) + "…"
            st.markdown(
                f"""<div class="src">
<span class="tag">S{i}</span><span class="cite">{html.escape(str(m.get('source_file', 'unknown')))}
 · Page {m.get('page_number', '?')}/{m.get('total_pages', '?')}</span>
<span class="pill" style="float:right">{s['score']:.0%} match</span>
<div class="meta">Chunk ID: {html.escape(str(m.get('chunk_id', '-')))}</div>
<div class="snip">{snippet}</div></div>""",
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------------
# Load knowledge base
# ----------------------------------------------------------------------------
config = read_json(CONFIG_PATH)
manifest = read_json(MANIFEST_PATH)

if not (FAISS_DIR / "index.faiss").exists():
    st.error("Vector database not found. Add the Colab output to `rag_database/faiss_db/`.")
    st.stop()

vectorstore = load_vectorstore(config.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
doc_names = sorted(d["source_file"] for d in manifest.get("documents", []))

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 Knowledge base")
    st.markdown(
        f"""<div class="stat"><span>Documents</span><b>{config.get('total_documents', len(doc_names))}</b></div>
<div class="stat"><span>Pages</span><b>{config.get('total_pages', '-')}</b></div>
<div class="stat"><span>Chunks</span><b>{config.get('total_chunks', '-')}</b></div>""",
        unsafe_allow_html=True,
    )
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Sources per answer", 2, 10, 5)
    temperature = st.slider("Creativity", 0.0, 1.0, 0.1, 0.05)
    selected = st.multiselect("Limit to documents", doc_names, placeholder="All documents")
    show_sources = st.toggle("Show sources", value=True)
    if st.button("🗑️ Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.markdown('<div class="disclaimer">Answers come from the uploaded university documents. Please confirm important deadlines and fees with the official office.</div>', unsafe_allow_html=True)
    st.caption(f"Model: `{GROQ_MODEL}`  \nEmbeddings: `{config.get('embedding_model', DEFAULT_EMBEDDING_MODEL)}`")

# ----------------------------------------------------------------------------
# Main UI
# ----------------------------------------------------------------------------
st.markdown(
    f"""<div class="hero">
<div class="badge"><svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10 12 5 2 10l10 5 10-5z"/><path d="M6 12v5c0 1.5 2.7 3 6 3s6-1.5 6-3v-5"/><path d="M22 10v6"/></svg></div>
<div><h1>{APP_TITLE}</h1>
<p>Ask about admissions, scholarships, attendance, fees and more. Every answer links back to its source document and page.</p></div></div>""",
    unsafe_allow_html=True,
)

api_key = get_api_key()
if not api_key:
    st.warning("Add `GROQ_API_KEY` to Streamlit secrets (or an environment variable) to start chatting.")
    st.stop()
client = get_client(api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("**Try a question**")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTIONS):
        if cols[i % 2].button(q, key=f"sugg_{i}", width="stretch"):
            st.session_state.queued = q
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"] == "user" else ":material/school:"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources:
            render_sources(msg.get("sources", []))

prompt = st.chat_input("Ask a question about the university…") or st.session_state.pop("queued", None)

if prompt:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=":material/school:"):
        try:
            with st.spinner("Searching documents…"):
                sources = retrieve(vectorstore, prompt, top_k, selected)
            answer = st.write_stream(
                stream_answer(client, build_messages(prompt, sources, history), temperature)
            )
            if show_sources:
                render_sources(sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error(f"Something went wrong while answering: {e}")
