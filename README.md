# 🎓 University Student & Academic Knowledge Assistant

A RAG chatbot that answers student questions from official university PDFs, with **every answer traced back to its source file and page**.

- **UI:** Streamlit
- **Vector DB:** FAISS (pre-built, loaded at startup, no runtime embedding of documents)
- **Embeddings:** `BAAI/bge-small-en-v1.5` (HuggingFace)
- **LLM:** Groq, `openai/gpt-oss-120b`

## Architecture

```
            OFFLINE (Google Colab, run once)
 Public Drive PDFs -> PyPDFLoader -> Chunking -> Embeddings -> FAISS + metadata
                                                                   |
                                                       rag_database/ (committed to GitHub)
                                                                   |
            ONLINE (Streamlit Cloud)                               v
 Student question -> embed query -> FAISS top-k search -> context + citations
                                                                   |
                                        Groq gpt-oss-120b (streamed answer with [S1], [S2] tags)
                                                                   |
                                     Answer + source cards (file, page, chunk ID, match %)
```

Only the query is embedded at runtime. The original PDFs are **not** in this repo.

## Repository structure

```
university-rag-assistant/
├── app.py                      # Streamlit application
├── requirements.txt
├── README.md
├── .gitignore                  # blocks PDFs and secrets
├── .streamlit/
│   ├── config.toml             # theme
│   └── secrets.toml.example    # template (real secrets.toml is never committed)
└── rag_database/               # output of the Colab notebook
    ├── faiss_db/
    │   ├── index.faiss
    │   └── index.pkl
    ├── config/
    │   └── rag_config.json
    └── metadata/
        ├── document_manifest.json
        └── chunk_metadata.json
```

## Source traceability

Each chunk stores: `source_file`, `page_number`, `total_pages`, `chunk_id`, `document_id`, `citation` and more. The answer cites sources as `[S1]`, `[S2]`, and the **Sources** panel shows the file, page, chunk ID, relevance and a text snippet.

## Setup

1. Run the Colab notebook and download `university_rag_database.zip`.
2. Extract it into the repo root **inside a folder named `rag_database/`** so that `rag_database/faiss_db/index.faiss` exists.
3. Push to GitHub:
   ```bash
   git add . && git commit -m "Add RAG app" && git push
   ```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .streamlit && cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then add your key
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app from your repo, main file `app.py`.
2. Open **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
3. Deploy.

## Important notes

- The embedding model in `rag_config.json` must match the one used to build the index. The app reads it automatically.
- `index.pkl` is loaded with `allow_dangerous_deserialization=True`. Only load index files you created yourself.
- First startup downloads the embedding model (~130 MB), which takes a moment.
- If deployment runs out of resources, uncomment the CPU-only `torch` lines in `requirements.txt`.
- To update the knowledge base, rebuild in Colab and replace the `rag_database/` folder.

## License

For educational use.
