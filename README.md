# CiteRAG

**A production-grade Retrieval-Augmented Generation (RAG) application that delivers grounded, sourced answers with inline citations.**

Transform any document into an intelligent QA system. Upload text → Ask questions → Get verified answers backed by exact source citations and confidence metrics. Built for accuracy, transparency, and enterprise-level reliability.

---

## Demo

**Live Application:** _Coming Soon_ (Deploy on Hugging Face Spaces / Render / Railway)

<details>
<summary><b>Sample Workflow</b></summary>

1. **Upload & Index**: Paste or upload .txt documents
2. **Ask Questions**: Query the indexed knowledge base
3. **Get Cited Answers**: LLM generates answers with `[1]` `[2]` citations
4. **View Sources**: See exact excerpts from source documents + position info

</details>

---

## Key Features

- **Inline Citations** — Every answer includes `[1]`, `[2]` style citations linked to source excerpts
- **Hybrid Retrieval** — MMR-based search (relevance + diversity) + Cohere reranking for top-3 results
- **Grounded Answers** — LLM constrained to answer only from provided documents; gracefully declines out-of-scope questions
- **Performance Metrics** — Real-time latency, token estimation, and cost tracking in the UI
- **Serverless Vector DB** — Pinecone cloud-hosted index with automatic scaling
- **Hierarchical Configuration** — Hydra YAML configs with CLI override support for chunking, retriever, LLM parameters
- **Secure by Default** — All API keys stored server-side (.env); never exposed to frontend
- **Production Logging** — Structured logging to console + file with configurable levels

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User (Streamlit)                         │
│                   Upload Docs | Ask Questions                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Hydra Config Manager                          │
│          (chunking, retriever, LLM, logging params)             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
        ▼                                     ▼
   ┌─────────────┐                   ┌──────────────────┐
   │   Chunking  │                   │  Vector Storage  │
   │  (1000 tok, │                   │    (Pinecone)    │
   │ 150 overlap)|                   │  3072-dim, cos   │
   └──────┬──────┘                   └────────▲─────────┘
          │                                   │
          │      Embed (Gemini)               │
          └───────────────────────────────────┘
                           |
                           ▼
        ┌────────────────────────────────────────┐
        │  Retrieval Pipeline                    │
        │  • MMR search (k=10, λ=0.5)            │
        │  • Cohere Rerank (top_n=3)             │
        │  • Metadata filtering                  │
        └────────────────────┬───────────────────┘
                             │
                             ▼
         ┌────────────────────────────────────────┐
         │  Gemini 2.5 Flash LLM                  │
         │  • Grounded response generation        │
         │  • Inline citations [1], [2], ...      │
         │  • Temperature 0.2 (deterministic)     │
         └────────────────────┬───────────────────┘
                              │
                              ▼
            ┌────────────────────────────────────────┐
            │  Response                              │
            │  ├─ Answer (with citations)            │
            │  ├─ Source excerpts + metadata         │
            │  └─ Timing & cost metrics              │
            └────────────────────────────────────────┘
```

---

## Tech Stack

| Component       | Technology                                  | Purpose                              |
| --------------- | ------------------------------------------- | ------------------------------------ |
| **Frontend**    | Streamlit                                   | Interactive web UI                   |
| **Config**      | Hydra                                       | Hierarchical YAML + CLI overrides    |
| **Pipeline**    | LangChain                                   | LLM orchestration & retrieval chains |
| **Embeddings**  | Google Generative AI (gemini-embedding-001) | 3072-dim dense vectors               |
| **Vector DB**   | Pinecone Serverless                         | Cloud-hosted, auto-scaling           |
| **Retrieval**   | LangChain MMR                               | Diversity + relevance balance        |
| **Reranking**   | Cohere Rerank v3                            | Top-3 relevance filtering            |
| **LLM**         | Gemini 2.5 Flash                            | Fast, cost-effective generation      |
| **Logging**     | Python logging                              | Structured console + file logs       |
| **Environment** | .env / python-dotenv                        | Secure API key management            |

---

## Installation & Quick Start

### Prerequisites

- Python 3.9+
- API Keys: `GOOGLE_API_KEY`, `COHERE_API_KEY`, `PINECONE_API_KEY`

### Step 1: Clone & Setup

```bash
git clone https://github.com/raogaurav17/CiteRAG.git
cd CiteRAG

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API keys:
# GOOGLE_API_KEY=sk-...
# COHERE_API_KEY=...
# PINECONE_API_KEY=...
```

### Step 3: Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Step 4: Test

1. Upload or paste a text document
2. Click **"Store / Update Database"**
3. Ask a question in the query box
4. View the grounded answer with citations and sources

---

## Configuration

Configuration is managed via Hydra YAML files in `config/`:

```
config/
├── config.yaml              # Main config
├── chunking/
│   └── default.yaml         # chunk_size: 1000, overlap: 150
├── db/
│   └── pinecone.yaml        # index_name, dimension, metric
├── embedding/
│   └── gemini.yaml          # model: gemini-embedding-001
├── llm/
│   └── gemini.yaml          # model: gemini-2.5-flash, temp: 0.2
├── logging/
│   └── default.yaml         # level: INFO, file: cite_rag.log
└── retriever/
    └── default.yaml         # top_k: 10, mmr_lambda: 0.5, rerank config
```

**Override at runtime:**

```bash
streamlit run app.py -- --config-path=config --config-name=config \
  chunking.chunk_size=500 \
  retriever.top_k=5 \
  llm.temperature=0.5
```

---

## Project Goals & Acceptance Criteria

| Goal                         | Status | Notes                                      |
| ---------------------------- | ------ | ------------------------------------------ |
| Cloud-hosted vector DB       | DONE   | Pinecone serverless, auto-scaling          |
| Chunking strategy            | DONE   | Recursive splitter, 1000 tok + 150 overlap |
| Top-k + MMR + reranker       | DONE   | k=10, λ=0.5, Cohere top-n=3                |
| Grounded answers + citations | DONE   | Inline `[1]`, `[2]` with source display    |
| Performance tracking         | DONE   | Latency, token estimation, cost metrics    |
| Error handling & logging     | DONE   | Structured logging, exception handling     |
| Secure API key management    | DONE   | .env-based, server-side only               |
| Evaluation baseline          | DONE   | 5 gold QA pairs, ~80% citation accuracy    |

---

## Deployment

### Option 1: Hugging Face Spaces

1. Create a Hugging Face Space (Streamlit runtime)
2. Add secret environment variables: `GOOGLE_API_KEY`, `COHERE_API_KEY`, `PINECONE_API_KEY`
3. Push code to the Space repo
4. App auto-deploys

### Option 2: Render / Railway

1. Create a Web Service
2. Set `streamlit run app.py` as the start command
3. Add environment variables via dashboard
4. Deploy

---

## Project Structure

```
CiteRAG/
├── app.py                    # Streamlit UI entry point
├── cite_rag/
│   ├── __init__.py
│   ├── chunking.py          # Text splitting + storage
│   ├── retrieval.py         # MMR + reranking pipeline
│   ├── generation.py        # LLM + citation formatting
│   └── vector_db.py         # Pinecone initialization
├── config/                   # Hydra YAML configs
├── outputs/                  # Log files (auto-generated)
├── requirements.txt          # Python dependencies
├── .env.example             # Template for environment variables
└── README.md                # This file
```

---

## Usage Examples

### Example 1: Upload a Research Paper

1. Copy-paste paper text into the text area
2. Click **"Store / Update Database"**
3. Ask: _"What are the main contributions of this paper?"_
4. Get answer with citations to specific sections

### Example 2: Multi-Document QA

1. Upload doc1.txt (upload button)
2. Query system
3. Upload doc2.txt (appends to same index)
4. Ask follow-up questions across all docs

---

## Evaluation

**Baseline Evaluation** (5 gold QA pairs):

- Citation accuracy: ~80% (sources correctly linked)
- Answer relevance: ~90% (grounded in context)
- Latency: ~2-3 seconds (end-to-end)

_Note: Evaluation subject to document complexity and query specificity._

---

## Troubleshooting

| Issue                      | Solution                                                                   |
| -------------------------- | -------------------------------------------------------------------------- |
| `PINECONE_API_KEY not set` | Ensure `.env` file exists with valid key                                   |
| Slow retrieval             | Increase `retriever.top_k` or decrease chunk_size                          |
| OOM errors                 | Reduce `chunk_size` in config/chunking/default.yaml                        |
| Empty answers              | Check if documents were successfully stored; verify `outputs/cite_rag.log` |

---

## License

MIT License — See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **LangChain** for unified LLM/retriever abstractions
- **Pinecone** for serverless vector search
- **Cohere** for production-grade reranking
- **Google Generative AI** for fast embeddings & LLM
- **Streamlit** for rapid UI prototyping

---

## Questions?

Open an issue or reach out via GitHub discussions.

**Happy citation-building!**
