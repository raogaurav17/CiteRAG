# CiteRAG

**A production-grade Retrieval-Augmented Generation (RAG) application that delivers grounded, sourced answers with inline citations.**

Transform any document into an intelligent QA system. Upload text → Ask questions → Get verified answers backed by exact source citations and confidence metrics. Built for accuracy, transparency, and enterprise-level reliability.

---

## Live Demo & Repository

**Live Application:** https://citerag.streamlit.app/  
**Public GitHub Repository:** https://github.com/raogaurav17/CiteRAG  
**Portfolio/Resume:** [link](https://drive.google.com/file/d/1Yx94s16_yCBzLsI3_DTvLLjIrtx-Tysn/view?usp=sharing)

### Sample Workflow

1. **Upload & Index**: Paste or upload .txt documents
2. **Ask Questions**: Query the indexed knowledge base
3. **Get Cited Answers**: LLM generates answers with `[1]` `[2]` citations
4. **View Sources**: See exact excerpts from source documents + position info
5. **Track Costs**: Real-time token counts and cost estimation

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Configuration & Schema](#configuration--schema)
- [API & Components](#api--components)
- [Performance & Metrics](#performance--metrics)
- [Remarks & Trade-offs](#remarks--trade-offs)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)

---

## Quick Start

### Prerequisites

- **Python 3.9+**
- **API Keys**:
  - Google Gemini (`GOOGLE_API_KEY`)
  - Pinecone (`PINECONE_API_KEY`)
  - Cohere (optional, `COHERE_API_KEY`)

### Installation

```bash
# Clone repository
git clone https://github.com/raogaurav17/CiteRAG.git
cd CiteRAG

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
# GOOGLE_API_KEY=your_gemini_key
# PINECONE_API_KEY=your_pinecone_key
# COHERE_API_KEY=your_cohere_key (optional)
```

### Running the Application

```bash
# Start Streamlit app (local)
streamlit run app.py

# Access at http://localhost:8501
```

---

## Architecture

### System Design & Data Flow

```
┌─────────────────┐
│  User Input     │ (Text upload or paste)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  1. CHUNKING MODULE (cite_rag/chunking.py)              │
│  - RecursiveCharacterTextSplitter                       │
│  - Chunk Size: 1000 tokens (configurable)               │
│  - Overlap: 150 tokens (preserves context)              │
│  - Metadata tracking (source, position)                 │
│  Returns: (num_chunks, tokens, cost, elapsed_time)      │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  2. EMBEDDING MODULE (via Pinecone)                     │
│  - Model: Gemini Embedding 001                          │
│  - Dimension: 3072                                      │
│  - Cost: $0.15 per 1M tokens                            │
│  - Storage: Pinecone Serverless (AWS, us-east-1)        │
│  - Metric: Cosine Similarity                            │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  3. RETRIEVAL PIPELINE (cite_rag/retrieval.py)          │
│  - MMR (Max Marginal Relevance): λ=0.5                  │
│  - Top-K: 10 documents (configurable)                   │
│  - Optional Cohere Reranking (top-3)                    │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  4. GENERATION MODULE (cite_rag/generation.py)          │
│  - Model: Gemini 2.5 Flash                              │
│  - Temperature: 0.2 (grounded, deterministic)           │
│  - Max Tokens: 3072 output                              │
│  - Cost: $0.30 per 1M input tokens                      │
│  - Enforces inline citations [1], [2], ...              │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  RESPONSE                                    │
│  ├─ Answer + Citations [1]                   │
│  ├─ Source Documents with metadata           │
│  ├─ Performance metrics (time, tokens, cost) │
│  └─ Confidence indicators                    │
└──────────────────────────────────────────────┘
```

### Component Overview

| Component      | Purpose                                | Key Tech               | Config              |
| -------------- | -------------------------------------- | ---------------------- | ------------------- |
| **Chunking**   | Split documents into manageable chunks | LangChain TextSplitter | `config/chunking/`  |
| **Vector DB**  | Store & retrieve embeddings            | Pinecone Serverless    | `config/db/`        |
| **Embedding**  | Convert text to 3072-dim vectors       | Google Generative AI   | `config/embedding/` |
| **Retrieval**  | Find relevant documents via MMR        | LangChain + Cohere     | `config/retriever/` |
| **Generation** | Generate grounded answers              | Gemini 2.5 Flash       | `config/llm/`       |
| **UI**         | Modern, responsive frontend            | Streamlit + Custom CSS | `app.py`            |

---

## Configuration & Schema

### Configuration Directory Structure

```
config/
├── config.yaml                 # Main app config + defaults
├── embedding/
│   └── gemini.yaml            # Embedding model & pricing
├── llm/
│   └── gemini.yaml            # LLM model & pricing
├── db/
│   └── pinecone.yaml          # Database configuration
├── chunking/
│   └── default.yaml           # Text splitting parameters
├── retriever/
│   └── default.yaml           # Retrieval strategy
└── logging/
    └── default.yaml           # Logging configuration
```

### Core Configuration Schemas

**`config/config.yaml`** (Main)

```yaml
defaults:
  - _self_
  - embedding: gemini
  - db: pinecone
  - llm: gemini
  - chunking: default
  - retriever: default
  - logging: default

app:
  title: "CiteRAG"
  description: "Upload text -> Ask questions -> Get cited answers"
```

**`config/embedding/gemini.yaml`** (Embeddings)

```yaml
model: "gemini-embedding-001"
cost_per_million_tokens: 0.15 # Pricing for cost tracking
```

**`config/llm/gemini.yaml`** (Language Model)

```yaml
model: gemini-2.5-flash
temperature: 0.2 # Grounded responses (0.0-1.0)
max_tokens: 3072 # Max output tokens
cost_per_million_tokens: 0.30 # Pricing per 1M input tokens
```

**`config/db/pinecone.yaml`** (Vector Database)

```yaml
index_name: citerag
dimension: 3072 # Must match embedding model output
metric: cosine # Similarity metric
cloud: aws
region: us-east-1
```

**`config/chunking/default.yaml`** (Text Processing)

```yaml
chunk_size: 1000 # Tokens per chunk
chunk_overlap: 150 # Overlapping tokens for context
```

**`config/retriever/default.yaml`** (Retrieval Strategy)

```yaml
top_k: 10 # Initial documents to retrieve
mmr_lambda: 0.5 # MMR diversity (0=diversity, 1=relevance)
rerank:
  enabled: true # Use Cohere reranking
  top_n: 3 # Final ranked documents returned
  model: rerank-english-v3.0
```

### Pinecone Vector Schema

Each vector stored in the index contains:

```json
{
  "id": "md5_hash_of_content",
  "values": [0.123, -0.456, ..., 0.789],  // 3072 dimensions
  "metadata": {
    "source": "user_input",
    "title": "Uploaded / Pasted Text",
    "section": "main",
    "position": 0,                        // Chunk order in document
    "content": "Full text content..."     // For citation display
  }
}
```

---

## API & Components

### Core Modules & Functions

#### `chunking.process_and_store_text()`

```python
def process_and_store_text(raw_text: str, vectorstore, cfg) -> tuple:
    """
    Split text into chunks and upsert to Pinecone.

    Returns:
        (num_chunks: int,
         embedding_tokens: int,
         embedding_cost: float,
         elapsed_time: float)
    """
```

#### `retrieval.build_retriever()`

```python
def build_retriever(vectorstore, cfg):
    """
    Build retriever with MMR + optional Cohere reranking.

    Returns:
        LangChain Retriever instance
    """
```

#### `generation.generate_grounded_answer()`

```python
def generate_grounded_answer(retriever, query: str, cfg):
    """
    Generate answer from retrieved documents with citations.

    Returns:
        (answer_text: str,
         citations: List[Dict],
         total_tokens: int,
         estimated_cost: float)

    Citation format:
    [
      {
        "number": 1,
        "content": "Excerpt from source",
        "source": "user_input",
        "position": 0
      },
      ...
    ]
    """
```

#### `vector_db.get_vectorstore()`

```python
def get_vectorstore(cfg):
    """
    Initialize or retrieve Pinecone vectorstore.

    Returns:
        PineconeVectorStore instance
    """
```

#### `vector_db.clear_database()`

```python
def clear_database(cfg):
    """Delete all vectors from Pinecone index"""
```

---

## Performance & Metrics

### Typical Performance Characteristics

| Operation                 | Avg Time | Tokens Est.     | Cost (USD)   |
| ------------------------- | -------- | --------------- | ------------ |
| Embed 1 chunk (1K tokens) | 200ms    | ~1,000          | $0.00015     |
| Retrieval + Reranking     | 300ms    | ~500 (metadata) | Negligible   |
| Answer Generation         | 1-2s     | ~200-500        | $0.00030     |
| **Full Pipeline (1 Q)**   | **2-3s** | **~1,700**      | **$0.00050** |

### Cost Breakdown (Example)

For storing 100 chunks + 1 question:

- **Embedding 100 chunks**: 100K tokens × ($0.15/1M) = $0.015
- **Query retrieval**: Minimal overhead
- **Answer generation**: 1.5K tokens × ($0.30/1M) = $0.00045
- **Total**: ~$0.016 (under 2 cents)

---

## Remarks & Trade-offs

### Key Design Decisions

#### 1. **Token Estimation vs. Accuracy**

- **Decision**: Use heuristic `chars ÷ 4 ≈ tokens`
- **Why**: Fast estimation without additional API calls
- **Trade-off**: ~10-20% variance from actual token counts
- **Mitigation**: Costs still tracked realistically for budgeting

#### 2. **Fixed Embedding Dimension (3072)**

- **Decision**: Locked to Gemini embedding output
- **Why**: Cannot change without full index recreation
- **Trade-off**: Cannot switch embedding models without downtime
- **Future**: Plan for multi-model support with separate indexes

#### 3. **Chunk Size Tuning (1000 tokens)**

- **Decision**: 1000 tokens with 150 overlap
- **Why**: Balances context preservation and retrieval speed
- **Trade-off**:
  - Smaller chunks (500): Faster retrieval, less context
  - Larger chunks (2000): Better context, slower retrieval
- **Optimization**: No auto-tuning; fixed config applies to all documents

#### 4. **MMR + Reranking (2-stage)**

- **Decision**: MMR for diversity, Cohere for relevance
- **Why**: Retrieves 10 diverse docs, reranks to top-3
- **Trade-off**:
  - Improves relevance but adds 50-100ms latency
  - Increases API costs slightly
  - Could be disabled for speed-focused use cases

#### 5. **Temperature = 0.2 (Grounded Answers)**

- **Decision**: Low temperature for deterministic output
- **Why**: Ensures consistent citations and factuality
- **Trade-off**: Less creative; may sound repetitive
- **Alternative**: Increase to 0.5+ for more varied responses (risks hallucinations)

#### 6. **Cloud-Only, No Local Inference**

- **Decision**: All processing via remote APIs
- **Why**: Scalability, ease of deployment, no GPU requirements
- **Trade-off**:
  - Requires internet connection
  - Privacy concerns (data sent to cloud)
  - Latency (network overhead)
- **Future**: Option for local embeddings (Ollama, SentenceTransformers)

---

### Limitations

| Limitation                            | Impact                          | Workaround                        |
| ------------------------------------- | ------------------------------- | --------------------------------- |
| **Heuristic token counting**          | ~10-20% cost estimation error   | Use in budgeting with buffer      |
| **Fixed 1000-token chunks**           | May not suit all doc types      | Could implement adaptive chunking |
| **No document deduplication**         | Duplicate uploads waste storage | Manual cleanup or hash checking   |
| **Cohere reranking adds latency**     | +50-100ms per query             | Disable if speed critical         |
| **No query caching**                  | Repeated questions re-process   | Implement Redis caching           |
| **No user authentication**            | Multi-user safety concerns      | Add Streamlit session management  |
| **Temperature 0.2 limits creativity** | Boring but accurate answers     | Allow user temperature slider     |

---

### Future Enhancements

#### Short-term (1-2 weeks)

- [ ] Deploy to Hugging Face Spaces or Render
- [ ] Add temperature slider in UI
- [ ] Implement query result caching
- [ ] Support PDF uploads (pypdf)

#### Medium-term (1-2 months)

- [ ] Hybrid search (BM25 + semantic)
- [ ] Multi-document type support (PDFs, images, tables)
- [ ] User authentication & session management
- [ ] Cost analytics dashboard
- [ ] Streaming LLM output

#### Long-term (3-6 months)

- [ ] Local embedding option (SentenceTransformers)
- [ ] Fine-tuning on domain-specific data
- [ ] Automatic chunk size optimization
- [ ] Multi-language support
- [ ] Evaluation metrics (BLEU, ROUGE, METEOR)
- [ ] A/B testing framework for models

---

### Known Issues & Workarounds

| Issue                  | Root Cause                     | Workaround                        |
| ---------------------- | ------------------------------ | --------------------------------- |
| First query slow (~5s) | Pinecone index warmup          | Expected; becomes ~2-3s after     |
| Missing citations      | Poor document retrieval        | Increase `top_k` from 10 to 15+   |
| High token count       | Large context window           | Reduce chunk size to 500 tokens   |
| Rate limiting errors   | Too many simultaneous requests | Add 1s delay between requests     |
| Index not found        | Pinecone key misconfigured     | Verify `PINECONE_API_KEY` env var |

---

## Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests (when available)
pytest tests/ -v

# Code formatting
black cite_rag/ app.py

# Linting
pylint cite_rag/ app.py

# Type checking
mypy cite_rag/

# Build docs (when available)
sphinx-build -b html docs/ docs/_build/
```

---

## License

MIT License - See [LICENSE](LICENSE)

---

## Author

**Gaurav Rao**

- GitHub: [@raogaurav17](https://github.com/raogaurav17)
- Email: raogaurav17@gmail.com
- [Portfolio / Resume Link](https://drive.google.com/file/d/1Yx94s16_yCBzLsI3_DTvLLjIrtx-Tysn/view?usp=sharing)

---

## References

- [LangChain Docs](https://python.langchain.com)
- [Pinecone Vector DB](https://www.pinecone.io)
- [Google Generative AI](https://ai.google.dev)
- [Cohere Reranking](https://cohere.com/rerank)
- [RAG Best Practices](https://arxiv.org/abs/2312.10997)
- [MMR Algorithm](https://en.wikipedia.org/wiki/Maximal_marginal_relevance)
- [Streamlit Docs](https://docs.streamlit.io)
- [Hydra Configuration](https://hydra.cc)
