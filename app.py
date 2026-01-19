"""
CiteRAG: Citation-aware Retrieval-Augmented Generation (RAG) Application

A Streamlit web application that enables users to upload documents, store them in a vector database,
and ask questions with grounded, cited answers using Gemini LLM and Pinecone vector storage.

Flow:
  1. User uploads/pastes text → chunked and stored in Pinecone with embeddings
  2. User asks a question → retrieved from vector DB, reranked by Cohere
  3. LLM generates answer with inline citations [1], [2], etc.
  4. Sources displayed with content snippets and metadata
"""

import logging
import time
import os
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
import streamlit as st
from dotenv import load_dotenv

from cite_rag.vector_db import get_vectorstore
from cite_rag.chunking import process_and_store_text
from cite_rag.retrieval import build_retriever
from cite_rag.generation import generate_grounded_answer

load_dotenv()


def init_custom_css():
    """Initialize custom CSS for modern styling."""
    st.markdown("""
    <style>
        /* Root color scheme */
        :root {
            --primary-color: #2563eb;
            --primary-dark: #1e40af;
            --primary-light: #60a5fa;
            --accent-color: #f59e0b;
            --success-color: #10b981;
            --error-color: #ef4444;
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --border-color: #334155;
        }
        
        /* Main container styling */
        .main {
            background-color: var(--bg-color);
            color: var(--text-primary);
        }
        
        /* Header styling with gradient */
        .header-container {
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
            padding: 3rem 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 10px 30px rgba(37, 99, 235, 0.3);
        }
        
        .header-container h1 {
            color: white;
            font-size: 2.8rem;
            margin: 0;
            font-weight: 800;
            letter-spacing: -0.5px;
        }
        
        .header-container p {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.1rem;
            margin: 0.5rem 0 0 0;
            font-weight: 300;
        }
        
        /* Card styling */
        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        .card:hover {
            border-color: var(--primary-color);
            box-shadow: 0 8px 24px rgba(37, 99, 235, 0.2);
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary-light);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.7rem 1.5rem;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
        }
        
        .stButton > button:active {
            transform: translateY(0px);
        }
        
        /* Secondary button */
        .secondary-btn {
            background: linear-gradient(135deg, #475569 0%, #334155 100%);
            color: white;
            border: 1px solid var(--border-color) !important;
        }
        
        .secondary-btn:hover {
            background: linear-gradient(135deg, #64748b 0%, #475569 100%);
            box-shadow: 0 4px 12px rgba(100, 116, 139, 0.3);
        }
        
        /* Input styling */
        .stTextInput input, .stTextArea textarea {
            background-color: #0f172a;
            color: var(--text-primary);
            border: 1px solid var(--border-color) !important;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            font-size: 1rem;
        }
        
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }
        
        /* Success message styling */
        .stSuccess {
            background-color: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success-color);
            border-radius: 8px;
            padding: 1rem;
        }
        
        /* Info message styling */
        .stInfo {
            background-color: rgba(59, 130, 246, 0.1);
            border: 1px solid var(--primary-color);
            border-radius: 8px;
            padding: 1rem;
        }
        
        /* Warning message styling */
        .stWarning {
            background-color: rgba(245, 158, 11, 0.1);
            border: 1px solid var(--accent-color);
            border-radius: 8px;
            padding: 1rem;
        }
        
        /* Source citation styling */
        .source-card {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.05) 0%, rgba(124, 58, 237, 0.05) 100%);
            border: 1px solid rgba(37, 99, 235, 0.3);
            border-radius: 10px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        
        .source-card:hover {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
            border-color: var(--primary-color);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
        }
        
        .citation-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
            color: white;
            border-radius: 50%;
            font-weight: 700;
            font-size: 0.9rem;
            margin-right: 0.75rem;
        }
        
        .source-content {
            color: var(--text-primary);
            font-size: 0.95rem;
            line-height: 1.6;
            margin-bottom: 0.75rem;
        }
        
        .source-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        .source-meta-item {
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }
        
        /* Answer section styling */
        .answer-section {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.05) 0%, rgba(16, 185, 129, 0.05) 100%);
            border: 1px solid rgba(37, 99, 235, 0.3);
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .answer-text {
            font-size: 1.05rem;
            line-height: 1.8;
            color: var(--text-primary);
            margin-bottom: 1rem;
        }
        
        /* Metrics section */
        .metrics-container {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        
        .metric-item {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }
        
        .metric-value {
            color: var(--primary-light);
            font-weight: 600;
        }
        
        /* Spinner customization */
        .stSpinner {
            color: var(--primary-color);
        }
        
        /* Spinner animation */
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* File uploader styling */
        .stFileUploader {
            border: 2px dashed var(--border-color);
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
            background-color: rgba(37, 99, 235, 0.02);
        }
        
        .stFileUploader:hover {
            border-color: var(--primary-color);
            background-color: rgba(37, 99, 235, 0.05);
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .header-container {
                padding: 2rem 1rem;
            }
            
            .header-container h1 {
                font-size: 2rem;
            }
            
            .metrics-container {
                flex-direction: column;
            }
        }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_config() -> DictConfig:
    """
    Load configuration from YAML files using Hydra.
    Clears the global Hydra instance to support Streamlit's re-run behavior.
    """
    try:
        GlobalHydra.instance().clear()
    except Exception:
        pass

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name="config")
    return cfg


def render_header(title: str, description: str):
    """Render a modern header with gradient background."""
    st.markdown(f"""
    <div class="header-container">
        <h1>{title}</h1>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    # Clear Hydra global instance to prevent conflicts between Streamlit reruns
    try:
        GlobalHydra.instance().clear()
    except Exception:
        pass

    cfg = load_config()

    # Configure logging with both console and file output
    if hasattr(cfg.logging, 'file') and cfg.logging.file:
        log_dir = os.path.dirname(cfg.logging.file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    handlers = [logging.StreamHandler()]
    if hasattr(cfg.logging, 'file') and cfg.logging.file:
        handlers.append(logging.FileHandler(cfg.logging.file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        format="%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
        handlers=handlers
    )
    logger = logging.getLogger(__name__)

    # Log app startup only once to avoid spamming logs during reruns
    if "app_started" not in st.session_state:
        logger.info("RAG app started via Streamlit\n" + OmegaConf.to_yaml(cfg, resolve=True))
        st.session_state.app_started = True

    # Initialize Streamlit UI with modern styling
    st.set_page_config(page_title=cfg.app.title, layout="wide", initial_sidebar_state="expanded")
    
    # Apply custom CSS
    init_custom_css()
    
    # Render modern header
    render_header(cfg.app.title, cfg.app.description)

    # Initialize vector store (cached in session to avoid reinitialization)
    if "vectorstore" not in st.session_state:
        with st.spinner("🔧 Initializing vector database..."):
            st.session_state.vectorstore = get_vectorstore(cfg)
    vectorstore = st.session_state.vectorstore

    # Document upload and storage section
    st.markdown('<div class="section-header"> Prepare Your Knowledge Base</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload .txt file (optional)", type=["txt"], label_visibility="collapsed")
    with col2:
        if st.button("🗑️ Clear DB", use_container_width=True):
            # Reset vectorstore to force re-initialization on next use
            if "vectorstore" in st.session_state:
                del st.session_state.vectorstore
            st.rerun()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    text_input = st.text_area("Enter your text", height=140, label_visibility="collapsed", placeholder="Paste text here (or upload above)")
    
    col_upload, col_clear = st.columns([4, 1])
    with col_upload:
        if st.button(" Store / Update Database", use_container_width=True, type="primary"):
            # Extract text from either uploaded file or text area
            text = ""
            if uploaded_file is not None:
                text = uploaded_file.read().decode("utf-8")
            elif text_input.strip():
                text = text_input.strip()

            if text:
                with st.spinner(" Chunking → Embedding → Upserting..."):
                    num_chunks, emb_tokens, emb_cost, emb_elapsed = process_and_store_text(text, vectorstore, cfg)
                
                # Display embedding metrics
                st.markdown('<div class="answer-section">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">✅ Knowledge Base Updated</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-text">Successfully stored <strong>{num_chunks} chunks</strong> with embeddings</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-item">⏱️ <span class="metric-value">{emb_elapsed:.2f}s</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-item">📊 <span class="metric-value">~{emb_tokens:,}</span> tokens</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-item">💰 <span class="metric-value">${emb_cost:.5f}</span> estimated</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ No text provided.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Question answering section
    st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">❓ Ask Your Question</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    query = st.text_input("Enter your question", placeholder="What does the document say about ...?", key="query_input", label_visibility="collapsed")
    
    if st.button("🔍 Generate Answer", use_container_width=True, type="primary") and query.strip():
        with st.spinner("🔎 Retrieving → Reranking → Generating..."):
            start_t = time.time()
            retriever = build_retriever(vectorstore, cfg)

            # Generate answer with citations from retrieved documents
            answer, citations, est_tokens, est_cost = generate_grounded_answer(retriever, query, cfg)
            elapsed = time.time() - start_t

        # Answer section
        st.markdown('<div class="answer-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">💡 Answer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-text">{answer if answer else "No answer generated."}</div>', unsafe_allow_html=True)

        # Performance metrics
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-item">⏱️ <span class="metric-value">{elapsed:.2f}s</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-item">📊 <span class="metric-value">~{est_tokens:,}</span> tokens</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-item">💰 <span class="metric-value">${est_cost:.5f}</span> estimated</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Display source documents with citations
        if citations:
            st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">📚 Source Documents</div>', unsafe_allow_html=True)
            for i, c in enumerate(citations, 1):
                content = c.get('content', '')
                source = c.get('source', 'Unknown')
                pos = c.get('position', 'N/A')

                # Truncate long content for readability
                display_content = content[:280] + "..." if len(content) > 280 else content

                st.markdown(f"""
                <div class="source-card">
                    <div style="display: flex; align-items: flex-start; gap: 1rem;">
                        <span class="citation-badge">[{i}]</span>
                        <div style="flex: 1;">
                            <div class="source-content">{display_content}</div>
                            <div class="source-meta">
                                <div class="source-meta-item">📄 <strong>{source}</strong></div>
                                <div class="source-meta-item">📍 Position {pos}</div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ No relevant chunks retrieved.")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()