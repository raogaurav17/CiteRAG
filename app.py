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

    # Initialize Streamlit UI
    st.set_page_config(page_title=cfg.app.title, layout="wide")
    st.title(cfg.app.title)
    st.caption(cfg.app.description)

    # Initialize vector store (cached in session to avoid reinitialization)
    if "vectorstore" not in st.session_state:
        with st.spinner("Initializing vector database..."):
            st.session_state.vectorstore = get_vectorstore(cfg)
    vectorstore = st.session_state.vectorstore

    # Document upload and storage section
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload .txt file (optional)", type=["txt"])
    with col2:
        if st.button("Clear DB", type="secondary"):
            # Reset vectorstore to force re-initialization on next use
            if "vectorstore" in st.session_state:
                del st.session_state.vectorstore
            st.rerun()

    text_input = st.text_area("Paste text here (or upload above)", height=140)

    if st.button("Store / Update Database"):
        # Extract text from either uploaded file or text area
        text = ""
        if uploaded_file is not None:
            text = uploaded_file.read().decode("utf-8")
        elif text_input.strip():
            text = text_input.strip()

        if text:
            with st.spinner("Chunking → Embedding → Upserting..."):
                start_t = time.time()
                num_chunks = process_and_store_text(text, vectorstore, cfg)
                elapsed = time.time() - start_t
            st.success(f"Stored **{num_chunks}** chunks in {elapsed:.1f} seconds")
        else:
            st.warning("No text provided.")

    # Question answering section
    st.subheader("Ask a question")
    query = st.text_input("", placeholder="What does the document say about ...?", key="query_input")

    if st.button("Generate Answer") and query.strip():
        with st.spinner("Retrieving → Reranking → Generating..."):
            start_t = time.time()
            retriever = build_retriever(vectorstore, cfg)

            # Generate answer with citations from retrieved documents
            answer, citations, est_tokens, est_cost = generate_grounded_answer(retriever, query, cfg)
            elapsed = time.time() - start_t

        st.markdown("### Answer")
        st.markdown(answer if answer else "No answer generated.")

        # Display performance metrics
        st.caption(f"⏱ {elapsed:.2f} s  •  ~{est_tokens:,} tokens  •  est. cost ${est_cost:.5f}")

        # Display source documents with citations
        if citations:
            st.markdown("### Sources")
            for i, c in enumerate(citations, 1):
                content = c.get('content', '')
                source = c.get('source', 'Unknown')
                pos = c.get('position', 'N/A')

                # Truncate long content for readability
                display_content = content[:280] + "..." if len(content) > 280 else content

                st.markdown(
                    f"""
                    <div style="background-color: #262730; padding: 12px; border-radius: 5px; margin-bottom: 8px; border: 1px solid #464b59;">
                        <div style="font-size: 0.9em; margin-bottom: 4px;"><b>[{i}]</b> {display_content}</div>
                        <div style="font-size: 0.8em; color: #b4b4b4;"><i>Source: {source} • Position: {pos}</i></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No relevant chunks retrieved.")

if __name__ == "__main__":
    main()