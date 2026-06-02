import logging
import time
import os
from pathlib import Path
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
import streamlit as st
from dotenv import load_dotenv

from cite_rag.auth import authenticate_user, create_user, ensure_auth_storage, is_admin
from cite_rag.db import record_event
from cite_rag.vector_db import get_vectorstore, clear_database
from cite_rag.chunking import process_and_store_text
from cite_rag.retrieval import build_retriever
from cite_rag.generation import generate_grounded_answer

load_dotenv()

CSS_PATH = Path(__file__).with_name("styles").joinpath("app.css")


def init_custom_css():
    """Initialize custom CSS from the dedicated stylesheet."""
    if not CSS_PATH.exists():
        st.warning(f"Missing stylesheet: {CSS_PATH.name}")
        return

    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_config() -> DictConfig:
    """
    Load configuration from YAML files using Hydra.
    Clears the global Hydra instance to support Streamlit's re-run behavior.
    """
    # Clear Hydra's global state to avoid conflicts reloads.
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
        <div class="demo-chip-row" style="justify-content:center;">
            <span class="demo-chip">Demo workspace</span>
            <span class="demo-chip">Citation-aware RAG</span>
            <span class="demo-chip">PostgreSQL auth</span>
        </div>
        <h1>{title}</h1>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


def render_auth_gate(cfg):
    """Render login and signup forms until a user is authenticated."""

    st.markdown('<div class="section-header">Access Control</div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1.15, 0.85], gap="large")

    with left_col:
        st.markdown('<div class="auth-panel">', unsafe_allow_html=True)
        login_tab, signup_tab = st.tabs(["Log in", "Create account"])

        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="you@example.com")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in", use_container_width=True)

            if submitted:
                try:
                    user = authenticate_user(cfg, email, password)
                    if user:
                        st.session_state.current_user = user
                        st.session_state.authenticated = True
                        st.success(f"Welcome back, {user['display_name']}.")
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")
                except Exception as exc:
                    st.error(f"Login failed: {exc}")

        with signup_tab:
            # Conditionally allow self-signup based on config
            allow_signup = getattr(getattr(cfg, "auth", None), "allow_self_signup", True)
            if not allow_signup:
                st.warning("Self-signup is disabled for this demo.")
            else:
                with st.form("signup_form"):
                    display_name = st.text_input("Display name", placeholder="Your name")
                    email = st.text_input("Email address", placeholder="you@example.com")
                    password = st.text_input("Password", type="password")
                    confirm_password = st.text_input("Confirm password", type="password")
                    submitted = st.form_submit_button("Create account", use_container_width=True)

                if submitted:
                    if password != confirm_password:
                        st.error("Passwords do not match.")
                        return

                    if not display_name.strip() or not email.strip() or not password:
                        st.warning("Fill in all fields to continue.")
                        return

                    try:
                        user = create_user(cfg, display_name, email, password)
                        st.session_state.current_user = user
                        st.session_state.authenticated = True
                        st.success(f"Account created for {user['display_name']}.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Signup failed: {exc}")
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown(
            """
            <div class="auth-credentials">
                <div class="auth-credentials-title">Demo credentials</div>
                <div class="auth-credentials-grid">
                    <div class="auth-credentials-item"><span>Admin</span><br>Username: admin<br>Email: admin@citerag.com<br>Password: admin1234</div>
                    <div class="auth-credentials-item"><span>User</span><br>Username: user1<br>Email: user1@citerag.com<br>Password: user1234</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_user_sidebar(cfg, current_user):
    """Render the signed-in user's account controls."""

    with st.sidebar:
        st.markdown("### Account")
        st.write(f"**User:** {current_user['display_name']}")
        st.write(f"**Email:** {current_user['email']}")
        st.write(f"**Role:** {current_user['role']}")
        if st.button("Log out", use_container_width=True):
            record_event(cfg, current_user["id"], "logout")
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.session_state.pop("vectorstore", None)
            st.rerun()


def main():
    """Main application entry point."""

    cfg = load_config()

    # Ensure database for authentication is ready
    try:
        ensure_auth_storage(cfg)
    except Exception as exc:
        st.set_page_config(page_title=cfg.app.title, layout="wide", initial_sidebar_state="expanded")
        st.error(f"Unable to initialize PostgreSQL auth storage: {exc}")
        st.stop()

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

    # Gate content until user is authenticated
    if not st.session_state.get("authenticated"):
        render_header(cfg.app.title, cfg.app.description)
        render_auth_gate(cfg)
        return

    current_user = st.session_state.get("current_user")
    if not current_user:
        st.session_state.authenticated = False
        st.rerun()

    render_user_sidebar(cfg, current_user)
    
    # Render modern header
    render_header(cfg.app.title, cfg.app.description)

    # Initialize vector store (cached in session to avoid reinitialization)
    if "vectorstore" not in st.session_state:
        with st.spinner("Initializing vector database..."):
            st.session_state.vectorstore = get_vectorstore(cfg)
    vectorstore = st.session_state.vectorstore

    # Document upload and storage section
    st.markdown('<div class="section-header">Prepare Your Knowledge Base</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload .txt file (optional)", type=["txt"], label_visibility="collapsed")
    with col2:
        # Admin-only feature to clear the database
        if st.button("Clear Database", use_container_width=True, disabled=not is_admin(current_user)):
            try:
                with st.spinner("Clearing remote database..."):
                    clear_database(cfg)
                # Reset vectorstore to force re-initialization
                if "vectorstore" in st.session_state:
                    del st.session_state.vectorstore
                st.success("Database cleared successfully")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear database: {str(e)}")
        if not is_admin(current_user):
            st.caption("Admin only")

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
                    num_chunks, emb_tokens, emb_cost, emb_elapsed = process_and_store_text(
                        text,
                        vectorstore,
                        cfg,
                        owner_context=current_user,
                    )
                record_event(
                    cfg,
                    current_user["id"],
                    "upload",
                    {
                        "chunks": num_chunks,
                        "embedding_tokens": emb_tokens,
                        "embedding_cost": emb_cost,
                    },
                )
                
                # Display embedding metrics
                st.markdown('<div class="answer-section">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">Knowledge Base Updated</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-text">Successfully stored <strong>{num_chunks} chunks</strong> with embeddings</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-item"><strong>Time:</strong> <span class="metric-value">{emb_elapsed:.2f}s</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-item"><strong>Tokens:</strong> <span class="metric-value">~{emb_tokens:,}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-item"><strong>Cost:</strong> <span class="metric-value">${emb_cost:.5f}</span> estimated</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No text provided.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Question answering section
    st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Ask Your Question</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    query = st.text_input("Enter your question", placeholder="What does the document say about ...?", key="query_input", label_visibility="collapsed")
    
    if st.button("Generate Answer", use_container_width=True, type="primary") and query.strip():
        with st.spinner("Retrieving, reranking, and generating..."):
            start_t = time.time()
            retriever = build_retriever(vectorstore, cfg, user_context=current_user)

            # Generate answer with citations from retrieved documents
            answer, citations, est_tokens, est_cost = generate_grounded_answer(retriever, query, cfg)
            elapsed = time.time() - start_t

        # Answer section
        st.markdown('<div class="answer-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Answer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-text">{answer if answer else "No answer generated."}</div>', unsafe_allow_html=True)

        # Performance metrics
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-item"><strong>Time:</strong> <span class="metric-value">{elapsed:.2f}s</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-item"><strong>Tokens:</strong> <span class="metric-value">~{est_tokens:,}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-item"><strong>Cost:</strong> <span class="metric-value">${est_cost:.5f}</span> estimated</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Display source documents with citations
        if citations:
            st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Source Documents</div>', unsafe_allow_html=True)
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
                                <div class="source-meta-item"><strong>Source:</strong> {source}</div>
                                <div class="source-meta-item"><strong>Position:</strong> {pos}</div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No relevant chunks retrieved.")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()