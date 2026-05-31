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

from cite_rag.auth import authenticate_user, create_user, ensure_auth_storage, is_admin
from cite_rag.db import record_event
from cite_rag.vector_db import get_vectorstore, clear_database
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

        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        #MainMenu,
        .stDeployButton {
            display: none;
            visibility: hidden;
        }
        
        /* Header styling with gradient */
        .header-container {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.96) 0%, rgba(30, 41, 59, 0.98) 100%);
            max-width: 980px;
            padding: 1.7rem 1.9rem;
            border-radius: 18px;
            margin: 0 auto 1.4rem auto;
            text-align: center;
            border: 1px solid rgba(148, 163, 184, 0.16);
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.18);
        }
        
        .header-container h1 {
            color: #f8fafc !important;
            font-size: 2.35rem;
            margin: 0;
            font-weight: 750;
            letter-spacing: -0.5px;
            text-shadow: 0 1px 1px rgba(15, 23, 42, 0.35);
        }
        
        .header-container p {
            color: #cbd5e1 !important;
            font-size: 0.98rem;
            margin: 0.5rem 0 0 0;
            font-weight: 400;
            text-shadow: 0 1px 1px rgba(15, 23, 42, 0.2);
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
            font-size: 0.9rem;
            font-weight: 700;
            color: var(--text-secondary);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-color) 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.7rem 1.5rem;
            font-size: 1rem;
            transition: all 0.2s ease;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.12);
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 22px rgba(15, 23, 42, 0.14);
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
            background-color: #ffffff;
            color: var(--text-primary);
            border: 1px solid var(--border-color) !important;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            font-size: 1rem;
            box-shadow: none;
        }
        
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px rgba(29, 78, 216, 0.08);
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

        /* Professional dark-theme overrides */
        :root {
            --primary-color: #1d4ed8;
            --primary-dark: #0f172a;
            --primary-light: #3b82f6;
            --accent-color: #0f766e;
            --success-color: #15803d;
            --error-color: #b91c1c;
            --bg-color: #0b0f19;
            --card-bg: #111827;
            --card-soft: #1f2937;
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --border-color: #334155;
        }

        .main {
            background-color: var(--bg-color);
            color: var(--text-primary);
        }

        .card,
        .source-card,
        .answer-section {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.25);
        }

        .header-container {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.98) 0%, rgba(30, 41, 59, 1) 100%);
            border-radius: 18px;
            margin-bottom: 1.5rem;
            padding: 2rem;
            border: 1px solid rgba(148, 163, 184, 0.16);
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.18);
        }

        .header-container h1 {
            color: #f8fafc;
            font-size: 2.4rem;
            font-weight: 750;
            letter-spacing: -0.04em;
        }

        .header-container p {
            color: var(--text-secondary);
            font-size: 1rem;
            font-weight: 400;
        }

        .section-header {
            color: var(--text-primary);
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .stButton > button {
            background: var(--primary-dark);
            border-radius: 10px;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.12);
        }

        .stButton > button:hover {
            box-shadow: 0 12px 22px rgba(15, 23, 42, 0.14);
            transform: translateY(-1px);
        }

        .stTextInput input, .stTextArea textarea {
            background: transparent !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px;
            box-shadow: none;
            color: var(--text-primary);
        }

        .stTextInput [data-baseweb="base-input"],
        .stTextArea [data-baseweb="base-input"] {
            background-color: rgba(15, 23, 42, 0.94) !important;
            border-radius: 10px;
        }

        .stTextInput [data-baseweb="base-input"] > div,
        .stTextArea [data-baseweb="base-input"] > div {
            background-color: rgba(15, 23, 42, 0.94) !important;
        }

        .stTextInput [data-baseweb="base-input"] input,
        .stTextArea [data-baseweb="base-input"] textarea {
            background: transparent !important;
            color: #f8fafc !important;
            -webkit-text-fill-color: #f8fafc;
        }

        .auth-panel .stTextInput input,
        .auth-panel .stTextArea textarea {
            background-color: rgba(15, 23, 42, 0.94);
            border-color: rgba(148, 163, 184, 0.2) !important;
            color: #f8fafc;
        }

        .auth-panel [data-baseweb="base-input"] input:-webkit-autofill,
        .auth-panel [data-baseweb="base-input"] input:-webkit-autofill:hover,
        .auth-panel [data-baseweb="base-input"] input:-webkit-autofill:focus,
        .stTextInput [data-baseweb="base-input"] input:-webkit-autofill,
        .stTextInput [data-baseweb="base-input"] input:-webkit-autofill:hover,
        .stTextInput [data-baseweb="base-input"] input:-webkit-autofill:focus {
            -webkit-box-shadow: 0 0 0 1000px rgba(15, 23, 42, 0.94) inset;
            box-shadow: 0 0 0 1000px rgba(15, 23, 42, 0.94) inset;
            -webkit-text-fill-color: #f8fafc;
            caret-color: #f8fafc;
        }

        .auth-panel .stTextInput input::placeholder,
        .auth-panel .stTextArea textarea::placeholder {
            color: #94a3b8;
        }

        .auth-panel label,
        .auth-panel .stMarkdown,
        .auth-panel p {
            color: #e2e8f0;
        }

        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px rgba(29, 78, 216, 0.08);
        }

        .auth-credentials {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.92) 0%, rgba(30, 41, 59, 0.96) 100%);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 14px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
            margin: 0.5rem 0 1rem 0;
            padding: 0.95rem 1rem;
        }

        .auth-credentials-title {
            color: #f8fafc;
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin-bottom: 0.8rem;
            text-transform: uppercase;
        }

        .auth-credentials-grid {
            display: grid;
            gap: 0.65rem;
        }

        .auth-credentials-item {
            background: rgba(30, 41, 59, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 10px;
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.5;
            padding: 0.7rem 0.85rem;
        }

        .auth-credentials-item strong,
        .auth-credentials-item span {
            color: #f8fafc;
        }

        .auth-layout {
            display: grid;
            grid-template-columns: minmax(0, 1.08fr) minmax(300px, 0.92fr);
            gap: 1rem;
            align-items: start;
            margin-top: 0.5rem;
        }

        .auth-panel {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.88) 0%, rgba(17, 24, 39, 0.96) 100%);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.18);
            backdrop-filter: blur(12px);
        }

        .auth-panel .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .auth-panel .stTabs [data-baseweb="tab"] {
            background: rgba(30, 41, 59, 0.9);
            color: #cbd5e1;
            border-radius: 999px;
            padding: 0.55rem 0.95rem;
        }

        .auth-panel .stTabs [aria-selected="true"] {
            background: #f8fafc;
            color: #0f172a;
        }

        .auth-note {
            border-left: 3px solid var(--primary-color);
            color: var(--text-secondary);
            font-size: 0.94rem;
            line-height: 1.55;
            margin-bottom: 0.9rem;
            padding-left: 0.9rem;
        }

        .demo-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-bottom: 0.75rem;
        }

        .demo-chip {
            background: rgba(30, 41, 59, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 999px;
            color: #cbd5e1;
            font-size: 0.8rem;
            padding: 0.35rem 0.7rem;
        }

        .stInfo, .stWarning, .stSuccess {
            border-radius: 10px;
            padding: 1rem;
        }

        .stInfo {
            background-color: rgba(29, 78, 216, 0.06);
            border: 1px solid rgba(29, 78, 216, 0.18);
        }

        .stWarning {
            background-color: rgba(15, 118, 110, 0.06);
            border: 1px solid rgba(15, 118, 110, 0.18);
        }

        .stSuccess {
            background-color: rgba(21, 128, 61, 0.06);
            border: 1px solid rgba(21, 128, 61, 0.18);
        }

        .source-card {
            background: linear-gradient(135deg, rgba(29, 78, 216, 0.04) 0%, rgba(15, 118, 110, 0.03) 100%);
            border: 1px solid rgba(29, 78, 216, 0.14);
            border-radius: 12px;
            padding: 1.1rem;
            margin-bottom: 0.9rem;
        }

        .citation-badge {
            background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-color) 100%);
        }

        .answer-section {
            background: linear-gradient(135deg, rgba(29, 78, 216, 0.03) 0%, rgba(15, 118, 110, 0.03) 100%);
            border: 1px solid rgba(29, 78, 216, 0.14);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1.25rem;
        }

        .metric-item {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 0.75rem 1rem;
        }

        .stFileUploader {
            background-color: rgba(15, 23, 42, 0.02);
            border-radius: 14px;
            border: 2px dashed var(--border-color);
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
    # Clear Hydra global instance to prevent conflicts between Streamlit reruns
    try:
        GlobalHydra.instance().clear()
    except Exception:
        pass

    cfg = load_config()

    try:
        ensure_auth_storage(cfg)
    except Exception as exc:
        st.set_page_config(page_title=cfg.app.title, layout="wide", initial_sidebar_state="expanded")
        st.error(f"❌ Unable to initialize PostgreSQL auth storage: {exc}")
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