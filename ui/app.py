"""Streamlit chat client for the grounded RAG assistant."""

import json
import os
from collections.abc import Iterator
from html import escape

import requests
import streamlit as st
from dotenv import load_dotenv

from rag_eval_suite.utils.get_root import get_project_root

load_dotenv(dotenv_path=get_project_root() / ".env")

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
QUICK_PROMPTS = (
    "Summarize the main ideas in the lectures",
    "Explain the most important concepts step by step",
    "What are the key trade-offs discussed in the material?",
)

st.set_page_config(
    page_title="RAG Eval Suite",
    page_icon=":material/auto_awesome:",
    layout="centered",
    initial_sidebar_state="expanded",
)


def initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    if "stream_sources" not in st.session_state:
        st.session_state.stream_sources = []


def apply_theme(dark_mode: bool) -> None:
    """Apply the app's high-contrast chat surfaces and responsive spacing."""
    palette = (
        {
            "page": "#0b1020",
            "surface": "#121a2b",
            "surface_alt": "#1a2438",
            "text": "#f4f7fb",
            "muted": "#a9b6ca",
            "border": "#2b3a54",
            "accent": "#a78bfa",
            "accent_strong": "#8b5cf6",
            "user": "#28204f",
            "code": "#0b1324",
        }
        if dark_mode
        else {
            "page": "#f7f8fc",
            "surface": "#ffffff",
            "surface_alt": "#f1f3f9",
            "text": "#182033",
            "muted": "#5c687d",
            "border": "#dfe4ee",
            "accent": "#5b4de8",
            "accent_strong": "#4738c7",
            "user": "#eeebff",
            "code": "#f4f5f9",
        }
    )

    st.html(
        f"""
        <style>
        :root {{
            --rag-page: {palette['page']};
            --rag-surface: {palette['surface']};
            --rag-surface-alt: {palette['surface_alt']};
            --rag-text: {palette['text']};
            --rag-muted: {palette['muted']};
            --rag-border: {palette['border']};
            --rag-accent: {palette['accent']};
            --rag-accent-strong: {palette['accent_strong']};
            --rag-user: {palette['user']};
            --rag-code: {palette['code']};
        }}
        .stApp {{ background: var(--rag-page) !important; color: var(--rag-text) !important; }}
        [data-testid="stHeader"] {{ background: transparent !important; }}
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div {{
            background: var(--rag-surface) !important;
            border-right: 1px solid var(--rag-border) !important;
        }}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {{
            color: var(--rag-text) !important;
        }}
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{
            color: var(--rag-muted) !important;
        }}
        section.main > div.block-container {{
            max-width: 820px;
            padding: 2.4rem 1.25rem 7.5rem;
        }}
        .rag-brand {{ align-items: center; display: flex; gap: .65rem; margin-bottom: 3rem; }}
        .rag-mark {{
            align-items: center;
            background: linear-gradient(135deg, var(--rag-accent), var(--rag-accent-strong));
            border-radius: .8rem;
            color: #ffffff;
            display: flex;
            font-size: 1rem;
            font-weight: 800;
            height: 2.15rem;
            justify-content: center;
            width: 2.15rem;
        }}
        .rag-brand-name {{ color: var(--rag-text); font-size: .96rem; font-weight: 750; letter-spacing: -.02em; }}
        .rag-kicker {{
            color: var(--rag-accent);
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .15em;
            margin-bottom: .6rem;
            text-transform: uppercase;
        }}
        .rag-title {{
            color: var(--rag-text);
            font-size: clamp(2rem, 6vw, 3.4rem);
            font-weight: 800;
            letter-spacing: -.065em;
            line-height: 1;
            margin: 0;
        }}
        .rag-subtitle {{ color: var(--rag-muted); font-size: 1rem; line-height: 1.6; margin: .9rem 0 2rem; max-width: 650px; }}
        .rag-connected {{ color: var(--rag-muted); font-size: .76rem; line-height: 1.45; overflow-wrap: anywhere; }}
        .rag-connected strong {{ color: var(--rag-text); }}
        [data-testid="stChatMessage"] {{
            background: var(--rag-surface) !important;
            border: 1px solid var(--rag-border) !important;
            border-radius: 1.15rem !important;
            margin: .85rem 0 !important;
            padding: .95rem 1rem !important;
        }}
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
            background: var(--rag-user) !important;
            border-color: transparent !important;
        }}
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{ color: var(--rag-text) !important; line-height: 1.65; }}
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {{ color: var(--rag-text) !important; }}
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] a {{ color: var(--rag-accent) !important; }}
        [data-testid="stChatMessage"] pre {{ background: var(--rag-code) !important; border: 1px solid var(--rag-border) !important; border-radius: .7rem !important; }}
        [data-testid="stChatMessage"] pre code {{ color: var(--rag-text) !important; }}
        [data-testid="stChatInput"] > div {{
            background: var(--rag-surface) !important;
            border: 1px solid var(--rag-border) !important;
            border-radius: 1rem !important;
            box-shadow: 0 10px 28px rgba(20, 28, 48, .10);
        }}
        [data-testid="stChatInput"] textarea {{ color: var(--rag-text) !important; -webkit-text-fill-color: var(--rag-text) !important; }}
        [data-testid="stChatInput"] textarea::placeholder {{ color: var(--rag-muted) !important; -webkit-text-fill-color: var(--rag-muted) !important; }}
        [data-testid="stChatInput"] button {{ color: var(--rag-accent) !important; }}
        [data-testid="stExpander"] {{ background: var(--rag-surface) !important; border: 1px solid var(--rag-border) !important; border-radius: .85rem !important; }}
        [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color: var(--rag-text) !important; }}
        [data-testid="stPills"] button {{ background: var(--rag-surface) !important; border: 1px solid var(--rag-border) !important; color: var(--rag-text) !important; }}
        [data-testid="stPills"] button:hover, [data-testid="stPills"] button[aria-checked="true"] {{ background: var(--rag-accent) !important; border-color: var(--rag-accent) !important; color: #ffffff !important; }}
        div.stButton > button {{ background: var(--rag-surface) !important; border: 1px solid var(--rag-border) !important; color: var(--rag-text) !important; }}
        div.stButton > button:hover {{ border-color: var(--rag-accent) !important; color: var(--rag-accent) !important; }}
        </style>
        """
    )


def stream_answer(question: str, retrieval_k: int = 5) -> Iterator[str]:
    """Yield answer tokens from the API's server-sent event stream."""
    with requests.post(
        f"{API_URL.rstrip('/')}/ask/stream",
        json={"question": question, "retrieval_k": retrieval_k},
        stream=True,
        timeout=180,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            payload = raw_line.removeprefix("data: ")
            if payload == "[DONE]":
                return
            event = json.loads(payload)
            if event.get("type") == "sources":
                st.session_state.stream_sources = event.get("sources", [])
            elif event.get("type") == "error":
                raise RuntimeError(event.get("detail", "The RAG service failed."))
            elif event.get("type") == "token":
                yield str(event.get("content", ""))


def render_sources(sources: list[dict[str, object]]) -> None:
    """Render source metadata with native Streamlit text elements."""
    with st.expander(f":material/source: {len(sources)} sources", expanded=False):
        for index, source in enumerate(sources, start=1):
            location = escape(str(source.get("source", "Unknown source")))
            if source.get("page"):
                location += f" · page {escape(str(source['page']))}"
            st.markdown(f"**{index}. {location}**", unsafe_allow_html=True)
            st.caption(str(source.get("excerpt", "")).strip())


initialize_state()
apply_theme(st.session_state.dark_mode)

with st.sidebar:
    st.markdown("## :material/tune: Settings")
    st.toggle("Dark mode", key="dark_mode")
    retrieval_k = st.slider("Sources to retrieve", 1, 10, 5)
    st.caption("The assistant grounds answers in your indexed Zilliz collection.")
    st.space("small")
    if st.button(":material/delete_sweep: Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.space("small")
    st.markdown(
        f'<div class="rag-connected"><strong>API</strong><br>{escape(API_URL)}</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="rag-brand"><div class="rag-mark">✦</div>'
    '<div class="rag-brand-name">RAG Eval Suite</div></div>'
    '<div class="rag-kicker">Grounded intelligence</div>'
    '<div class="rag-title">Ask your knowledge base.</div>'
    '<div class="rag-subtitle">A focused research assistant that retrieves relevant context, '
    "answers clearly, and shows where the answer came from.</div>",
    unsafe_allow_html=True,
)

suggestion_slot = st.empty()
if not st.session_state.messages:
    with st.container(border=True):
        st.markdown("**Start with a question**")
        st.caption("Try one of these prompts or write your own below.")
        prompt_choice = st.pills(
            "Quick prompts",
            QUICK_PROMPTS,
            selection_mode="single",
            label_visibility="collapsed",
        )
else:
    prompt_choice = None

for message in st.session_state.messages:
    role = message["role"]
    with st.chat_message(
        role,
        avatar=(":material/person:" if role == "user" else ":material/auto_awesome:"),
    ):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

question = prompt_choice or st.chat_input("Ask a question about your documents…")
if question:
    suggestion_slot.empty()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        st.session_state.stream_sources = []
        status = st.status("Searching your knowledge base…", expanded=False)
        try:
            answer = st.write_stream(stream_answer(question, retrieval_k))
            sources = st.session_state.get("stream_sources", [])
            status.update(label="Answer ready", state="complete")
            if sources:
                render_sources(sources)
        except (requests.RequestException, RuntimeError) as exc:
            answer = f"I couldn’t reach the RAG service. {exc}"
            status.update(label="Could not reach the RAG service", state="error")
            st.error(answer, icon=":material/error:")
            sources = []

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
