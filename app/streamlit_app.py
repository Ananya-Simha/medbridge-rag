import time
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st

API_URL = "http://34.130.153.113:8000/answer"  # replace VM IP

st.set_page_config(page_title="MedBridge Chat", layout="wide")

# --- Global styling ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0f172a;  /* dark slate */
        color: #e5e7eb;
    }
    .chat-bubble-user {
        background-color: #1f2937;
        color: #e5e7eb;
        padding: 0.75rem 1rem;
        border-radius: 999px;
        max-width: 70%;
        margin-left: auto;
        margin-bottom: 0.5rem;
    }
    .chat-bubble-bot {
        background-color: #111827;
        color: #e5e7eb;
        padding: 1rem 1.25rem;
        border-radius: 1rem;
        max-width: 70%;
        margin-right: auto;
        margin-bottom: 0.5rem;
        border: 1px solid #374151;
    }
    .chat-input textarea {
        border-radius: 999px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar: logo + info ---
with st.sidebar:
    logo_path = Path(__file__).parent / "logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=100)
    st.markdown("### MedBridge")
    st.markdown(
        "A Retrieval-Augmented Generation (RAG) chatbot over the **MedQuAD** dataset. "
        "It retrieves authoritative medical content and rewrites it in "
        "patient-friendly language with citations."
    )
    st.markdown("**Note:** This interface is for educational use and is not medical advice.")

# --- Main hero section ---
if "name" not in st.session_state:
    st.session_state.name = ""

st.markdown("<br>", unsafe_allow_html=True)
name = st.text_input("What's your name?", value=st.session_state.name, label_visibility="collapsed")
if name:
    st.session_state.name = name

greeting = f"Hi there, {st.session_state.name}" if st.session_state.name else "Hi there"
st.markdown(
    f"<h1 style='text-align: center;'>{greeting}</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #9ca3af;'>💊 A MedBridge chatbot powered by Retrieval-Augmented Generation.</p>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# --- Chat history ---
if "history" not in st.session_state:
    st.session_state.history = []

# Display previous turns
for turn in st.session_state.history:
    # user
    st.markdown(f"<div class='chat-bubble-user'>{turn['user']}</div>", unsafe_allow_html=True)
    # bot
    st.markdown(f"<div class='chat-bubble-bot'>{turn['bot']}</div>", unsafe_allow_html=True)

# --- Input box at bottom ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("#### Your message", unsafe_allow_html=True)

with st.container():
    col_in, col_btn = st.columns([6, 1])
    with col_in:
        user_msg = st.text_input(
            "",
            key="user_input",
            placeholder="Ask a medical question... (e.g., What causes glaucoma?)",
        )
    with col_btn:
        send = st.button("➤", use_container_width=True)

if send and user_msg.strip():
    question = user_msg.strip()
    with st.spinner("Retrieving evidence and generating answer..."):
        start = time.time()
        try:
            resp = requests.post(API_URL, json={"question": question}, timeout=90)
            resp.raise_for_status()
            data: Dict[str, Any] = resp.json()
        except Exception as e:
            bot_text = f"Request failed: {e}"
            passages: List[Dict[str, Any]] = []
            latency = None
        else:
            latency = time.time() - start
            answer: str = data.get("answer", "")
            passages = data.get("passages", [])
            # Build a short sources summary
            if passages:
                src_lines = []
                for i, p in enumerate(passages, start=1):
                    url = p.get("url") or ""
                    topic = p.get("topic") or ""
                    if url.startswith("http"):
                        src_lines.append(f"[{i}] {topic} ({url})")
                if src_lines:
                    answer = answer + "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in src_lines)
            bot_text = answer or "Sorry, I could not generate an answer."

    st.session_state.history.append({"user": question, "bot": bot_text})
    st.experimental_rerun()
