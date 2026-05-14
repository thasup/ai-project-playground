"""
Context Window Demo — Demystifying LLM Memory Management
=========================================================
A companion demo app for the "Demystifying LLM Context Windows" knowledge
sharing session. Demonstrates how different memory management strategies
(complexity levels) affect what the LLM actually "sees" in its context window.

Levels implemented:
  L0 · Full Context (Naive)    — send every message every time
  L1 · Sliding Window (FIFO)   — keep only the last N message pairs
  L2 · Summarization           — compress older turns into a digest

Each mode shows a live "Context Window Inspector" so your audience can
see exactly what payload is sent to the model on each turn.
"""

import streamlit as st
import os
import io
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

# ── Constants ───────────────────────────────────────────────────────────────
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly AI assistant. Keep responses concise (2-3 sentences). "
    "When the user tells you personal facts (name, favorite color, etc.), "
    "acknowledge and remember them. This helps demonstrate memory behavior."
    "Must end response with 'meow!'"
)

LEVEL_META = {
    "L0 · Full Context (Naive)": {
        "tag": "L0",
        "color": "#ef4444",
        "icon": "🔴",
        "description": "Send the **entire** conversation history every time. Perfect recall, but hits the token wall fast.",
        "analogy": "⚛ Like passing the entire Redux store as props to every component",
        "pros": ["Perfect recall", "Zero complexity"],
        "cons": ["Hits hard token limits", "API cost scales linearly"],
    },
    "L1 · Sliding Window (FIFO)": {
        "tag": "L1",
        "color": "#f97316",
        "icon": "🟠",
        "description": "Keep only the **last N message pairs**. Older turns are permanently evicted — FIFO amnesia.",
        "analogy": "⚛ Like react-window — items scroll off the viewport and are gone",
        "pros": ["Bounded API cost", "No DB needed", "Fast"],
        "cons": ["Permanent amnesia for old turns", "Naive FIFO eviction"],
    },
    "L2 · Summarization": {
        "tag": "L2",
        "color": "#eab308",
        "icon": "🟡",
        "description": "Compress older turns into a **summary digest**, keeping recent turns verbatim. Lossy but preserves semantic gist.",
        "analogy": "⚛ Like useMemo() — expensive state collapsed into a cached digest",
        "pros": ["Preserves semantic gist", "Bounded context size"],
        "cons": ["Lossy compression", "Extra latency for summarization"],
    },
    "L3 · RAG": {
        "tag": "L3",
        "color": "#22c55e",
        "icon": "🟢",
        "description": "Don't send history/data blindly. **Search** for the most relevant 'chunks' from a large database and inject only them.",
        "analogy": "⚛ Like a Database Query — only fetch the specific rows needed for the current view",
        "pros": ["Infinite 'virtual' context", "Low cost per query", "Scales to millions of docs"],
        "cons": ["Retrieval latency", "Complexity (Vector DB, Embeddings)", "Risk of 'hallucination' if retrieval fails"],
    },
}

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Context Window Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Inspector panel */
    .ctx-inspector {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        max-height: 520px;
        overflow-y: auto;
    }
    .ctx-inspector .msg-system {
        color: #6d28d9;
        border-left: 3px solid #8b5cf6;
        padding: 6px 10px;
        margin-bottom: 6px;
        background: #f5f3ff;
        border-radius: 0 6px 6px 0;
    }
    .ctx-inspector .msg-human {
        color: #1d4ed8;
        border-left: 3px solid #3b82f6;
        padding: 6px 10px;
        margin-bottom: 6px;
        background: #eff6ff;
        border-radius: 0 6px 6px 0;
    }
    .ctx-inspector .msg-ai {
        color: #047857;
        border-left: 3px solid #10b981;
        padding: 6px 10px;
        margin-bottom: 6px;
        background: #ecfdf5;
        border-radius: 0 6px 6px 0;
    }
    .ctx-inspector .msg-summary {
        color: #b45309;
        border-left: 3px solid #f59e0b;
        padding: 6px 10px;
        margin-bottom: 6px;
        background: #fffbeb;
        border-radius: 0 6px 6px 0;
    }
    .ctx-inspector .msg-label {
        font-size: 9px;
        letter-spacing: 2px;
        margin-bottom: 2px;
        opacity: 0.7;
    }
    .ctx-inspector .msg-content {
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* Level badge */
    .level-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.5px;
    }

    /* Stats bar */
    .stats-bar {
        display: flex;
        gap: 16px;
        padding: 10px 16px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .stats-bar .stat {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .stats-bar .stat-label {
        font-size: 9px;
        color: #64748b;
        letter-spacing: 1.5px;
    }
    .stats-bar .stat-value {
        font-size: 16px;
        font-weight: 700;
        color: #1e293b;
    }

    /* Evicted messages */
    .evicted-msg {
        color: #dc2626;
        border-left: 3px solid #ef4444;
        padding: 6px 10px;
        margin-bottom: 6px;
        background: #fef2f2;
        border-radius: 0 6px 6px 0;
        opacity: 0.6;
        text-decoration: line-through;
    }

    /* Chat container tweaks */
    [data-testid="stChatMessageContainer"] {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ── Helper functions ────────────────────────────────────────────────────────
def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list) -> int:
    """Estimate total tokens across a list of LangChain messages."""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.content)
    return total


def build_context_l0(
    system_prompt: str, full_history: list[dict],
) -> list:
    """L0: Full Context — send everything."""
    messages = [SystemMessage(content=system_prompt)]
    for turn in full_history:
        messages.append(HumanMessage(content=turn["human"]))
        if turn.get("ai"):
            messages.append(AIMessage(content=turn["ai"]))
    return messages


def build_context_l1(
    system_prompt: str, full_history: list[dict], window_size: int,
) -> tuple[list, list]:
    """L1: Sliding Window — keep only the last `window_size` pairs.
    Returns (context_messages, evicted_turns).
    """
    messages = [SystemMessage(content=system_prompt)]
    evicted = []
    kept = full_history[-window_size:] if len(full_history) > window_size else full_history
    evicted = full_history[: len(full_history) - len(kept)]
    for turn in kept:
        messages.append(HumanMessage(content=turn["human"]))
        if turn.get("ai"):
            messages.append(AIMessage(content=turn["ai"]))
    return messages, evicted


def build_context_l3(
    system_prompt: str,
    full_history: list[dict],
    retrieved_chunks: list[str],
) -> list:
    """L3: RAG — system prompt + retrieved knowledge + recent history."""
    messages = [SystemMessage(content=system_prompt)]

    # Inject retrieved knowledge
    if retrieved_chunks:
        knowledge_block = (
            "### EXTRACTED KNOWLEDGE (Contextually Relevant)\n"
            "The following facts were retrieved from the external database based on your query:\n\n"
            + "\n---\n".join(retrieved_chunks)
            + "\n\n### END KNOWLEDGE"
        )
        messages.append(SystemMessage(content=knowledge_block))

    # For L3 demo, we'll keep the last 2 turns of history to maintain conversational flow
    recent = full_history[-2:] if len(full_history) > 2 else full_history
    for turn in recent:
        messages.append(HumanMessage(content=turn["human"]))
        if turn.get("ai"):
            messages.append(AIMessage(content=turn["ai"]))
    return messages


def simple_retrieval(query: str, knowledge_base: list[str], top_k: int = 2) -> list[str]:
    """A simple 'semantic' retrieval mockup.
    In a real app this would use Vector Embeddings (Chroma, Pinecone, etc.).
    Here we use basic keyword overlap for pedagogical transparency.
    """
    if not query:
        return []

    query_words = set(query.lower().split())
    scores = []
    for chunk in knowledge_base:
        chunk_words = set(chunk.lower().replace(".", "").replace(",", "").split())
        overlap = len(query_words.intersection(chunk_words))
        scores.append((overlap, chunk))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scores[:top_k] if s[0] > 0]


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by character count.
    Uses sentence-boundary awareness: tries to break at newlines/periods.
    """
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to snap to a sentence boundary within the last 60 chars
        if end < len(text):
            snap_zone = text[max(start, end - 60):end]
            for sep in ["\n\n", ".\n", ". ", "\n"]:
                idx = snap_zone.rfind(sep)
                if idx != -1:
                    end = max(start, end - 60) + idx + len(sep)
                    break
        chunks.append(text[start:end].strip())
        start = end - overlap
        if start >= len(text):
            break
    return [c for c in chunks if c]


def parse_uploaded_file(uploaded_file) -> str:
    """Extract plain text from .txt, .md, or .pdf uploaded files."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except Exception as e:
            return f"[PDF parse error: {e}]"
    elif name.endswith((".md", ".txt")):
        return raw.decode("utf-8", errors="replace")
    else:
        return "[Unsupported file type]"


def generate_summary(llm, turns_to_summarize: list[dict], existing_summary: str = "") -> str:
    """Use the LLM to generate a running summary of conversation turns."""
    if not turns_to_summarize:
        return existing_summary

    turns_text = ""
    for t in turns_to_summarize:
        turns_text += f"Human: {t['human']}\n"
        if t.get("ai"):
            turns_text += f"AI: {t['ai']}\n"

    prompt = (
        "You are a conversation summarizer. Produce a concise summary that "
        "preserves all key facts, names, preferences, and decisions mentioned.\n\n"
    )
    if existing_summary:
        prompt += f"Previous summary:\n{existing_summary}\n\n"
    prompt += f"New conversation turns to incorporate:\n{turns_text}\n\n"
    prompt += "Updated summary (be concise, preserve facts):"

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return response.content


def render_inspector(context_messages: list, evicted_turns: list | None = None):
    """Render the Context Window Inspector panel showing what the LLM sees."""
    html_parts = []

    # Show evicted messages first (L1 only)
    if evicted_turns:
        for turn in evicted_turns:
            html_parts.append(
                f'<div class="evicted-msg">'
                f'<div class="msg-label">EVICTED · HUMAN</div>'
                f'<div class="msg-content">{turn["human"]}</div>'
                f"</div>"
            )
            if turn.get("ai"):
                html_parts.append(
                    f'<div class="evicted-msg">'
                    f'<div class="msg-label">EVICTED · AI</div>'
                    f'<div class="msg-content">{turn["ai"]}</div>'
                    f"</div>"
                )

    # Show active context
    for msg in context_messages:
        if isinstance(msg, SystemMessage):
            if "[CONVERSATION SUMMARY" in msg.content:
                css_class = "msg-summary"
                label = "SUMMARY DIGEST"
            elif "EXTRACTED KNOWLEDGE" in msg.content:
                css_class = "msg-ai"  # Use AI green for knowledge
                label = "RETRIEVED KNOWLEDGE (RAG)"
                # Style override for RAG chunk
                msg_content = msg.content
            else:
                css_class = "msg-system"
                label = "SYSTEM"
        elif isinstance(msg, HumanMessage):
            css_class = "msg-human"
            label = "HUMAN"
        elif isinstance(msg, AIMessage):
            css_class = "msg-ai"
            label = "AI"
        else:
            css_class = "msg-system"
            label = "UNKNOWN"

        content_escaped = msg.content.replace("<", "&lt;").replace(">", "&gt;")
        html_parts.append(
            f'<div class="{css_class}">'
            f'<div class="msg-label">{label}</div>'
            f'<div class="msg-content">{content_escaped}</div>'
            f"</div>"
        )

    html = '<div class="ctx-inspector">' + "".join(html_parts) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_stats(context_messages: list, full_history: list, evicted_count: int = 0):
    """Render token/message stats."""
    token_count = estimate_messages_tokens(context_messages)
    msg_count = len(context_messages)
    total_turns = len(full_history)

    st.markdown(
        f"""
    <div class="stats-bar">
        <div class="stat">
            <div class="stat-label">CONTEXT MESSAGES</div>
            <div class="stat-value">{msg_count}</div>
        </div>
        <div class="stat">
            <div class="stat-label">EST. TOKENS</div>
            <div class="stat-value">~{token_count:,}</div>
        </div>
        <div class="stat">
            <div class="stat-label">TOTAL TURNS</div>
            <div class="stat-value">{total_turns}</div>
        </div>
        <div class="stat">
            <div class="stat-label">EVICTED</div>
            <div class="stat-value" style="color: {'#dc2626' if evicted_count > 0 else '#059669'}">{evicted_count}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # API Key
    with st.expander("🔑 API Settings", expanded=True):
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=os.getenv("OPEN_ROUTER_API_KEY", ""),
            key="input_api_key",
            help="Get your key at openrouter.ai",
        )
        model_name = st.text_input(
            "Model",
            value=DEFAULT_MODEL,
            key="input_model_name",
            help="OpenRouter model identifier",
        )

    # Mode Selection
    with st.expander("🧠 Memory Level", expanded=True):
        selected_mode = st.radio(
            "Select Complexity Level",
            list(LEVEL_META.keys()),
            index=0,
            key="radio_memory_level",
            help="Each level demonstrates a different memory management strategy",
        )
        meta = LEVEL_META[selected_mode]

        # Level-specific description card
        st.markdown(
            f"""
        <div style="
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 3px solid {meta['color']};
            border-radius: 8px;
            padding: 12px 14px;
            margin-top: 8px;
        ">
            <div style="font-size: 11px; color: {meta['color']}; letter-spacing: 1.5px; margin-bottom: 6px; font-weight: 600;">
                {meta['tag']}
            </div>
            <div style="font-size: 12px; color: #475569; line-height: 1.5;">
                {meta['description']}
            </div>
            <div style="font-size: 11px; color: #92400e; margin-top: 8px; line-height: 1.4;">
                {meta['analogy']}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Level-specific settings
    with st.expander("🎛️ Level Settings", expanded=True):
        if "L1" in selected_mode:
            window_size = st.slider(
                "Window Size (message pairs)",
                min_value=1,
                max_value=20,
                value=4,
                key="slider_window_size",
                help="Number of recent user/AI message pairs to keep",
            )
        elif "L2" in selected_mode:
            recent_window = st.slider(
                "Recent Window (pairs kept verbatim)",
                min_value=1,
                max_value=10,
                value=3,
                key="slider_recent_window",
                help="Number of recent pairs kept verbatim (older ones get summarized)",
            )
            summarize_threshold = st.slider(
                "Summarize after N turns",
                min_value=2,
                max_value=10,
                value=3,
                key="slider_summarize_threshold",
                help="Trigger summarization when unsummarized turns exceed this",
            )
        elif "L3" in selected_mode:
            top_k = st.slider(
                "Top-K Retrieval",
                min_value=1,
                max_value=5,
                value=2,
                key="slider_top_k",
                help="Number of chunks to retrieve from Knowledge Base",
            )
            st.info("Edit the Knowledge Base in the section below!")
        else:
            st.info("L0 has no tunable parameters — everything is sent!")

    # Knowledge Base (for L3)
    if "L3" in selected_mode:
        with st.expander("📚 Knowledge Base (RAG Source)", expanded=True):
            tab_manual, tab_upload = st.tabs(["✏️ Manual", "📁 Upload File"])

            with tab_manual:
                kb_text = st.text_area(
                    "Enter facts (one per line)",
                    value="\n".join(st.session_state.knowledge_base),
                    height=160,
                    key="textarea_kb",
                )
                st.session_state.knowledge_base = [
                    line.strip() for line in kb_text.split("\n") if line.strip()
                ]

            with tab_upload:
                chunk_size = st.slider(
                    "Chunk size (chars)",
                    min_value=100, max_value=1000, value=300, step=50,
                    key="slider_chunk_size",
                    help="How many characters per chunk",
                )
                overlap_size = st.slider(
                    "Chunk overlap (chars)",
                    min_value=0, max_value=200, value=50, step=10,
                    key="slider_overlap_size",
                    help="How many characters overlap between consecutive chunks",
                )

                uploaded_files = st.file_uploader(
                    "Upload .txt, .md, or .pdf",
                    type=["txt", "md", "pdf"],
                    accept_multiple_files=True,
                    key="file_uploader_rag",
                    label_visibility="collapsed",
                )

                if uploaded_files:
                    for uf in uploaded_files:
                        if uf.name not in st.session_state.file_chunks:
                            with st.spinner(f"Parsing {uf.name}..."):
                                text = parse_uploaded_file(uf)
                                chunks = chunk_text(text, chunk_size, overlap_size)
                                st.session_state.file_chunks[uf.name] = chunks
                            st.success(f"✅ {uf.name} — {len(chunks)} chunks added")

                # Show loaded files
                if st.session_state.file_chunks:
                    st.markdown("**Loaded files:**")
                    for fname, chunks in list(st.session_state.file_chunks.items()):
                        col_a, col_b = st.columns([4, 1])
                        col_a.markdown(
                            f"📄 `{fname}` — **{len(chunks)}** chunks",
                        )
                        if col_b.button("✕", key=f"btn_remove_{fname}"):
                            del st.session_state.file_chunks[fname]
                            st.rerun()

    # System prompt
    with st.expander("📝 System Prompt", expanded=False):
        system_prompt = st.text_area(
            "System Prompt",
            value=DEFAULT_SYSTEM_PROMPT,
            height=120,
            key="textarea_system_prompt",
        )

    # Actions
    st.divider()
    if st.button("🗑️ Clear Conversation", key="btn_clear_conversation", use_container_width=True):
        st.session_state.full_history = []
        st.session_state.summary = ""
        st.session_state.chat_display = []
        st.rerun()

# ── Validate API Key ────────────────────────────────────────────────────────
if not api_key:
    st.warning("🔑 Please enter your OpenRouter API key in the sidebar to begin.")
    st.stop()

# ── Initialize LLM ─────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model=model_name,
    openai_api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
)

# ── Session State ───────────────────────────────────────────────────────────
if "full_history" not in st.session_state:
    st.session_state.full_history = []  # list of {"human": str, "ai": str}
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []  # for UI display
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = [
        "The project code name is 'Project Antigravity'.",
        "The launch date is set for June 15, 2026.",
        "The lead engineer is Dr. Sarah Chen.",
        "The project headquarters is located in Bangkok, Thailand.",
        "Project Antigravity uses a state-of-the-art context management engine.",
    ]
if "retrieved_chunks" not in st.session_state:
    st.session_state.retrieved_chunks = []
# file_chunks: dict[filename -> list[chunk_str]]
if "file_chunks" not in st.session_state:
    st.session_state.file_chunks = {}

# ── Main Layout ─────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
    <span style="font-size: 28px;">🧠</span>
    <div>
        <div style="font-size: 24px; font-weight: 800; color: #1e293b; letter-spacing: -0.5px;">
            Context Window Demo
        </div>
        <div style="font-size: 12px; color: #94a3b8; letter-spacing: 1px;">
            DEMYSTIFYING LLM MEMORY MANAGEMENT
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Active mode indicator
st.markdown(
    f"""
<div style="
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 6px 14px;
    border-radius: 6px;
    margin-bottom: 16px;
">
    <div style="width: 8px; height: 8px; border-radius: 50%; background: {meta['color']};"></div>
    <span style="font-size: 12px; color: {meta['color']}; letter-spacing: 1px; font-weight: 600;">
        {meta['tag']} ACTIVE
    </span>
    <span style="font-size: 12px; color: #64748b;">— {selected_mode.split('·')[1].strip()}</span>
</div>
""",
    unsafe_allow_html=True,
)

# Two-column layout: Chat | Inspector
chat_col, inspector_col = st.columns([3, 2])

with chat_col:
    st.markdown("##### 💬 Conversation")

    # Scrollable chat container
    chat_container = st.container(height=520)

    # Display chat history inside container
    with chat_container:
        for msg in st.session_state.chat_display:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat input (Outside container to stay sticky at bottom)
    user_input = st.chat_input("Try: 'Who is the lead engineer?'", key="chat_input_main")

    if user_input:
        # Display user message in container immediately
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        # Add to full history (ai will be filled after response)
        current_turn = {"human": user_input, "ai": ""}
        st.session_state.full_history.append(current_turn)

        # Build context based on selected mode
        if "L0" in selected_mode:
            context_messages = build_context_l0(system_prompt, st.session_state.full_history)
        elif "L1" in selected_mode:
            context_messages, evicted = build_context_l1(
                system_prompt, st.session_state.full_history, window_size
            )
        elif "L2" in selected_mode:
            # Check if we need to summarize
            total_turns = len(st.session_state.full_history)
            if total_turns > recent_window + summarize_threshold:
                turns_to_summarize = st.session_state.full_history[
                    : total_turns - recent_window
                ]
                with st.spinner("📝 Generating summary digest..."):
                    st.session_state.summary = generate_summary(
                        llm, turns_to_summarize, st.session_state.summary
                    )

            context_messages = build_context_l2(
                system_prompt,
                st.session_state.full_history,
                st.session_state.summary,
                recent_window,
            )
        elif "L3" in selected_mode:
            # Combine manual KB + all file chunks into one pool for retrieval
            all_chunks = list(st.session_state.knowledge_base)
            for file_chunk_list in st.session_state.file_chunks.values():
                all_chunks.extend(file_chunk_list)
            # Perform retrieval
            with st.spinner("🔍 Searching Knowledge Base..."):
                st.session_state.retrieved_chunks = simple_retrieval(
                    user_input, all_chunks, top_k
                )
            context_messages = build_context_l3(
                system_prompt, st.session_state.full_history, st.session_state.retrieved_chunks
            )

        # Get AI response
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = llm.invoke(context_messages)
                        ai_text = response.content
                        st.markdown(ai_text)
                    except Exception as e:
                        ai_text = f"Error: {str(e)}"
                        st.error(ai_text)

        # Update history with AI response
        st.session_state.full_history[-1]["ai"] = ai_text

        # Update display history
        st.session_state.chat_display.append({"role": "user", "content": user_input})
        st.session_state.chat_display.append({"role": "assistant", "content": ai_text})

        st.rerun()

with inspector_col:
    st.markdown("##### 🔍 Context Window Inspector")
    st.caption("What the LLM actually receives in its API payload")

    # Build current context for display
    evicted_turns = []
    if st.session_state.full_history:
        if "L0" in selected_mode:
            display_context = build_context_l0(system_prompt, st.session_state.full_history)
        elif "L1" in selected_mode:
            display_context, evicted_turns = build_context_l1(
                system_prompt, st.session_state.full_history, window_size
            )
        elif "L2" in selected_mode:
            display_context = build_context_l2(
                system_prompt,
                st.session_state.full_history,
                st.session_state.summary,
                recent_window,
            )
        elif "L3" in selected_mode:
            display_context = build_context_l3(
                system_prompt, st.session_state.full_history, st.session_state.retrieved_chunks
            )
    else:
        display_context = [SystemMessage(content=system_prompt)]

    # Render stats
    render_stats(
        display_context,
        st.session_state.full_history,
        evicted_count=len(evicted_turns),
    )

    # Render inspector
    render_inspector(display_context, evicted_turns if "L1" in selected_mode else None)

    # L2 summary display
    if "L2" in selected_mode and st.session_state.summary:
        st.markdown("##### \U0001f4cb Current Summary Digest")
        st.info(st.session_state.summary)

    # L3 retrieved chunk display
    if "L3" in selected_mode:
        total_pool = len(st.session_state.knowledge_base) + sum(
            len(v) for v in st.session_state.file_chunks.values()
        )
        n_retrieved = len(st.session_state.retrieved_chunks)
        st.markdown(
            f"##### \U0001f4da RAG Search Pool: **{total_pool}** chunks"
            f" &nbsp;&mdash;&nbsp; Retrieved: "
            f"<span style='color:#22c55e;font-weight:700;'>{n_retrieved}</span>",
            unsafe_allow_html=True,
        )
        if st.session_state.retrieved_chunks:
            for i, chunk in enumerate(st.session_state.retrieved_chunks, 1):
                with st.expander(f"Chunk {i} — {chunk[:60].strip()}...", expanded=True):
                    st.markdown(chunk)

    # Raw JSON toggle
    with st.expander("📄 Raw API Payload (JSON)", expanded=False):
        payload = []
        for msg in display_context:
            role = "system"
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            payload.append({"role": role, "content": msg.content})
        st.json(payload)
