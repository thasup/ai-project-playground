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
import json
import time
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


def build_context_l2(
    system_prompt: str,
    full_history: list[dict],
    summary: str,
    recent_window: int,
) -> list:
    """L2: Summary + recent window. The summary covers older turns,
    recent turns are kept verbatim.
    """
    messages = [SystemMessage(content=system_prompt)]

    # Inject summary of older turns
    if summary:
        summary_block = (
            f"[CONVERSATION SUMMARY — compressed state of older turns]\n{summary}\n"
            f"[END SUMMARY]"
        )
        messages.append(SystemMessage(content=summary_block))

    # Keep recent turns verbatim
    recent = full_history[-recent_window:] if len(full_history) > recent_window else full_history
    for turn in recent:
        messages.append(HumanMessage(content=turn["human"]))
        if turn.get("ai"):
            messages.append(AIMessage(content=turn["ai"]))
    return messages


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
        else:
            st.info("L0 has no tunable parameters — everything is sent!")

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
    st.markdown("##### 💬 Chat")

    # Display chat history
    for msg in st.session_state.chat_display:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Try: 'My name is Alice' then ask 'What is my name?'", key="chat_input_main")

    if user_input:
        # Display user message
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

        # Get AI response
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
        st.markdown("##### 📋 Current Summary Digest")
        st.info(st.session_state.summary)

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
