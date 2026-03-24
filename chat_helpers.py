"""
Shared chat UI helpers.

Eliminates duplicated Streamlit chat rendering / export logic
across RAG and Agent tabs.
"""

from __future__ import annotations

from typing import Any, TypedDict, cast

import streamlit as st
from langchain_core.documents import Document


class ChatMessage(TypedDict, total=False):
    """Chat message payload stored in Streamlit session state."""

    role: str
    content: str
    source_docs: list[Document]


def init_session_key(key: str, default: Any) -> None:
    """Initialise a session-state key if it does not yet exist."""
    if key not in st.session_state:
        st.session_state[key] = default


def render_chat_history(
    messages: list[ChatMessage], *, show_sources: bool = False
) -> None:
    """
    Render a list of chat messages (``role`` / ``content`` dicts).

    If *show_sources* is ``True``, assistant messages with a
    ``source_docs`` key will render an expandable reference section.
    """
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if (
                show_sources
                and msg["role"] == "assistant"
                and "source_docs" in msg
            ):
                _render_source_expander(msg["source_docs"])


def _render_source_expander(source_docs: list[Document]) -> None:
    """Render a collapsible section listing source document excerpts."""
    with st.expander("📄 查看参考文档来源"):
        for idx, doc in enumerate(source_docs, 1):
            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page", "N/A")
            st.markdown(f"**片段 {idx}** — 来源：`{source}` | 页码：`{page}`")
            st.caption(doc.page_content)
            if idx < len(source_docs):
                st.divider()


def export_as_markdown(
    messages: list[ChatMessage],
    title: str,
    *,
    include_sources: bool = False,
) -> str:
    """
    Serialise chat messages to a downloadable Markdown string.

    Args:
        messages: The chat message list.
        title: Document title.
        include_sources: Whether to append reference-source sections.
    """
    lines = [f"# {title}\n"]
    for msg in messages:
        role_label = "🧑 用户" if msg["role"] == "user" else "🤖 助手"
        lines.append(f"## {role_label}\n\n{msg['content']}\n")

        if include_sources and msg["role"] == "assistant" and "source_docs" in msg:
            lines.append("### 参考来源\n")
            for idx, doc in enumerate(msg["source_docs"], 1):
                source = doc.metadata.get("source", "未知")
                page = doc.metadata.get("page", "N/A")
                lines.append(
                    f"- **片段 {idx}**（{source}，页码 {page}）\n"
                    f"  > {doc.page_content}\n"
                )

    return "\n".join(lines)


def render_chat_controls(
    messages_key: str,
    export_title: str,
    export_filename: str,
    *,
    include_sources: bool = False,
    extra_reset_keys: list[str] | None = None,
) -> None:
    """
    Render 'Clear history' and 'Export' buttons for a chat tab.

    Args:
        messages_key: Session-state key holding the message list.
        export_title: Title used in the exported Markdown.
        export_filename: Name for the download file.
        include_sources: Whether the export includes source references.
        extra_reset_keys: Additional session-state keys to reset on clear.
    """
    messages = cast(list[ChatMessage], st.session_state.get(messages_key, []))
    if not messages:
        return

    st.divider()
    col_clear, col_export = st.columns(2)

    with col_clear:
        if st.button("🗑️ 清空对话", key=f"{messages_key}_clear"):
            st.session_state[messages_key] = []
            for key in extra_reset_keys or []:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    with col_export:
        md_content = export_as_markdown(
            messages,
            export_title,
            include_sources=include_sources,
        )
        st.download_button(
            label="📥 导出对话记录",
            data=md_content,
            file_name=export_filename,
            mime="text/markdown",
            key=f"{messages_key}_export",
        )
