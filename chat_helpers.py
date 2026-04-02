"""
通用聊天 UI 辅助函数。

在 RAG 与 Agent 页面间复用渲染与导出逻辑，避免重复代码。
"""

from __future__ import annotations

from typing import Any, TypedDict, cast

import streamlit as st
from langchain_core.documents import Document


class ChatMessage(TypedDict, total=False):
    """保存在 Streamlit 会话状态中的聊天消息载荷。"""

    role: str
    content: str
    source_docs: list[Document]


def init_session_key(key: str, default: Any) -> None:
    """如果会话状态中不存在该键，则进行初始化。"""
    if key not in st.session_state:
        st.session_state[key] = default


def render_chat_history(
    messages: list[ChatMessage], *, show_sources: bool = False
) -> None:
    """
    渲染聊天消息列表（包含 ``role`` / ``content`` 字段）。

    当 *show_sources* 为 ``True`` 时，带有 ``source_docs`` 的助手消息
    会显示可展开的参考来源区块。
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
    """渲染可折叠的参考来源区块，展示文档片段。"""
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
    将聊天消息序列化为可下载的 Markdown 字符串。

    参数：
        messages: 聊天消息列表。
        title: 文档标题。
        include_sources: 是否附加参考来源片段。
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
    在聊天页渲染“清空对话”与“导出”按钮。

    参数：
        messages_key: 会话状态中存放消息列表的键。
        export_title: 导出 Markdown 文档的标题。
        export_filename: 下载文件名。
        include_sources: 导出内容是否包含参考来源。
        extra_reset_keys: 清空时需要额外重置的会话键。
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
