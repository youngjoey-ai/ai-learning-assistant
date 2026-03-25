import os
import shutil
from typing import cast

import streamlit as st
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from chains import build_rag_chain, run_agent_task
from chat_helpers import (
    ChatMessage,
    init_session_key,
    render_chat_controls,
    render_chat_history,
)
from config import AppConfig, resolve_api_key
from document_processing import DocumentProcessor
from vector_store_manager import VectorStoreManager


app_config = AppConfig()

st.set_page_config(
    page_title=app_config.page_title,
    page_icon=app_config.page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

API_KEY = resolve_api_key()


@st.cache_resource(show_spinner="正在初始化模型...")
def init_models(api_key: str):
    """Initialise embeddings and chat model once per app session."""
    os.environ["DASHSCOPE_API_KEY"] = api_key
    embeddings = DashScopeEmbeddings(model=app_config.model.embedding_model)
    llm = ChatOpenAI(
        api_key=SecretStr(api_key),
        base_url=app_config.model.llm_base_url,
        model=app_config.model.llm_model,
        temperature=app_config.model.temperature,
        max_completion_tokens=app_config.model.max_tokens,
    )
    return embeddings, llm


embeddings, llm = init_models(API_KEY)
document_processor = DocumentProcessor(app_config)
vector_store_manager = VectorStoreManager(
    embeddings,
    app_config.vector_store_path,
)
vector_store_manager.try_load_persisted()

init_session_key("rag_messages", [])
init_session_key("agent_messages", [])
init_session_key("upload_widget_version", 0)
init_session_key("upload_success_message", "")
init_session_key("clear_success_message", "")


def inject_responsive_styles() -> None:
    """Add responsive typography so headings stay tidy on smaller screens."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 3.5rem;
        }

        .app-page-title,
        .app-section-title {
            display: flex;
            align-items: flex-start;
            gap: 0.6rem;
            width: 100%;
        }

        .app-page-title {
            margin: 0 0 0.35rem;
        }

        .app-section-title {
            margin: 1.1rem 0 0.25rem;
        }

        .app-page-title__icon,
        .app-section-title__icon {
            flex: 0 0 auto;
            line-height: 1;
        }

        .app-page-title__icon {
            font-size: clamp(2rem, 8vw, 3rem);
        }

        .app-section-title__icon {
            font-size: clamp(1.75rem, 6vw, 2.4rem);
            margin-top: 0.12rem;
        }

        .app-page-title__text,
        .app-section-title__text {
            min-width: 0;
            font-weight: 800;
            line-height: 1.08;
            letter-spacing: -0.02em;
            text-wrap: balance;
        }

        .app-page-title__text {
            font-size: clamp(2.25rem, 8.2vw, 4rem);
        }

        .app-section-title__text {
            font-size: clamp(1.95rem, 6.3vw, 3rem);
        }

        @media (max-width: 768px) {
            .block-container {
                padding-top: calc(5rem + env(safe-area-inset-top));
                padding-left: 1rem;
                padding-right: 1rem;
            }

            button[data-baseweb="tab"] {
                padding-left: 0.55rem;
                padding-right: 0.55rem;
            }

            button[data-baseweb="tab"] p {
                font-size: 0.95rem;
                white-space: nowrap;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_title(icon: str, title_html: str) -> None:
    """Render a responsive page title with cleaner mobile wrapping."""
    st.markdown(
        f"""
        <div class="app-page-title">
            <span class="app-page-title__icon">{icon}</span>
            <span class="app-page-title__text">{title_html}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(icon: str, title_html: str) -> None:
    """Render a responsive section heading with controlled break points."""
    st.markdown(
        f"""
        <div class="app-section-title">
            <span class="app-section-title__icon">{icon}</span>
            <span class="app-section-title__text">{title_html}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clear_knowledge_base() -> None:
    """Clear the active knowledge base, with a fallback for older manager versions."""
    clear_method = getattr(vector_store_manager, "clear", None)
    if callable(clear_method):
        clear_method()
        return

    st.session_state["vector_store"] = None
    persist_path = app_config.vector_store_path
    try:
        if os.path.isdir(persist_path):
            shutil.rmtree(persist_path)
        elif os.path.exists(persist_path):
            os.remove(persist_path)
    except Exception as exc:
        raise RuntimeError("清空知识库失败，请稍后重试。") from exc


with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/ai.png", width=80)
    st.title("AI学习助手")
    st.caption("基于LangChain+RAG的智能问答系统")
    st.divider()

    if vector_store_manager.is_ready:
        st.success("✅ 知识库已加载")
    else:
        st.warning("⚠️ 未加载知识库")

    st.divider()
    with st.expander("📖 使用说明"):
        st.markdown("""
        1. 「知识库上传」：支持TXT/PDF，自动处理编码问题
        2. 「RAG精准问答」：LCEL架构，严格基于文档回答
        3. 「Agent智能助手」：LangGraph预构建Agent，支持复杂任务
        """)

inject_responsive_styles()
render_page_title("🤖", "我的AI学习<wbr>助手")

tab_upload, tab_rag, tab_agent = st.tabs([
    "📂 知识库上传",
    "💬 RAG精准问答",
    "🤖 Agent智能助手",
])


with tab_upload:
    render_section_title("📂", "上传你的<wbr>知识库文档")
    st.caption("支持 TXT、PDF 格式，自动处理编码问题，向量库永久保存")

    upload_success_message = cast(str, st.session_state.pop("upload_success_message", ""))
    clear_success_message = cast(str, st.session_state.pop("clear_success_message", ""))
    if upload_success_message:
        st.success(upload_success_message)
        st.info("现在可以切换到「RAG精准问答」或「Agent智能助手」继续使用。")
    if clear_success_message:
        st.success(clear_success_message)

    if vector_store_manager.is_ready:
        if st.button("🗑️ 清空知识库", type="secondary", use_container_width=True):
            try:
                clear_knowledge_base()
                st.session_state["rag_messages"] = []
                st.session_state["agent_messages"] = []
                st.session_state["upload_widget_version"] += 1
                st.session_state["clear_success_message"] = "✅ 知识库已清空"
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    uploaded_file = st.file_uploader(
        "选择文档",
        type=list(app_config.supported_file_types),
        key=f"knowledge_upload_{st.session_state['upload_widget_version']}",
    )

    if uploaded_file:
        with st.spinner("正在处理文档..."):
            try:
                split_docs = document_processor.process(uploaded_file)
                vector_store_manager.build_from_documents(split_docs)
                st.session_state["rag_messages"] = []
                st.session_state["agent_messages"] = []
                st.session_state["upload_widget_version"] += 1
                st.session_state["upload_success_message"] = (
                    f"✅ 文档处理完成！共生成 {len(split_docs)} 个文本块，向量库已永久保存"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"处理失败：{exc}")


with tab_rag:
    render_section_title("💬", "知识库精准<wbr>问答")
    st.caption("严格基于你上传的文档回答，尽量减少幻觉，并提供参考来源")

    rag_messages = cast(list[ChatMessage], st.session_state["rag_messages"])

    if not vector_store_manager.is_ready:
        st.warning("请先在「知识库上传」标签页上传文档并构建向量库。")
    else:
        render_chat_history(rag_messages, show_sources=True)

        user_question = st.chat_input("请输入你的问题...", key="rag_chat_input")
        if user_question:
            with st.chat_message("user"):
                st.markdown(user_question)
            rag_messages.append({"role": "user", "content": user_question})

            with st.spinner("正在检索知识库并生成回答..."):
                try:
                    rag = build_rag_chain(
                        vector_store_manager.get_retriever(
                            app_config.rag.retriever_top_k
                        ),
                        llm,
                    )
                    result = rag(user_question)
                    answer = result.answer
                    source_docs = result.source_docs
                except Exception as exc:
                    answer = f"调用 RAG 失败：{exc}"
                    source_docs = []

            with st.chat_message("assistant"):
                st.markdown(answer)
                if source_docs:
                    with st.expander("📄 查看参考文档来源"):
                        for idx, doc in enumerate(source_docs, 1):
                            source = doc.metadata.get("source", "未知")
                            page = doc.metadata.get("page", "N/A")
                            st.markdown(
                                f"**片段 {idx}** — 来源：`{source}` | 页码：`{page}`"
                            )
                            st.caption(doc.page_content)
                            if idx < len(source_docs):
                                st.divider()

            rag_messages.append({
                "role": "assistant",
                "content": answer,
                "source_docs": source_docs,
            })

        render_chat_controls(
            "rag_messages",
            "RAG精准问答对话记录",
            "RAG问答记录.md",
            include_sources=True,
        )


with tab_agent:
    render_section_title("🤖", "Agent智能<wbr>助手")
    st.caption("自主拆解任务、调用工具，适合总结、检索、归纳等复杂任务")

    agent_messages = cast(list[ChatMessage], st.session_state["agent_messages"])

    if not vector_store_manager.is_ready:
        st.warning("请先在「知识库上传」标签页上传文档并构建向量库。")
    else:
        render_chat_history(agent_messages)

        user_input = st.chat_input("请输入你要完成的任务...", key="agent_chat_input")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            agent_messages.append({"role": "user", "content": user_input})

            with st.spinner("Agent 正在思考并执行任务..."):
                try:
                    conversation = [
                        (msg["role"], msg["content"])
                        for msg in agent_messages
                    ]
                    answer = run_agent_task(
                        vector_store_manager.get_retriever(top_k=4),
                        llm,
                        conversation,
                        user_input,
                    )
                except Exception as exc:
                    answer = f"Agent 执行失败：{exc}\n请检查问题格式或重试"

            with st.chat_message("assistant"):
                st.markdown(answer)

            agent_messages.append({
                "role": "assistant",
                "content": answer,
            })

        render_chat_controls(
            "agent_messages",
            "Agent智能助手对话记录",
            "Agent对话记录.md",
        )
