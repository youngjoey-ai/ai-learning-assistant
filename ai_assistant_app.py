import os
import shutil
from uuid import uuid4
from typing import cast

import streamlit as st
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from chains import AgentRunDetails, run_agent_task, run_task_agent
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
AGENT_TIMEOUT_SECONDS = 45.0

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

init_session_key("agent_messages", [])
init_session_key("agent_thread_id", f"agent-{uuid4().hex}")
init_session_key("task_messages", [])
init_session_key("task_thread_id", f"task-{uuid4().hex}")
init_session_key("agent_mode", "📚 文档问答（知识库）")
init_session_key("mode_recommendation_message", "")
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


def recommend_agent_mode(task_text: str) -> str:
    query = task_text.lower()
    task_mode_keywords = (
        "今天",
        "最新",
        "实时",
        "天气",
        "新闻",
        "联网",
        "搜索",
        "计算",
        "算一下",
        "时间",
        "待办",
        "计划",
        "deadline",
        "todo",
    )
    if any(keyword in query for keyword in task_mode_keywords):
        return "🛠️ 任务执行（联网/计算）"
    return "📚 文档问答（知识库）"


with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/ai.png", width=80)
    st.title("AI学习助手")
    st.caption("基于LangChain+RAG的智能问答系统")
    st.caption("定位：Agent编排驱动，兼具文档问答与任务执行")
    st.divider()

    if vector_store_manager.is_ready:
        st.success("✅ 知识库已加载")
    else:
        st.warning("⚠️ 未加载知识库")

    st.divider()
    with st.expander("📖 使用说明"):
        st.markdown("""
        「知识库上传」：支持TXT/PDF，自动处理编码问题。\n
        「Agent 对话」：所有问题先进入Agent决策，再自动调用知识库与联网工具。\n
         优先知识库回答：知识库无相关信息时，自动触发联网搜索补充。
        """)

inject_responsive_styles()
render_page_title("🤖", "我的AI学习<wbr>助手")

tab_upload, tab_agent = st.tabs([
    "📂 知识库上传",
    "🤖 Agent 对话",
])


with tab_upload:
    render_section_title("📂", "上传你的<wbr>知识库文档")
    st.caption("支持 TXT、PDF 格式，自动处理编码问题，向量库永久保存")

    upload_success_message = cast(str, st.session_state.pop("upload_success_message", ""))
    clear_success_message = cast(str, st.session_state.pop("clear_success_message", ""))
    if upload_success_message:
        st.success(upload_success_message)
        st.info("现在可以切换到「Agent 对话」继续使用。")
    if clear_success_message:
        st.success(clear_success_message)

    if vector_store_manager.is_ready:
        if st.button("🗑️ 清空知识库", type="secondary", use_container_width=True):
            try:
                clear_knowledge_base()
                st.session_state["agent_messages"] = []
                st.session_state["agent_thread_id"] = f"agent-{uuid4().hex}"
                st.session_state["task_messages"] = []
                st.session_state["task_thread_id"] = f"task-{uuid4().hex}"
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
                st.session_state["agent_messages"] = []
                st.session_state["agent_thread_id"] = f"agent-{uuid4().hex}"
                st.session_state["task_messages"] = []
                st.session_state["task_thread_id"] = f"task-{uuid4().hex}"
                st.session_state["upload_widget_version"] += 1
                st.session_state["upload_success_message"] = (
                    f"✅ 文档处理完成！共生成 {len(split_docs)} 个文本块，向量库已永久保存"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"处理失败：{exc}")


with tab_agent:
    render_section_title("🤖", "Agent<wbr> 对话")
    recommendation_message = cast(str, st.session_state.pop("mode_recommendation_message", ""))
    if recommendation_message:
        st.info(recommendation_message)
    mode_hint = st.text_input(
        "不确定选哪个模式？先输入一句任务！",
        key="mode_hint_input",
    )
    if st.button("✨ 智能推荐模式", use_container_width=True):
        if mode_hint.strip():
            recommended_mode = recommend_agent_mode(mode_hint.strip())
            st.session_state["agent_mode"] = recommended_mode
            st.session_state["mode_recommendation_message"] = f"推荐：{recommended_mode}"
        else:
            st.session_state["mode_recommendation_message"] = "请先输入任务内容，再进行模式推荐。"
        st.rerun()
    mode = st.radio(
        "选择Agent模式",
        ("📚 文档问答（知识库）", "🛠️ 任务执行（联网/计算）"),
        horizontal=True,
        key="agent_mode",
    )
    is_knowledge_mode = mode == "📚 文档问答（知识库）"
    if is_knowledge_mode:
        st.caption("适合：解释你上传文档、提炼知识点、基于资料回答。示例：帮我总结这份PDF第3章。")
    else:
        st.caption("适合：实时信息、计算、通用任务。示例：帮我查今天AI新闻并生成待办清单。")
    messages_key = "agent_messages" if is_knowledge_mode else "task_messages"
    thread_key = "agent_thread_id" if is_knowledge_mode else "task_thread_id"
    messages = cast(list[ChatMessage], st.session_state[messages_key])

    if is_knowledge_mode and not vector_store_manager.is_ready:
        st.warning("请先在「知识库上传」标签页上传文档并构建向量库。")
    else:
        render_chat_history(messages)

        input_placeholder = (
            "例如：请根据我上传的文档解释RAG和Agent区别"
            if is_knowledge_mode
            else "例如：查一下今天AI行业要闻，并给我3条行动建议"
        )
        user_input = st.chat_input(input_placeholder, key="agent_chat_input")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            messages.append({"role": "user", "content": user_input})

            with st.spinner("Agent 正在思考并执行任务..."):
                try:
                    conversation = [
                        (msg["role"], msg["content"])
                        for msg in messages
                    ]
                    if is_knowledge_mode:
                        answer = run_agent_task(
                            vector_store_manager.get_retriever(top_k=4),
                            llm,
                            conversation,
                            user_input,
                            cast(str, st.session_state[thread_key]),
                            return_details=True,
                            timeout_seconds=AGENT_TIMEOUT_SECONDS,
                        )
                    else:
                        answer = run_task_agent(
                            llm,
                            conversation,
                            user_input,
                            cast(str, st.session_state[thread_key]),
                            return_details=True,
                            timeout_seconds=AGENT_TIMEOUT_SECONDS,
                        )
                except Exception as exc:
                    answer = f"Agent 执行失败：{exc}\n请检查问题格式或重试"

            with st.chat_message("assistant"):
                if isinstance(answer, AgentRunDetails):
                    st.markdown(answer.answer)
                    with st.expander("🧭 Agent执行观测"):
                        st.json(answer.metrics)
                        st.json(answer.trace)
                    answer_text = answer.answer
                else:
                    st.markdown(answer)
                    answer_text = answer

            messages.append({
                "role": "assistant",
                "content": answer_text,
            })

        render_chat_controls(
            messages_key,
            "Agent智能助手对话记录" if is_knowledge_mode else "任务执行Agent对话记录",
            "Agent对话记录.md" if is_knowledge_mode else "任务执行Agent对话记录.md",
            extra_reset_keys=[thread_key],
        )
