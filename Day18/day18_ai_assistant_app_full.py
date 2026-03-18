import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from langchain_community.chat_message_histories import ChatMessageHistory

# ======================
# 1. 页面基础配置
# ======================
st.set_page_config(
    page_title="我的AI学习助手",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 我的AI学习助手 | 18天学习成果落地")

# ======================
# 2. 加载配置与初始化
# ======================
load_dotenv()
API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not API_KEY:
    st.error("未检测到 DASHSCOPE_API_KEY，请先在 .env 中配置。")
    st.stop()


@st.cache_resource
def init_models():
    """初始化模型，仅初始化一次"""
    import os
    os.environ["DASHSCOPE_API_KEY"] = API_KEY
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo",
        temperature=0.2,
        max_tokens=1500
    )
    return embeddings, llm


embeddings, llm = init_models()

# session_state 初始化
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

if "agent_history" not in st.session_state:
    st.session_state.agent_history = ChatMessageHistory()


# ======================
# 3. 核心函数
# ======================
def load_txt_with_fallback(file_path: str):
    """TXT 编码兜底加载"""
    encodings = ["utf-8", "gb18030", "gbk"]
    last_error = None

    for enc in encodings:
        try:
            loader = TextLoader(file_path, encoding=enc)
            return loader.load()
        except Exception as e:
            last_error = e

    raise Exception(f"TXT 文件加载失败，请检查编码格式。原始错误：{last_error}")


def load_and_split_file(uploaded_file):
    """上传文件 -> 临时保存 -> 加载 -> 分块 -> 删除临时文件"""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    temp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_file_path = tmp_file.name

        # 加载文档
        if suffix == ".txt":
            docs = load_txt_with_fallback(temp_file_path)
        elif suffix == ".pdf":
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
        else:
            raise Exception("仅支持 TXT 和 PDF 格式")

        # 统一分块
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]
        )
        split_docs = splitter.split_documents(docs)
        return split_docs

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def build_rag_chain(vector_store):
    """构建 RAG 问答链，使用 LCEL 方式"""
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个严格依据知识库回答问题的助手。

【严格规则】
1. 只允许使用参考文档中的内容回答，禁止编造
2. 如果参考文档中没有相关内容，直接回复：知识库中无相关信息
3. 回答要简洁、清晰、分点说明
4. 回答结尾请附上信息来源"""),
        ("human", """【参考文档】
{context}

【用户问题】
{question}""")
    ])

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
    )
    return rag_chain


def build_agent_chain(vector_store):
    """构建 Agent，使用 LangGraph"""
    def knowledge_query(query: str):
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        if not docs:
            return "知识库中无相关信息"

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "未知页码")
            results.append(
                f"【片段{i}】\n来源：{source}\n页码：{page}\n内容：{doc.page_content}"
            )
        return "\n\n".join(results)

    def content_summary(text: str):
        return llm.invoke(f"请简洁总结这段学习内容，突出核心知识点：\n{text}")

    tools = [
        Tool(
            name="knowledge_query",
            func=knowledge_query,
            description="用于查询用户上传知识库中的内容。涉及学习笔记、知识点、项目、总结时，优先使用这个工具。"
        ),
        Tool(
            name="content_summary",
            func=content_summary,
            description="用于总结长文本，提炼核心知识点。"
        )
    ]

    agent = create_react_agent(llm, tools)
    return agent


# ======================
# 4. 页面分 Tab 展示
# ======================
tab1, tab2, tab3 = st.tabs(["📂 知识库上传", "💬 RAG精准问答", "🤖 Agent智能助手"])

# ----------------------
# Tab1：知识库上传
# ----------------------
with tab1:
    st.header("📂 上传你的知识库文档")
    st.caption("支持 TXT、PDF 格式，自动完成分块、向量化、构建向量库")

    uploaded_file = st.file_uploader("选择文档", type=["txt", "pdf"])

    if uploaded_file:
        with st.spinner("正在处理文档..."):
            try:
                split_docs = load_and_split_file(uploaded_file)
                st.session_state.vector_store = FAISS.from_documents(split_docs, embeddings)

                # 每次重建向量库，建议清空历史，避免旧对话干扰新知识库
                st.session_state.rag_messages = []
                st.session_state.agent_messages = []
                st.session_state.agent_history = ChatMessageHistory()

                st.success(f"✅ 文档处理完成！共生成 {len(split_docs)} 个文本块，向量库已构建")
                st.info("现在可以切换到「RAG精准问答」或「Agent智能助手」继续使用。")
            except Exception as e:
                st.error(f"处理失败：{str(e)}")


# ----------------------
# Tab2：RAG精准问答
# ----------------------
with tab2:
    st.header("💬 知识库精准问答")
    st.caption("严格基于你上传的文档回答，尽量减少幻觉，并提供参考来源")

    if st.session_state.vector_store is None:
        st.warning("请先在「知识库上传」标签页上传文档并构建向量库。")
    else:
        for msg in st.session_state.rag_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_question = st.chat_input("请输入你的问题...", key="rag_chat_input")

        if user_question:
            with st.chat_message("user"):
                st.markdown(user_question)
            st.session_state.rag_messages.append({
                "role": "user",
                "content": user_question
            })

            with st.spinner("正在检索知识库并生成回答..."):
                try:
                    rag_chain = build_rag_chain(st.session_state.vector_store)

                    # LCEL 链直接返回 AIMessage
                    result = rag_chain.invoke(user_question)
                    answer = result.content if hasattr(result, 'content') else str(result)
                    source_docs = []
                except Exception as e:
                    answer = f"调用 RAG 失败：{str(e)}"
                    source_docs = []

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.rag_messages.append({
                "role": "assistant",
                "content": answer
            })


# ----------------------
# Tab3：Agent 智能助手
# ----------------------
with tab3:
    st.header("🤖 Agent智能助手")
    st.caption("自主拆解任务、调用工具，适合总结、检索、归纳等复杂任务")

    if st.session_state.vector_store is None:
        st.warning("请先在「知识库上传」标签页上传文档并构建向量库。")
    else:
        for msg in st.session_state.agent_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("请输入你要完成的任务...", key="agent_chat_input")

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.agent_messages.append({
                "role": "user",
                "content": user_input
            })

            with st.spinner("Agent 正在思考并执行任务..."):
                try:
                    agent = build_agent_chain(st.session_state.vector_store)
                    result = agent.invoke({"messages": [("user", user_input)]})
                    # 从结果中提取 AI 回复
                    ai_messages = [m for m in result["messages"] if hasattr(m, 'type') and m.type == 'ai']
                    if ai_messages:
                        answer = ai_messages[-1].content
                    else:
                        answer = "Agent 未返回结果"
                except Exception as e:
                    answer = f"Agent 执行失败：{str(e)}"

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.agent_messages.append({
                "role": "assistant",
                "content": answer
            })