import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# 核心依赖导入（保留你的修改）
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from langchain_community.chat_message_histories import ChatMessageHistory

# ======================
# 1. 页面全局配置（优化质感+部署适配）
# ======================
st.set_page_config(
    page_title="我的AI学习助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 侧边栏（新增，提升产品质感）
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/ai.png", width=80)
    st.title("AI学习助手")
    st.caption("基于LangGraph+RAG的智能问答系统 | 部署适配版")
    st.divider()
    
    # 系统状态展示
    if st.session_state.get("vector_store"):
        st.success("✅ 知识库已加载")
    else:
        st.warning("⚠️ 未加载知识库")
    
    st.divider()
    # 使用说明
    with st.expander("📖 使用说明"):
        st.markdown("""
        1. 「知识库上传」：支持TXT/PDF，自动处理编码问题
        2. 「RAG精准问答」：LCEL架构，严格基于文档回答
        3. 「Agent智能助手」：LangGraph预构建Agent，支持复杂任务
        """)
    st.markdown("💡 适配Streamlit Cloud部署 | 2024版")

st.title("🤖 我的AI学习助手 | 18天学习成果落地（部署版）")

# ======================
# 2. API Key安全加载（兼容本地.env + 线上Secrets）
# ======================
load_dotenv()
# 优先读取Streamlit Secrets（线上部署），其次读取.env（本地）
try:
    API_KEY = st.secrets["DASHSCOPE_API_KEY"]
except:
    API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not API_KEY:
    st.error("未检测到 DASHSCOPE_API_KEY！\n- 本地运行：请在.env文件配置\n- 线上部署：请在Streamlit Secrets配置")
    st.stop()

# ======================
# 3. 模型初始化（缓存复用，避免重复加载）
# ======================
@st.cache_resource(show_spinner="正在初始化模型...")
def init_models():
    """初始化模型，仅初始化一次（兼容通义千问OpenAI接口）"""
    os.environ["DASHSCOPE_API_KEY"] = API_KEY
    # 向量化模型（保留你的配置）
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    # LLM模型（OpenAI兼容模式调用通义千问）
    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo",
        temperature=0.2,
        max_tokens=1500
    )
    return embeddings, llm

embeddings, llm = init_models()

# ======================
# 4. Session State初始化（新增向量库持久化）
# ======================
# 向量库持久化路径（部署环境也能读写）
VECTOR_STORE_PATH = "./saved_vector_store"

# 初始化核心状态
if "vector_store" not in st.session_state:
    # 启动时自动加载已保存的向量库（部署环境兼容）
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            st.session_state.vector_store = FAISS.load_local(
                VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
            )
            st.toast("✅ 已加载历史知识库", icon="📚")
        except Exception as e:
            st.warning(f"加载历史知识库失败：{str(e)}，将重新构建")
            st.session_state.vector_store = None
    else:
        st.session_state.vector_store = None

# 对话历史初始化（保留你的配置）
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

if "agent_history" not in st.session_state:
    st.session_state.agent_history = ChatMessageHistory()

# ======================
# 5. 核心函数（保留你的修改，新增部署适配）
# ======================
def load_txt_with_fallback(file_path: str):
    """TXT 编码兜底加载（保留你的核心逻辑）"""
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
    """上传文件处理（保留你的tempfile逻辑，优化容错）"""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    temp_file_path = None

    try:
        # 临时文件处理（部署环境兼容）
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_file_path = tmp_file.name

        # 加载文档（保留你的编码兜底）
        if suffix == ".txt":
            docs = load_txt_with_fallback(temp_file_path)
        elif suffix == ".pdf":
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
        else:
            raise Exception("仅支持 TXT 和 PDF 格式")

        # 分块（保留你的参数调整）
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]
        )
        split_docs = splitter.split_documents(docs)
        return split_docs

    finally:
        # 清理临时文件（部署环境容错）
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

def build_rag_chain(vector_store):
    """构建 LCEL 架构的 RAG 链（保留你的逻辑，新增源文档返回）"""
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
        """格式化文档（新增：保留源文档信息）"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "未知页码")
            formatted.append(f"【片段{i}】（{source} - 页码{page}）：\n{doc.page_content}")
        return "\n\n".join(formatted)
    
    # 新增：获取原始文档（用于展示参考来源）
    def get_source_docs(query):
        return retriever.invoke(query)

    # LCEL 链（拆分：先获取文档，再生成回答）
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "source_docs": RunnableLambda(get_source_docs)  # 新增：传递源文档
        }
        | rag_prompt
        | llm
    )
    return rag_chain, get_source_docs

def build_agent_chain(vector_store):
    """构建 LangGraph Prebuilt Agent（保留你的逻辑，新增日志）"""
    def knowledge_query(query: str):
        """知识库查询工具（保留你的格式化）"""
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
        """内容总结工具（保留你的逻辑）"""
        return llm.invoke(f"请简洁总结这段学习内容，突出核心知识点：\n{text}").content

    # 工具定义（保留你的配置）
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

    # LangGraph 预构建 Agent（保留你的核心）
    agent = create_react_agent(llm, tools)
    return agent

# ======================
# 6. 页面分 Tab 展示（保留你的逻辑，新增对话导出/清空）
# ======================
tab1, tab2, tab3 = st.tabs(["📂 知识库上传", "💬 RAG精准问答", "🤖 Agent智能助手"])

# ----------------------
# Tab1：知识库上传（新增向量库持久化）
# ----------------------
with tab1:
    st.header("📂 上传你的知识库文档")
    st.caption("支持 TXT、PDF 格式，自动处理编码问题，向量库永久保存")

    uploaded_file = st.file_uploader("选择文档", type=["txt", "pdf"])

    if uploaded_file:
        with st.spinner("正在处理文档..."):
            try:
                split_docs = load_and_split_file(uploaded_file)
                # 构建并持久化向量库（新增）
                st.session_state.vector_store = FAISS.from_documents(split_docs, embeddings)
                st.session_state.vector_store.save_local(VECTOR_STORE_PATH)

                # 清空历史（保留你的逻辑）
                st.session_state.rag_messages = []
                st.session_state.agent_messages = []
                st.session_state.agent_history = ChatMessageHistory()

                st.success(f"✅ 文档处理完成！共生成 {len(split_docs)} 个文本块，向量库已永久保存")
                st.info("现在可以切换到「RAG精准问答」或「Agent智能助手」继续使用。")
            except Exception as e:
                st.error(f"处理失败：{str(e)}")

# ----------------------
# Tab2：RAG精准问答（新增源文档展示+对话导出）
# ----------------------
with tab2:
    st.header("💬 知识库精准问答")
    st.caption("严格基于你上传的文档回答，尽量减少幻觉，并提供参考来源")

    if st.session_state.vector_store is None:
        st.warning("请先在「知识库上传」标签页上传文档并构建向量库。")
    else:
        # 展示历史对话（保留你的逻辑）
        for msg in st.session_state.rag_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # 展示参考来源（新增）
                if msg["role"] == "assistant" and "source_docs" in msg:
                    with st.expander("查看参考文档来源"):
                        for i, doc in enumerate(msg["source_docs"], 1):
                            st.markdown(f"**参考片段 {i}**")
                            st.markdown(f"来源：{doc.metadata.get('source', '未知')} | 页码：{doc.metadata.get('page', '未知')}")
                            st.markdown(doc.page_content)

        # 用户输入（保留你的逻辑）
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
                    # 构建RAG链（新增获取源文档）
                    rag_chain, get_source_docs = build_rag_chain(st.session_state.vector_store)
                    # 调用链
                    result = rag_chain.invoke(user_question)
                    answer = result.content if hasattr(result, 'content') else str(result)
                    # 获取源文档（新增）
                    source_docs = get_source_docs(user_question)
                except Exception as e:
                    answer = f"调用 RAG 失败：{str(e)}"
                    source_docs = []

            # 展示回答（新增源文档折叠）
            with st.chat_message("assistant"):
                st.markdown(answer)
                if source_docs:
                    with st.expander("查看参考文档来源"):
                        for i, doc in enumerate(source_docs, 1):
                            st.markdown(f"**参考片段 {i}**")
                            st.markdown(f"来源：{doc.metadata.get('source', '未知')} | 页码：{doc.metadata.get('page', '未知')}")
                            st.markdown(doc.page_content)

            # 保存对话（新增源文档）
            st.session_state.rag_messages.append({
                "role": "assistant",
                "content": answer,
                "source_docs": source_docs
            })
        
        # 对话管理：清空+导出（新增）
        if st.session_state.rag_messages:
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("清空对话历史", key="rag_clear"):
                    st.session_state.rag_messages = []
                    st.rerun()
            with col2:
                # 导出Markdown格式
                md_content = "# RAG精准问答对话记录\n\n"
                for msg in st.session_state.rag_messages:
                    md_content += f"## {msg['role']}\n{msg['content']}\n\n"
                    if msg["role"] == "assistant" and "source_docs" in msg:
                        md_content += "### 参考来源\n"
                        for i, doc in enumerate(msg["source_docs"], 1):
                            md_content += f"- 片段{i}：{doc.metadata.get('source', '未知')}（页码{doc.metadata.get('page', '未知')}）\n{doc.page_content}\n\n"
                st.download_button(
                    label="导出对话记录",
                    data=md_content,
                    file_name="RAG问答记录.md",
                    mime="text/markdown",
                    key="rag_export"
                )

# ----------------------
# Tab3：Agent智能助手（新增对话导出+优化错误处理）
# ----------------------
with tab3:
    st.header("🤖 Agent智能助手")
    st.caption("自主拆解任务、调用工具，适合总结、检索、归纳等复杂任务")

    if st.session_state.vector_store is None:
        st.warning("请先在「知识库上传」标签页上传文档并构建向量库。")
    else:
        # 展示历史对话（保留你的逻辑）
        for msg in st.session_state.agent_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 用户输入（保留你的逻辑）
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
                    # 调用Agent（保留你的逻辑，优化容错）
                    result = agent.invoke({"messages": [("user", user_input)]})
                    # 提取AI回复（优化容错）
                    ai_messages = [m for m in result.get("messages", []) if hasattr(m, 'type') and m.type == 'ai']
                    if ai_messages:
                        answer = ai_messages[-1].content
                    else:
                        answer = "Agent 未返回有效结果"
                except Exception as e:
                    answer = f"Agent 执行失败：{str(e)}\n请检查问题格式或重试"

            # 展示回答（保留你的逻辑）
            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.agent_messages.append({
                "role": "assistant",
                "content": answer
            })
        
        # 对话管理：清空+导出（新增）
        if st.session_state.agent_messages:
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("清空对话历史", key="agent_clear"):
                    st.session_state.agent_messages = []
                    st.session_state.agent_history = ChatMessageHistory()
                    st.rerun()
            with col2:
                # 导出Markdown格式
                md_content = "# Agent智能助手对话记录\n\n"
                for msg in st.session_state.agent_messages:
                    md_content += f"## {msg['role']}\n{msg['content']}\n\n"
                st.download_button(
                    label="导出对话记录",
                    data=md_content,
                    file_name="Agent对话记录.md",
                    mime="text/markdown",
                    key="agent_export"
                )