import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool

# ======================
# 1. 页面基础配置（一次性设置）
st.set_page_config(
    page_title="我的AI学习助手",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 我的AI学习助手 | 18天学习成果落地")

# ======================
# 2. 加载配置与初始化（全局复用）
load_dotenv()
API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 初始化核心模型（只初始化1次，提升性能）
@st.cache_resource
def init_models():
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

# 向量库全局变量
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ======================
# 3. 复用核心函数（全是你之前写过的）
# 3.1 文档加载与分块
def load_and_split_file(uploaded_file):
    # 保存上传的临时文件
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 加载文档
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load_and_split()
    else:
        os.remove(file_path)
        raise Exception("仅支持TXT和PDF格式")
    
    # 优化分块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；"]
    )
    split_docs = splitter.split_documents(docs)
    os.remove(file_path)  # 删除临时文件
    return split_docs

# 3.2 构建RAG问答链
@st.cache_resource
def build_rag_chain(_vector_store):
    rag_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
【严格规则】
1. 只允许使用参考文档里的内容回答，禁止编造
2. 文档里没有相关内容，直接回复："知识库中无相关信息"
3. 回答简洁清晰，分点说明
4. 结尾标注信息来源

【参考文档】
{context}

【用户问题】
{question}
"""
    )
    retriever = _vector_store.as_retriever(search_kwargs={"k": 3})
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": rag_prompt},
        return_source_documents=True
    )
    return rag_chain

# 3.3 构建Agent智能体
@st.cache_resource
def build_agent_chain(_vector_store):
    # 定义工具
    def knowledge_query(query):
        docs = _vector_store.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(query)
        return "\n".join([doc.page_content for doc in docs])
    
    def content_summary(text):
        return llm.invoke(f"请简洁总结这段学习内容，突出核心知识点：{text}")
    
    tools = [
        Tool(
            name="知识库查询",
            func=knowledge_query,
            description="用于查询用户的AI学习笔记、学习天数、项目、知识点，必须优先使用这个工具"
        ),
        Tool(
            name="内容总结",
            func=content_summary,
            description="用于总结文本、学习内容、长段落，提炼核心信息"
        )
    ]
    
    # 记忆组件
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # ReAct思考模板
    prompt = hub.pull("hwchase17/react-chat")
    # 创建Agent
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        memory=memory
    )
    return agent_executor

# ======================
# 4. 页面分Tab展示（3大核心模块）
tab1, tab2, tab3 = st.tabs(["📂 知识库上传", "💬 RAG精准问答", "🤖 Agent智能助手"])

# ----------------------
# Tab1：知识库上传模块
with tab1:
    st.header("📂 上传你的知识库文档")
    st.caption("支持TXT、PDF格式，自动完成分块、向量化、构建向量库")
    
    uploaded_file = st.file_uploader("选择文档", type=["txt", "pdf"])
    if uploaded_file:
        with st.spinner("正在处理文档..."):
            try:
                split_docs = load_and_split_file(uploaded_file)
                # 构建向量库
                st.session_state.vector_store = FAISS.from_documents(split_docs, embeddings)
                st.success(f"✅ 文档处理完成！共生成 {len(split_docs)} 个文本块，向量库已构建")
                st.info("可以切换到「RAG精准问答」或「Agent智能助手」标签页使用啦")
            except Exception as e:
                st.error(f"处理失败：{str(e)}")

# ----------------------
# Tab2：RAG精准问答模块
with tab2:
    st.header("💬 知识库精准问答")
    st.caption("严格基于你上传的文档回答，无幻觉、可溯源")
    
    if not st.session_state.vector_store:
        st.warning("请先在「知识库上传」标签页上传文档，构建向量库")
    else:
        # 初始化对话历史
        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = []
        
        # 展示历史对话
        for msg in st.session_state.rag_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # 用户输入
        user_question = st.chat_input("请输入你的问题...")
        if user_question:
            # 展示用户问题
            with st.chat_message("user"):
                st.markdown(user_question)
            st.session_state.rag_messages.append({"role": "user", "content": user_question})
            
            # 调用RAG链
            with st.spinner("正在检索知识库并生成回答..."):
                rag_chain = build_rag_chain(st.session_state.vector_store)
                result = rag_chain.invoke(user_question)
                answer = result["result"]
                source_docs = result["source_documents"]
            
            # 展示AI回答
            with st.chat_message("assistant"):
                st.markdown(answer)
                # 折叠展示参考文档
                with st.expander("查看参考文档来源"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**参考文档 {i+1}**")
                        st.markdown(doc.page_content)
            st.session_state.rag_messages.append({"role": "assistant", "content": answer})

# ----------------------
# Tab3：Agent智能助手模块
with tab3:
    st.header("🤖 Agent智能助手")
    st.caption("自主思考、规划步骤、调用工具，完成多步骤复杂任务")
    
    if not st.session_state.vector_store:
        st.warning("请先在「知识库上传」标签页上传文档，构建向量库")
    else:
        # 初始化对话历史
        if "agent_messages" not in st.session_state:
            st.session_state.agent_messages = []
        
        # 展示历史对话
        for msg in st.session_state.agent_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # 用户输入
        user_input = st.chat_input("请输入你要完成的任务...")
        if user_input:
            # 展示用户输入
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.agent_messages.append({"role": "user", "content": user_input})
            
            # 调用Agent
            with st.spinner("Agent正在思考并执行任务..."):
                agent_chain = build_agent_chain(st.session_state.vector_store)
                result = agent_chain.invoke({"input": user_input})
                answer = result["output"]
            
            # 展示AI回答
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.agent_messages.append({"role": "assistant", "content": answer})