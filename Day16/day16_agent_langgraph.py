import os
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

# ======================
# 1. 基础配置
load_dotenv()
API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not API_KEY:
    raise ValueError("未找到 DASHSCOPE_API_KEY，请检查 .env 文件！")

# 初始化 LLM
llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.2
)

# 初始化向量库
embeddings = DashScopeEmbeddings(model="text-embedding-v2")
try:
    store_path = os.path.join(os.path.dirname(__file__), "../Day15/rag_optimized_store")
    vector_store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print("✅ 知识库加载成功！")
except Exception as e:
    print(f"⚠️ 知识库加载失败: {e}")
    retriever = None


def _stringify_content(content: Any) -> str:
    """Normalize LangChain message content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)

# ======================
# 2. 定义工具
@tool
def knowledge_query(query: str) -> str:
    """查询本地学习笔记知识库，当用户问学习进度、天数、内容时使用"""
    print(f"🔧 [工具调用] knowledge_query: {query}")
    if retriever is None:
        return "知识库未加载"
    docs = retriever.invoke(query)
    result = "\n\n".join([d.page_content for d in docs])
    print(f"📖 [工具结果] {result[:150]}...")
    return result or "未找到相关内容"

@tool
def study_summary(text: str) -> str:
    """总结文本内容，当用户需要总结时使用"""
    print(f"🔧 [工具调用] study_summary")
    response = llm.invoke(f"请简洁总结：{text}")
    return _stringify_content(response.content)

tools = [knowledge_query, study_summary]

# ======================
# 3. 使用 LangGraph 创建 Agent
# 系统提示词
system_message = SystemMessage(content="""你是一个专业的AI学习助手，可以使用工具来回答用户关于学习的问题。

重要规则：
1. 如果用户问学习进度、内容、知识点、天数等，必须使用 knowledge_query 工具
2. 如果用户需要总结，使用 study_summary 工具
3. 基于工具返回的结果回答用户问题，不要编造信息
4. 回答时不要用 Markdown 格式（如 **粗体**、*斜体*），用纯文本格式
""")

# 创建记忆组件
memory = MemorySaver()

# 创建 Agent
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_message.content, 
    checkpointer=memory,
)

# ======================
# 4. 交互式体验
if __name__ == "__main__":
    print("===== 🤖 LangGraph Agent (完整版) =====")
    print("使用 create_react_agent + MemorySaver")
    print("输入「退出」结束对话\n")
    
    # 对话配置
    config = {"configurable": {"thread_id": "user_session_1"}}
    
    while True:
        user_input = input("你：")
        if user_input in ["退出", "q"]:
            print("AI：再见！")
            break
        
        if not user_input.strip():
            continue
        
        try:
            # 构建消息列表
            messages = [HumanMessage(content=user_input)]
            
            # 调用 Agent
            print("🤔 [Agent 思考中...]")
            response = agent.invoke(
                {"messages": messages},
                config=config
            )
            
            # 提取最后一条 AI 消息
            ai_messages = [m for m in response["messages"] if hasattr(m, 'type') and m.type == 'ai']
            if ai_messages:
                last_ai_message = ai_messages[-1]
                answer = _stringify_content(last_ai_message.content)
            else:
                answer = "没有获取到回答"
            
            print(f"\n🤖 回答：{answer}\n")
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")
            import traceback
            traceback.print_exc()
