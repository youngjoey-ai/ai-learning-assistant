import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决 OpenMP 冲突

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.llms import Tongyi
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings

# ======================
# 1. 基础配置
load_dotenv()
API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not API_KEY:
    raise ValueError("未找到 DASHSCOPE_API_KEY，请检查 .env 文件！")

# 初始化 LLM
llm = Tongyi(
    model="qwen-turbo",
    temperature=0.2
)

# 初始化 Embeddings 和 RAG 向量库
embeddings = DashScopeEmbeddings(model="text-embedding-v2")
try:
    store_path = os.path.join(os.path.dirname(__file__), "../Day15/rag_optimized_store")
    vector_store = FAISS.load_local(
        store_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print("✅ 知识库加载成功！")
except Exception as e:
    print(f"⚠️ 知识库加载失败: {e}")
    retriever = None

# ======================
# 2. 定义工具 (使用 @tool 装饰器)
@tool
def knowledge_query(query: str) -> str:
    """查询本地学习笔记知识库的工具。当用户问学习相关问题时使用。"""
    if retriever is None:
        return "知识库未加载，无法查询"
    try:
        relevant_docs = retriever.invoke(query)
        doc_content = "\n\n".join([doc.page_content for doc in relevant_docs])
        return doc_content if doc_content else "知识库中未找到相关内容"
    except Exception as e:
        return f"查询失败: {e}"

@tool
def study_summary(text: str) -> str:
    """总结文本内容、学习笔记。当用户需要总结时使用。"""
    try:
        summary_prompt = f"请简洁、分点总结以下学习内容（不超过100字）：{text}"
        return llm.invoke(summary_prompt)
    except Exception as e:
        return f"总结失败: {e}"

tools = [knowledge_query, study_summary]

# ======================
# 3. 简化版 Agent 实现
def run_agent(user_input: str) -> str:
    """
    简化版 Agent：总是先调用知识库查询工具
    """
    # 策略：总是先查询知识库
    print(f"🔧 使用工具：知识库查询")
    tool_result = knowledge_query.invoke(user_input)
    print(f"📖 工具结果：{tool_result[:100] if len(tool_result) > 100 else tool_result}...")
    
    # 如果知识库有内容，基于工具结果回答
    if tool_result and "未找到" not in tool_result and "未加载" not in tool_result:
        final_prompt = f"""基于以下信息回答用户问题：

用户问题：{user_input}

参考信息：
{tool_result}

请给出准确、有帮助的回答："""
    else:
        # 知识库没有内容，让 LLM 直接回答
        final_prompt = f"请回答用户问题：{user_input}"
    
    return llm.invoke(final_prompt)

# ======================
# 4. 交互式体验
if __name__ == "__main__":
    print("===== 🤖 你的AI学习Agent已上线 =====")
    print("支持：查学习进度、总结知识点、多步骤任务")
    print("输入「退出」结束对话\n")
    
    while True:
        user_input = input("你：")
        if user_input in ["退出", "q"]:
            print("AI：再见！")
            break
        
        if not user_input.strip():
            continue
        
        try:
            response = run_agent(user_input)
            print(f"\n🤖 回答：{response}\n")
        except Exception as e:
            print(f"\n❌ 运行出错: {e}\n")