import os
import traceback
from datetime import datetime
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
# 1. 基础配置（完全复用你的架构，新增工具相关配置）
load_dotenv()
API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not API_KEY:
    raise ValueError("未找到 DASHSCOPE_API_KEY，请检查 .env 文件！")

# 初始化 LLM（保持通义千问兼容模式）
llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.2
)

# 初始化向量库（路径保持你的原有配置）
embeddings = DashScopeEmbeddings(model="text-embedding-v2")
try:
    store_path = os.path.join(os.path.dirname(__file__), "../Day15/rag_optimized_store")
    vector_store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})  # 进阶版调大检索数量
    print("✅ 知识库加载成功！")
except Exception as e:
    print(f"⚠️ 知识库加载失败: {e}")
    retriever = None

# ======================
# 2. 工具集大升级：5大工具协同（全部用@tool装饰器）
# ----------------------
# 工具1：知识库查询（保留你的原有逻辑）
@tool
def knowledge_query(query: str) -> str:
    """查询本地学习笔记知识库，当用户问学习进度、天数、内容、知识点时必须优先使用"""
    print(f"🔧 [工具调用] knowledge_query: {query}")
    if retriever is None:
        return "知识库未加载，无法查询"
    docs = retriever.invoke(query)
    result = "\n\n".join([d.page_content for d in docs])
    print(f"📖 [工具结果] {result[:150]}...")
    return result or "未找到相关内容"

# 工具2：内容总结（保留你的原有逻辑，优化提示词）
@tool
def study_summary(text: str) -> str:
    """总结文本内容，当用户需要总结学习内容、生成周报、整理知识点时使用"""
    print(f"🔧 [工具调用] study_summary")
    prompt = """请按以下格式输出总结（必须严格遵守）：

第1点：xxx
第2点：xxx
第3点：xxx
...

要求：
1. 每一点必须单独一行
2. 不要用Markdown符号
3. 每条不超过20字

总结内容：{text}"""
    return llm.invoke(prompt.format(text=text)).content

# 工具3：学习统计（新增核心工具）
@tool
def study_statistics(query: str) -> str:
    """统计学习数据，当用户问学习总天数、模块数量、进度时使用"""
    print(f"🔧 [工具调用] study_statistics: {query}")
    # 先调用知识库获取全量笔记（使用 .invoke()）
    all_notes = knowledge_query.invoke("第1天到第17天的所有学习笔记")
    if "知识库未加载" in all_notes:
        return all_notes
    
    # 调用LLM做统计分析
    stat_prompt = f"""请统计以下学习笔记的核心信息，按以下格式输出：

总学习天数：xx天
核心模块：xxx
项目1：xxx
项目2：xxx
...

要求：每条单独一行，不要用Markdown符号

笔记内容：{all_notes[:1000]}"""
    stat_result = llm.invoke(stat_prompt).content
    print(f"📊 [统计结果] {stat_result[:100]}...")
    return stat_result

# 工具4：保存内容到文件（新增实用工具）
@tool
def save_to_file(content: str) -> str:
    """将文本内容保存为本地TXT文件，当用户要求保存总结、周报、统计结果时使用"""
    print(f"🔧 [工具调用] save_to_file")
    try:
        # 获取脚本所在目录（Day17目录）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 生成文件名（带时间戳避免覆盖）
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        file_name = f"AI学习总结_{timestamp}.txt"
        file_path = os.path.join(script_dir, file_name)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        result = f"内容已成功保存到文件：{file_path}"
        print(f"💾 [保存结果] {result}")
        return result
    except Exception as e:
        error_msg = f"保存失败：{str(e)}"
        print(f"❌ [保存失败] {error_msg}")
        return error_msg

# 工具5：简单计算（新增工具）
@tool
def simple_calculate(expr: str) -> str:
    """做简单的加减乘除计算，仅处理数字运算（如 17-10、8*2），禁止复杂表达式"""
    print(f"🔧 [工具调用] simple_calculate: {expr}")
    # 安全校验：只允许数字和+-*/()
    allowed_chars = "0123456789+-*/(). "
    if not all(c in allowed_chars for c in expr):
        return "计算表达式非法，仅支持数字和+-*/()"
    
    try:
        # 安全执行计算
        result = eval(expr, {"__builtins__": None}, {})
        return f"计算结果：{expr} = {result}"
    except:
        return "计算错误，请检查表达式（如：17-10、8*2）"

# 整合所有工具（新增工具加入列表）
tools = [knowledge_query, study_summary, study_statistics, save_to_file, simple_calculate]

# ======================
# 3. 升级System Prompt（明确多工具使用规则）
system_message = SystemMessage(content="""你是一个专业的AI学习助手，能自主选择工具完成复杂任务。

核心规则：
1. 优先判断任务类型，选择对应工具：
   - 查学习进度/天数/内容 → 用 knowledge_query
   - 总结/周报 → 用 study_summary
   - 统计天数/模块 → 用 study_statistics
   - 保存内容 → 用 save_to_file
   - 计算数字 → 用 simple_calculate
2. 复杂任务需要多工具协同（如生成周报并保存）：先查知识库→总结→保存
3. 必须基于工具返回结果回答，禁止编造信息
4. 回答用纯文本格式，分点清晰，无Markdown符号（如**、*、#）
5. 多轮对话要记住上下文（比如用户先查进度，再要求总结，要关联之前的信息）
""")

# ======================
# 4. 创建进阶版Agent（保留你的MemorySaver记忆）
memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_message.content,
    checkpointer=memory
    # 注意：create_react_agent 不支持 agent_executor_kwargs 参数
)

# ======================
# 5. 交互式体验（优化复杂任务处理）
if __name__ == "__main__":
    print("===== 🤖 LangGraph进阶版Agent（多工具协同） =====")
    print("支持：查进度、统计、总结、生成周报、自动保存文件、计算")
    print("输入「退出」结束对话\n")
    
    # 对话配置（保留你的session ID）
    config = {"configurable": {"thread_id": "user_session_1"}}
    
    while True:
        user_input = input("你：")
        if user_input in ["退出", "q"]:
            print("AI：再见！")
            break
        
        if not user_input.strip():
            print("⚠️ 请输入有效问题！")
            continue
        
        try:
            # 构建消息列表
            messages = [HumanMessage(content=user_input)]
            
            # 调用进阶版Agent
            print("🤔 [Agent 思考中（可能调用多个工具）...]")
            response = agent.invoke(
                {"messages": messages},
                config=config
            )
            
            # 提取AI回答（保持你的原有逻辑）
            ai_messages = [m for m in response["messages"] if hasattr(m, 'type') and m.type == 'ai']
            if ai_messages:
                last_ai_message = ai_messages[-1]
                answer = last_ai_message.content
            else:
                answer = "没有获取到回答"
            
            print(f"\n🤖 回答：{answer}\n")
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")