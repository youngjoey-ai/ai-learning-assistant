import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Tongyi
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# ======================
# 1. 基础配置（复用第14天）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv()
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise ValueError("DASHSCOPE_API_KEY 未设置，请在.env文件中添加或设置环境变量")
os.environ["DASHSCOPE_API_KEY"] = API_KEY

llm = Tongyi(model="qwen-turbo", temperature=0.3, max_tokens=1500)
embeddings = DashScopeEmbeddings(model="text-embedding-v2")

# ======================
# 2. 更适合中文的文档分块（长文档友好）
def load_and_split_optimized(file_path):
    # 转换为绝对路径
    if not os.path.isabs(file_path):
        file_path = os.path.join(SCRIPT_DIR, file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load_and_split()
    else:
        raise Exception("仅支持TXT/PDF")

    # 优化分块：更小chunk + 更大重叠，中文语义更连贯
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,    # 缩小块，提高检索精度
        chunk_overlap=100, # 增大重叠，避免语义断裂
        separators=["\n\n", "\n", "。", "！", "？", "；"]
    )
    split_docs = splitter.split_documents(docs)
    print(f"✅ 优化分块完成：{len(split_docs)} 块")
    return split_docs

# ======================
# 3. 向量存储管理
def build_vector_store(split_docs):
    store_path = os.path.join(SCRIPT_DIR, "rag_optimized_store")
    vs = FAISS.from_documents(split_docs, embeddings)
    vs.save_local(store_path)
    return vs

def load_vector_store():
    store_path = os.path.join(SCRIPT_DIR, "rag_optimized_store")
    return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)

# 辅助函数：格式化文档
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# ======================
# 4. 带记忆的多轮对话RAG
def build_optimized_chain(vector_store):
    # 检索器：取Top3，精度更高
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 对话历史存储
    chat_history = ChatMessageHistory()

    # 构建带记忆的prompt模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """【严格规则】
1. 只允许使用下面【参考文档】里的内容回答
2. 文档里没有 → 直接说："知识库中无相关信息"
3. 回答要简洁、分点、口语化
4. 最后标注：【信息来源：参考文档】

【参考文档】
{context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    # 构建RAG链（使用管道操作符）
    def retrieve_and_format(question):
        docs = retriever.invoke(question)
        return format_docs(docs)

    rag_chain = (
        {
            "context": RunnablePassthrough() | retrieve_and_format,
            "question": RunnablePassthrough(),
            "history": lambda x: chat_history.messages
        }
        | prompt
        | llm
    )

    return rag_chain, chat_history

# ======================
# 5. 交互界面
def chat_ui():
    print("===== 🚀 优化版RAG知识库（带记忆功能）=====")
    try:
        vs = load_vector_store()
    except:
        path = input("输入文档路径（如study_notes.txt）：")
        split_docs = load_and_split_optimized(path)
        vs = build_vector_store(split_docs)

    chain, history = build_optimized_chain(vs)
    print("✅ 优化版RAG已启动！输入「退出」结束\n")

    while True:
        question = input("你：")
        if question in ["退出", "q"]:
            print("AI：再见！")
            break
        
        # 获取AI回答
        answer = chain.invoke(question)
        
        # 保存对话历史
        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=answer))
        
        print(f"AI：{answer}\n")

if __name__ == "__main__":
    chat_ui()