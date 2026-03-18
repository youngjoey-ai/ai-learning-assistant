# 导入核心模块
import os
from dotenv import load_dotenv
# 文档处理模块（第12天内容）
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 向量化与向量库模块（第13天内容）
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
# LLM与Chain模块（第11天内容）
from langchain_community.llms import Tongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================
# 1. 加载环境变量，初始化配置
load_dotenv()  # 读取.env文件中的API Key
API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 检查API Key是否设置
if not API_KEY:
    raise ValueError("DASHSCOPE_API_KEY 未设置，请在.env文件中添加或设置环境变量")

# 设置环境变量供DashScopeEmbeddings使用
os.environ["DASHSCOPE_API_KEY"] = API_KEY

# 初始化向量化模型（通义千问开源embedding，免费额度充足）
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2"
)

# 初始化LLM大模型
llm = Tongyi(
    model="qwen-turbo",
    temperature=0.3,  # RAG场景用低温度，保证回答严谨，减少幻觉
    max_tokens=1500
)

# ======================
# 2. 文档加载与分块（复用第12天的通用函数）
def load_and_split_document(file_path):
    """加载文档并分块，返回分块后的文本列表"""
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(file_path):
        file_path = os.path.join(SCRIPT_DIR, file_path)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    
    # 1. 加载文档
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load_and_split()
    else:
        raise Exception("仅支持TXT和PDF格式文件")
    
    # 2. 文本分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=0,
        separators=["\n", "。", "，"],
        length_function=len
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"✅ 文档处理完成，共生成 {len(split_docs)} 个文本块")
    return split_docs

# ======================
# 3. 构建向量库（存入本地，可重复使用）
def build_vector_store(split_docs, store_path="./rag_vector_store"):
    """基于分块文档构建FAISS向量库，并存入本地"""
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(store_path):
        store_path = os.path.join(SCRIPT_DIR, store_path)
    
    # 构建向量库
    vector_store = FAISS.from_documents(split_docs, embeddings)
    # 保存到本地，下次不用重新构建
    vector_store.save_local(store_path)
    print(f"✅ 向量库构建完成，已保存到 {store_path}")
    return vector_store

# ======================
# 4. 加载本地向量库（重复使用时无需重新处理文档）
def load_vector_store(store_path="./rag_vector_store"):
    """加载本地已保存的向量库"""
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(store_path):
        store_path = os.path.join(SCRIPT_DIR, store_path)
    
    try:
        vector_store = FAISS.load_local(
            store_path, 
            embeddings,
            allow_dangerous_deserialization=True  # 允许加载本地文件，本地使用无风险
        )
        print(f"✅ 向量库加载成功")
        return vector_store
    except Exception as e:
        raise Exception(f"向量库加载失败：{str(e)}，请先构建向量库")

# ======================
# 5. 构建RAG问答链（核心：检索+生成）
def build_rag_chain(vector_store):
    """构建RAG检索问答链（使用管道操作符）"""
    # 构建检索器：从向量库中检索Top3最相关的文本块
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 定义格式化文档的函数
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # 核心Prompt模板：强制AI基于检索内容回答，避免幻觉
    template = """你是一个严谨的知识库问答助手，必须严格基于以下参考文档回答用户的问题，禁止编造文档中不存在的内容。
如果参考文档中没有相关内容，请直接回答："抱歉，参考文档中没有找到相关内容，无法为您解答。"

参考文档：
{context}

用户问题：
{question}

回答："""
    prompt = ChatPromptTemplate.from_template(template)
    
    # 构建RAG链（使用管道操作符 |）
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}  # 准备输入
        | prompt  # 构建prompt
        | llm     # 调用大模型
    )
    
    return rag_chain

# ======================
# 主程序：测试RAG全流程
if __name__ == "__main__":
    # 1. 处理文档，构建向量库（第一次运行执行，后续可注释掉，直接加载本地向量库）
    # study_notes_path = os.path.join(SCRIPT_DIR, "study_notes.txt")
    # split_docs = load_and_split_document(study_notes_path)  # 用你第12天的学习笔记文件
    # vector_store = build_vector_store(split_docs)

    # 2. 加载本地向量库（后续重复使用时，注释上面2行，取消下面这行注释即可）
    vector_store = load_vector_store()

    # 3. 构建RAG问答链
    rag_chain = build_rag_chain(vector_store)

    # 4. 测试问答
    print("\n===== RAG知识库问答测试 =====")
    test_question = "我第7天完成了什么项目？"
    print(f"问题：{test_question}")

    # 调用RAG链获取回答
    answer = rag_chain.invoke(test_question)
    print(f"\nAI回答：{answer}")

    # 重新检索获取相关文档用于展示
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    source_documents = retriever.invoke(test_question)
    
    # 打印检索到的源文档，验证回答是否基于文档
    print("\n===== 检索到的参考文档 =====")
    for i, doc in enumerate(source_documents):
        print(f"\n--- 参考文档 {i+1} ---")
        print(doc.page_content)