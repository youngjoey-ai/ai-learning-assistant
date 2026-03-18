import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # LangChain封装的FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================
# 1. 加载分块 + 初始化向量化模型（复用之前的逻辑）
def prepare_data(file_path):
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(file_path):
        file_path = os.path.join(SCRIPT_DIR, file_path)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    
    # 加载分块
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50, separators=["\n", "。", "，"]
    )
    split_docs = text_splitter.split_documents(docs)  # 注意：这里用split_documents，返回Document对象
    
    # 初始化向量化模型
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return split_docs, embedding_model

# ======================
# 2. 构建向量库并检索
def build_vector_db_and_retrieve(query):
    """构建FAISS向量库，检索和问题最相关的文本块"""
    # 准备数据
    split_docs, embedding_model = prepare_data("study_notes.txt")
    
    # 构建FAISS向量库（文本块 → 向量 → 存入库）
    db = FAISS.from_documents(split_docs, embedding_model)
    print("✅ FAISS向量库构建完成")
    
    # 相似度检索：返回最相关的3个文本块（带真实分数）
    results = db.similarity_search_with_score(query, k=3)
    
    # 打印检索结果
    print(f"\n===== 检索结果（和「{query}」最相关的3个文本块）=====")
    for i, (doc, score) in enumerate(results):
        # FAISS默认返回L2距离，转换为余弦相似度（0-1范围，越大越相似）
        cosine_similarity = 1 / (1 + score)
        print(f"\n【相关块 {i+1}】")
        print(f"内容：{doc.page_content}")
        print(f"相似度：{round(cosine_similarity, 4)} (余弦相似度)")
    
    return [doc for doc, _ in results]

# ======================
# 主函数
if __name__ == "__main__":
    # 测试检索（用户问题）
    user_query = "我想知道第11天学了什么？"
    build_vector_db_and_retrieve(user_query)
    
    # 再测试一个问题
    print("\n" + "="*50)
    user_query2 = "LangChain能解决什么问题？"
    build_vector_db_and_retrieve(user_query2)