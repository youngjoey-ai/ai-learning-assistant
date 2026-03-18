import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 通用加载函数（支持TXT/PDF）
def load_document(file_path):
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(file_path):
        file_path = os.path.join(SCRIPT_DIR, file_path)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在：{file_path}")
        return None
    
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load_and_split()
    else:
        print("❌ 仅支持TXT/PDF格式")
        return None
    return docs

# 通用分块函数
def split_document(docs):
    # 按行分割，确保每天独立成块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=70, chunk_overlap=0, separators=["\n"]
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"✅ 分块完成，共 {len(split_docs)} 个块")
    return split_docs

# 初始化向量化模型
def init_embedding():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

# 构建向量库并检索
def rag_retrieve(file_path, query, top_k=3):
    # 1. 加载
    docs = load_document(file_path)
    if not docs:
        return
    # 2. 分块
    split_docs = split_document(docs)
    # 3. 初始化向量化模型
    embedding = init_embedding()
    # 4. 构建向量库
    db = FAISS.from_documents(split_docs, embedding)
    # 5. 检索

    results = db.similarity_search_with_score(query, k=top_k)

    print(f"\n===== 检索结果（和「{query}」最相关的{top_k}个文本块）=====")
    for i, (doc, _) in enumerate(results):
        print(f"\n【结果 {i+1}】")
        print(doc.page_content)
        
# 主函数
if __name__ == "__main__":
    print("===== RAG检索工具 =====")
    file_path = input("请输入文件路径（如study_notes.txt）：")
    query = input("请输入你的问题：")
    top_k = int(input("返回几个相关结果？（默认3）：") or 3)
    
    rag_retrieve(file_path, query, top_k)