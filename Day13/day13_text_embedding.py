# 导入核心模块
import os
from langchain_community.embeddings import HuggingFaceEmbeddings  # 开源向量化模型
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================
# 1. 加载并分块文本（复用第12天逻辑）
def load_and_split_text(file_path):
    """加载TXT并分块"""
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(file_path):
        file_path = os.path.join(SCRIPT_DIR, file_path)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    raw_text = docs[0].page_content
    
    # 分块器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n", "。", "，"]
    )
    chunks = text_splitter.split_text(raw_text)
    print(f"✅ 文本分块完成，共 {len(chunks)} 个块")
    return chunks

# ======================
# 2. 初始化向量化模型
def init_embedding_model():
    """初始化开源向量化模型（all-MiniLM-L6-v2）"""
    # 配置模型：轻量级、中文适配、免费
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # 模型名称（自动下载，首次运行稍慢）
        model_kwargs={"device": "cpu"},  # 用CPU运行（新手无需GPU）
        encode_kwargs={"normalize_embeddings": True}  # 归一化向量，提升相似度计算精度
    )
    return embedding_model

# ======================
# 3. 文本块向量化
def embed_text_chunks(chunks, embedding_model):
    """把文本块转换成向量"""
    # 向量化（批量转换）
    embeddings = embedding_model.embed_documents(chunks)
    
    print(f"✅ 向量化完成，共生成 {len(embeddings)} 个向量")
    print(f"📌 每个向量的维度：{len(embeddings[0])}（all-MiniLM-L6-v2固定384维）")
    
    # 打印第一个文本块和对应的向量（前10个值，避免输出过长）
    print("\n===== 示例：第一个文本块 & 向量 =====")
    print(f"文本块：\n{chunks[0]}")
    print(f"向量（前10维）：{embeddings[0][:10]}")
    
    return embeddings

# ======================
# 主函数
if __name__ == "__main__":
    # 1. 加载分块
    chunks = load_and_split_text("study_notes.txt")
    # 2. 初始化向量化模型
    embedding_model = init_embedding_model()
    # 3. 向量化
    embed_text_chunks(chunks, embedding_model)