# 导入核心模块
import os
from langchain_community.document_loaders import TextLoader  # TXT加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 通用分块器

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================
# 1. 加载TXT文件
def load_txt_file(file_path):
    """加载TXT文件"""
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(file_path):
        file_path = os.path.join(SCRIPT_DIR, file_path)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    
    # 初始化加载器，指定编码解决中文乱码
    loader = TextLoader(file_path, encoding="utf-8")
    # 加载文件，返回Document对象列表（每个Document包含page_content和metadata）
    documents = loader.load()
    # 提取原始文本
    raw_text = documents[0].page_content
    print(f"✅ 加载完成，原始文本长度：{len(raw_text)} 字符")
    return raw_text

# ======================
# 2. 文本分块
def split_txt(raw_text):
    """分块长文本"""
    # 初始化分块器（最常用的通用分块器）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,          # 每个块的最大字符数（根据模型窗口调整）
        chunk_overlap=50,        # 块之间的重叠字符数（保持上下文连贯）
        length_function=len,     # 长度计算方式（按字符数）
        separators=["\n", "。", "，"]  # 按中文分隔符拆分，更符合中文习惯
    )
    # 分块
    chunks = text_splitter.split_text(raw_text)
    print(f"✅ 分块完成，共生成 {len(chunks)} 个文本块")
    # 打印每个块的内容
    for i, chunk in enumerate(chunks, start=1):
        print(f"\n===== 块 {i} =====")
        print(chunk)
    return chunks

# ======================
# 主函数
if __name__ == "__main__":
    # 加载文件（使用相对路径）
    raw_text = load_txt_file("study_notes.txt")
    # 分块
    split_txt(raw_text)