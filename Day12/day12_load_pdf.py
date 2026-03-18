import os
from langchain_community.document_loaders import PyPDFLoader  # PDF加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================
# 1. 加载PDF文件
def load_pdf_file(file_path):
    """加载PDF文件（支持多页）"""
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(file_path):
        file_path = os.path.join(SCRIPT_DIR, file_path)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    
    # 初始化PDF加载器
    loader = PyPDFLoader(file_path)
    # 加载所有页面（返回多页Document对象）
    pages = loader.load_and_split()  # 先按页拆分，再做细粒度分块
    # 合并所有页面的文本
    raw_text = "\n".join([page.page_content for page in pages])
    print(f"✅ PDF加载完成，共 {len(pages)} 页，总字符数：{len(raw_text)}")
    return raw_text

# ======================
# 2. 分块（复用之前的分块逻辑）
def split_text(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # PDF文本更密集，适当增大块大小
        chunk_overlap=80,
        separators=["\n", "。", "，", "；"]
    )
    chunks = text_splitter.split_text(raw_text)
    print(f"✅ PDF分块完成，共生成 {len(chunks)} 个文本块")
    # 打印前3个块（避免输出过多）
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n===== 块 {i+1} =====")
        print(chunk)
    return chunks

# ======================
# 主函数
if __name__ == "__main__":
    try:
        raw_text = load_pdf_file("python_study.pdf")
        split_text(raw_text)
    except FileNotFoundError:
        print("❌ 未找到PDF文件，请检查文件路径")
    except Exception as e:
        print(f"❌ 加载失败：{str(e)}")