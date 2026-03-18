import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader  # PDF加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_document(file_path):
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(file_path):
        file_path = os.path.join(SCRIPT_DIR, file_path)
    # 检查文件是否存在
    if not os.path.exists(file_path):
        return False, f"文件不存在：{file_path}"
    
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        raw_text = docs[0].page_content
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        raw_text = "\n".join([page.page_content for page in pages])
    else:
        return False, "仅支持TXT和PDF格式"
    return True, raw_text

def split_document(raw_text, chunk_size=500, chunk_overlap=80):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n', '。', '，']
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def main():
    print("===== 文档加载分块工具 =====")
    file_path = input("请输入文档路径（支持TXT和PDF）：")
    
    success, raw_text = load_document(file_path)
    if not success:
        print(f"❌ 加载文档失败：{raw_text}")
        return  
    
    chunks = split_document(raw_text)
    print(f"\n✅ 处理完成！")
    print(f"原始文本长度：{len(raw_text)} 字符")
    print(f"生成文本块数量：{len(chunks)}")

    show_chunks = input("是否打印所有文本块？(y/n)：")
    if show_chunks.lower() == 'y':
        for i, chunk in enumerate(chunks):
            print(f"\n===== 块 {i+1} =====")
            print(chunk)

if __name__ == "__main__":
    main()