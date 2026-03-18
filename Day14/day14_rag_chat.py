from day14_rag_core import load_and_split_document, build_vector_store, load_vector_store, build_rag_chain

def main():
    print("===== 本地知识库RAG问答助手 =====")
    # 初始化向量库
    try:
        # 先尝试加载本地已有的向量库
        vector_store = load_vector_store()
    except:
        # 没有本地向量库，就新建
        file_path = input("请输入知识库文件路径（TXT/PDF）：")
        split_docs = load_and_split_document(file_path)
        vector_store = build_vector_store(split_docs)
    
    # 构建RAG问答链
    rag_chain = build_rag_chain(vector_store)
    print("\n✅ 知识库加载完成！可以开始提问，输入「退出」结束对话")

    # 交互式对话
    while True:
        user_question = input("\n你：")
        if user_question in ["退出", "q", "Q"]:
            print("AI：再见！")
            break
        # 调用RAG链
        answer = rag_chain.invoke(user_question)
        print(f"AI：{answer}")

if __name__ == "__main__":
    main()