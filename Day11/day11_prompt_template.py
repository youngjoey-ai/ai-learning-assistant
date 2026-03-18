from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = Tongyi(
    model_name="qwen-turbo",
    temperature=0.3
)

summary_template = PromptTemplate(
    input_variables=["text", "max_length"],
    template="请总结以下文本，要求：1. 不超过{max_length}字 2. 保留核心信息 3. 语言简洁\n文本内容：{text}"
)

if __name__ == "__main__":
    test_text = """
    第11天学习LangChain入门，理解了LangChain的核心价值是封装大模型调用逻辑，
    替代原生requests，还学会了PromptTemplate标准化提示词，后续会用Chain串联逻辑。
    """

    prompt = summary_template.format(
        text=test_text,
        max_length=80
    )
    
    summary = llm.invoke(prompt)
    print("渲染后的Prompt：\n", prompt)
    print("\n总结结果：\n", summary)