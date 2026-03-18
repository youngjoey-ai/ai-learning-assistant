from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = Tongyi(
    model_name="qwen-turbo",
    temperature=0.5
)

rewrite_template = PromptTemplate(
    input_variables=["text", "style"],
    template="请把以下文本改写成{style}风格，保留核心信息，语言流畅：\n{text}"
)

rewrite_chain = rewrite_template | llm

if __name__ == "__main__":
    result = rewrite_chain.invoke({
        "text": "我学了11天Python和大模型，现在能用LangChain调用AI了",
        "style": "专业技术文档"
    })
    
    print("\n改写结果：\n", result)