from langchain_community.llms.tongyi import Tongyi
from dotenv import load_dotenv

load_dotenv()

llm = Tongyi(
    model_name="qwen-turbo",
    temperature=0.7,
    max_tokens=1000
)

if __name__ == "__main__":
    question = "用大白话解释什么是LangChain，适合新手理解"
    answer = llm.invoke(question)
    print("提问：", question)
    print("AI回答：", answer)
