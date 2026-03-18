from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = Tongyi(
    model_name="qwen-turbo",
    temperature=0.3
)

study_template = PromptTemplate(
    input_variables=["day", "question"],
    template="你是一个AI开发学习导师，请针对第{day}天的学习内容，回答以下问题，要求：1. 面向新手 2. 简洁明了 3. 结合实战\n问题：{question}"
)

study_chain = study_template | llm

if __name__ == "__main__":
    print("===== AI学习问答助手 =====")
    while True:
        day = input("\n请输入学习天数（如11）：")
        question = input("请输入你的问题：")
        if day == "0" or question == "退出":
            print("再见！")
            break
        answer = study_chain.invoke({"day": day, "question": question})
        print("\n✅ 回答：\n", answer)