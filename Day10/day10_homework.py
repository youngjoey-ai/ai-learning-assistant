import requests
import json

API_KEY = "sk-ce3a7b0513e440e78e9343b5b9adf44c"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

def call_qwen_api(prompt):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {
        "model": "qwen-turbo",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(BASE_URL, headers=headers, json=data, timeout=15)
        response.raise_for_status() 
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"调用失败：{str(e)}"

def summarize(text):
    prompt = f"请简洁总结以下文本（不超过100字）：{text}"
    return call_qwen_api(prompt)

def rewrite(text):
    prompt = f"请把以下文本改写成专业的技术描述风格：{text}"
    return call_qwen_api(prompt)

def extract(text):
    prompt = f"从以下文本抽取学习天数和核心技能，返回JSON格式：{text}"
    try:
        result = call_qwen_api(prompt)
        return json.loads(result)
    except:
        return result

def main():
    print("===== AI文本处理工具箱 =====")
    print("1. 文本总结")
    print("2. 文本改写")
    print("3. 信息抽取")
    print("0. 退出")

    while True:
        choice = input("\n请选择功能：")
        if choice == "0":
            print("再见！")
            break
        if choice in ("1", "2", "3"):
            text = input("请输入要处理的文本：")
            if choice == "1":
                print("总结结果：", summarize(text))
            elif choice == "2":
                print("改写结果：", rewrite(text))
            elif choice == "3":
                print("抽取结果：", extract(text))
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main()