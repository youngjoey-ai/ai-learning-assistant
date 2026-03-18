import requests

API_KEY = "sk-ce3a7b0513e440e78e9343b5b9adf44c"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

def ai_rewrite(text, style="正式", requirement=""):
    """
    AI文本改写
    :param text: 原文本
    :param style: 改写风格（正式/口语化/专业/简洁）
    :param requirement: 额外要求
    :return: 改写后的文本
    """
    prompt = f"""请按照以下要求改写文本：
1. 改写风格：{style}
2. 额外要求：{requirement}
3. 保留原文本核心信息，不增不减
4. 语言流畅，符合指定风格

原文本：
{text}
"""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "qwen-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,  # 改写类任务适度增加随机性
    }

    try:
        response = requests.post(BASE_URL, json=data, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"改写失败：{str(e)}"

# 测试改写功能
if __name__ == "__main__":
    test_text = "我花了3天学Python基础，现在能调用AI API做聊天程序了"

    # 1. 改写成正式风格
    print("===== 正式风格 =====")
    formal_text = ai_rewrite(test_text, "正式", "突出学习效率和成果")
    print(formal_text)

    # 2. 改写成口语化风格
    print("\n===== 口语化风格 =====")
    casual_text = ai_rewrite(test_text, "口语化", "像和朋友聊天一样")
    print(casual_text)