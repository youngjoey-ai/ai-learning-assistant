import requests

# 基础配置（复用第9天）
API_KEY = "sk-ce3a7b0513e440e78e9343b5b9adf44c"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

def ai_summarize(text, summary_type="简洁版", max_length=100):
    """
    AI文本总结
    :param text: 要总结的文本
    :param summary_type: 总结类型（简洁版/详细版/要点版）
    :param max_length: 最大字数
    :return: 总结结果
    """
    # 精准的Prompt是关键！
    prompt = f"""请按照以下要求总结文本：
1. 总结类型：{summary_type}
2. 最大字数：{max_length}字
3. 保留核心信息，去掉无关内容
4. 语言流畅，逻辑清晰

需要总结的文本：
{text}
"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "qwen-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,  # 总结类任务用低温度，更严谨
        "max_tokens": 200
    }

    try:
        response = requests.post(BASE_URL, json=data, headers=headers, timeout=15)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"总结失败：{str(e)}"

# 测试总结功能
if __name__ == "__main__":
    # 测试文本（你的学习日志）
    test_text = """
第7天：完成Python基础阶段复盘，整合前6天知识点，做出了AI学习助手小工具。
这个工具包含日志管理和AI问答功能，用了类、函数、异常处理、API调用等知识点，
是第一个可演示的完整小项目。第8天学习了大模型核心概念，包括Prompt、API、RAG、Agent，
理解了RAG的完整流程，难度中等。第9天实现了调用通义千问API的AI聊天程序，
还完成了带记忆的多轮对话功能。
    """

    # 1. 简洁版总结
    print("===== 简洁版总结 =====")
    short_summary = ai_summarize(test_text, "简洁版", 80)
    print(short_summary)

    # 2. 要点版总结
    print("\n===== 要点版总结 =====")
    point_summary = ai_summarize(test_text, "要点版", 150)
    print(point_summary)