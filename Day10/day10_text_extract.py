import requests
import json  # 用于解析JSON结果

API_KEY = "sk-ce3a7b0513e440e78e9343b5b9adf44c"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

def ai_extract_info(text):
    """
    抽取文本中的学习信息，返回JSON格式
    """
    # 强制要求AI返回JSON，是结构化抽取的关键
    prompt = f"""请从以下文本中抽取学习信息，严格按照指定JSON格式返回，不要加任何额外内容：
JSON格式：
{{
    "total_days": 数字,  // 总学习天数
    "core_skills": ["技能1", "技能2"],  // 学会的核心技能
    "projects": ["项目1"],  // 完成的项目
    "stage": "阶段名称"  // 学习阶段（如Python基础/大模型API）
}}

文本内容：
{text}
"""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "qwen-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # 结构化抽取用极低温度，保证格式准确
    }

    try:
        response = requests.post(BASE_URL, json=data, headers=headers, timeout=15)
        response.raise_for_status()
        result_text = response.json()["choices"][0]["message"]["content"]
        
        # 解析JSON字符串为Python字典
        extract_result = json.loads(result_text)
        return extract_result
    except json.JSONDecodeError:
        return f"JSON解析失败，AI返回内容：{result_text}"
    except Exception as e:
        return f"抽取失败：{str(e)}"

# 测试信息抽取
if __name__ == "__main__":
    test_text = """
我用7天学完了Python基础，掌握了类、函数、文件操作、异常处理等技能，
完成了AI学习助手小工具。第8-9天学习大模型API调用，实现了带记忆的AI聊天程序，
现在处于大模型应用开发的入门阶段。
    """

    print("===== 结构化信息抽取 =====")
    extract_data = ai_extract_info(test_text)
    # 格式化输出JSON，更易读
    print(json.dumps(extract_data, ensure_ascii=False, indent=2))