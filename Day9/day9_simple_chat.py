import requests


API_KEY = "sk-ce3a7b0513e440e78e9343b5b9adf44c"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

def qwen_chat(message):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        'model': 'qwen-turbo',
        'messages': [{'role': 'user', 'content': message}],
        'temperature': 0.7,
        'max_tokens': 1000
    }

    try:
        response = requests.post(BASE_URL, headers=headers, json=data, timeout=15)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.Timeout:
        return "❌ 调用超时：AI服务器响应太慢，请稍后再试"
    except requests.HTTPError as e:
        return f"❌ HTTP错误：{e}，请检查API Key是否正确"
    except KeyError:
        return f"❌ 响应解析失败：{response.text}"
    except Exception as e:
        return f"❌ 未知错误：{str(e)}"

if __name__ == "__main__":
    print("===== 通义千问AI聊天 =====")
    while True:
        user_input = input("\n你: ")
        if user_input in ["退出", "q", "Q"]:
            print("AI：再见！")
            break
        response = qwen_chat(user_input)
        print(f"AI：{response}")