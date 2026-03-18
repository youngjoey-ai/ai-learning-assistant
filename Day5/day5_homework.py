import requests


def call_ai_api(url, api_key):
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'prompt': '你好，我在学习AI开发',
            'max_tokens': 100
        }
        response = requests.post(url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return "请求超时，请稍后重试"
    except requests.exceptions.HTTPError as e:
        return f"HTTP错误：{e}"
    except Exception as e:
        return f"未知错误：{e}"

test_url = "https://test-api.ai.com/chat"
test_key = "your_test_api_key"
result = call_ai_api(test_url, test_key)
print("API调用结果：", result)