import requests


class SimpleAIClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def chat(self, message):
        try:
            data = {
                'model': 'qwen-turbo',
                'messages': [{'role': 'user', 'content': message}]
            }
            response = requests.post(self.base_url, json=data, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"请求失败：{e}"


if __name__ == '__main__':
    api_key = 'sk-ce3a7b0513e440e78e9343b5b9adf44c'
    ai_client = SimpleAIClient(api_key)

    result = ai_client.chat("用一句话总结Python类的核心作用")
    print("AI回复：", result)