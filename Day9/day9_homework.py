import requests


class QwenChatBot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self._conversation_history = []

    def qwen_chat_with_memory(self, message):
        self._conversation_history.append({'role': 'user', 'content': message})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            'model': 'qwen-turbo',
            'messages': self._conversation_history + [{'role': 'user', 'content': message}],
            'temperature': 0.7,
            'max_tokens': 1000
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=15)
            response.raise_for_status()
            result = response.json()
            ai_answer = result['choices'][0]['message']['content']

            self._conversation_history.append({'role': 'assistant', 'content': ai_answer})

            return ai_answer
        except requests.Timeout:
            return "❌ 调用超时：AI服务器响应太慢，请稍后再试"
        except requests.HTTPError as e:
            return f"❌ HTTP错误：{e}，请检查API Key是否正确"
        except KeyError:
            return f"❌ 响应解析失败：{response.text}"
        except Exception as e:
            return f"❌ 未知错误：{str(e)}"

    def clear_history(self):
        self._conversation_history = []

    def get_history(self):
        return self._conversation_history.copy()

if __name__ == "__main__":
    chat_bot = QwenChatBot("sk-ce3a7b0513e440e78e9343b5b9adf44c")

    print("===== 带记忆的AI聊天 =====")
    print("提示：输入“退出”结束对话\n")

    while True:
        user_input = input("你: ")
        if user_input in ["退出", "q", "Q"]:
            print("AI：再见！")
            break
        response = chat_bot.qwen_chat_with_memory(user_input)
        print(f"AI：{response}")