from cgitb import text
from email import message
import requests
import datetime


class QwenAIClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _call_api(self, messages):
        try:
            data = {
                'model': 'qwen-turbo',
                'messages': messages,
                'temperature': 0.7
            }
            response = requests.post(self.base_url, headers=self.headers, json=data, timeout=15)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.Timeout:
            return "❌ 调用超时，请检查网络"
        except requests.HTTPError as e:
            return f"❌ API错误：{e}"
        except Exception as e:
            return f"❌ 未知错误：{e}"
        
    def summarize(self, text):
        prompt = f'''请总结以下学习日志，要求：
1. 简洁明了，不超过100字
2. 突出学习的核心技能
3. 语气积极

日志内容：
{text}
'''
        messages = [{"role": "user", "content": prompt}]
        return self._call_api(messages)

    def aks_ai(self, question):
        messages = [{"role": "user", "content": question}]
        return self._call_api(messages)


class StudyLogger:
    def __init__(self, log_file='ai_study_log.txt'):
        self.log_file = log_file
        self._init_file()

    def _init_file(self):
        try:
            with open(self.log_file, 'r', encoding='utf-8'):
                pass
        except FileNotFoundError:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"AI学习日志｜创建时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write('='*50 + '\n')

    def add_log(self, day, content):
        log_line = f"【第{day}天】{datetime.datetime.now().strftime('%Y-%m-%d')}:{content}\n"
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_line)
            return True, f'日志添加成功：{log_line.strip()}'
        except Exception as e:
            return False, f"添加失败：{str(e)}"

    def read_log(self):
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            return True, content
        except Exception as e:
            return False, f"读取失败：{str(e)}"


def main():
    API_KEY = "sk-ce3a7b0513e440e78e9343b5b9adf44c"

    ai_client = QwenAIClient(API_KEY)
    logger = StudyLogger()

    print("""===== AI学习助手 =====
1. 添加学习日志
2. 查看所有日志
3. AI总结日志
4. AI学习答疑
0. 退出""")
    print("="*20)

    while True:
        choice = input('\n请选择功能（0-4）：')
        if choice == '0':
            print("👋 再见！")
            break
        elif choice == '1':
            day = input("输入学习的天数（如第1天）：")
            content = input("输入学习内容：")
            success, msg = logger.add_log(day, content)
            print(msg)
        elif choice == '2':
            success, msg = logger.read_log()
            print(f"\n📝 学习日志：{msg}" if success else msg)
        elif choice == '3':
            success, logs = logger.read_log()
            if success:
                print("🤖 AI正在总结...")
                summary = ai_client.summarize(logs)
                print("\n✨ 日志总结：")
                print(summary)
            else:
                print(logs)
        elif choice == '4':
            question = input("❓ 输入你的学习问题：")
            print("🤖 AI正在回答...")
            answer = ai_client.aks_ai(question)
            print("\n💡 AI回答：")
            print(answer)
        else:
            print("❌ 输入错误，请选择0-4")


if __name__ == "__main__":
    main()