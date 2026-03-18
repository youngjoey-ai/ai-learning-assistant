from pathlib import Path


class AILogger:
    def __init__(self, log_file='ai_study_log.txt'):
        script_dir = Path(__file__).parent.absolute()
        self.log_file = script_dir / log_file
        self._init_log_file()
    
    def _init_log_file(self):
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                pass
        except FileNotFoundError:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("AI学习日志\n==========\n")

    def add_log(self, day, content):
        log_line = f"第{day}天：{content}\n"
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_line)
            print(f"✅ 第{day}天日志添加成功")
        except Exception as e:
            print(f"❌ 日志添加失败：{e}")

    def show_log(self):
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"\n【全部学习日志】\n{content}")
        except Exception as e:
            print(f"❌ 日志读取失败：{e}")


if __name__ == '__main__':
    logger = AILogger()
    logger.add_log(6, "学会了Python类的封装，能写AI工具类了")
    logger.show_log()