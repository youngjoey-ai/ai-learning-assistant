try:
    with open('ai_log.txt', 'r', encoding='utf-8') as f:
        log_content = f.read()      
except FileNotFoundError:
    log_content = "AI学习日志\n==========\n"

today_study = input("请输入今天（第4天）的学习内容：")
new_log = f'第4天：{today_study}\n'

with open('ai_log.txt', 'a', encoding='utf-8') as f:
    f.write(new_log)

with open('ai_log.txt', 'r', encoding='utf-8') as f:
    print("\n最新学习日志：")
    print(f.read())