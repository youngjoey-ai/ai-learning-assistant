import json


data = {
    'study_day': 4,
    'skill': '文件操作',
    'goals': ['读取文件', '写入文件', '处理json'],
    'process': '已完成'
}

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

with open("data.json", "r", encoding="utf-8") as f:
    loaded_data = json.load(f)
    print("读取的JSON数据：")
    print(f"第{loaded_data['study_day']}天，学习了{loaded_data['skill']}")
    print(f"学习目标：{loaded_data['goals']}")
