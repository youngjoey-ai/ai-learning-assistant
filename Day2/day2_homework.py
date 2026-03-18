info = {
    'name': '我',
    'target': 'AI开发上岸',
    'study_day': 2,
    'skills': ['python','列表','字典','字符串']
}

print('我是：', info['name'])
print('今天第', info['study_day'],'天')

for skill in info['skills']:
    print('正在学：', skill)