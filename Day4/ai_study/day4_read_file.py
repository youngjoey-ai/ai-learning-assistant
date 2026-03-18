with open('test.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    print("文件全部内容：")
    print(content)

print('\n按行读取：')
with open('test.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        print('行内容：',line.strip())