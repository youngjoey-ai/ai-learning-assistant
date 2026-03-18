with open('test.txt', 'w', encoding='utf-8') as f:
    f.write('这是AI学习的笔记\n')
    f.write('第4天：学会文件操作\n')

with open('test.txt', 'a', encoding='utf-8') as f:
    f.write('追加的内容：文件操作是AI项目的基础\n')

with open('test.txt', 'r', encoding='utf-8') as f:
    print("写入后的文件内容：")
    print(f.read())