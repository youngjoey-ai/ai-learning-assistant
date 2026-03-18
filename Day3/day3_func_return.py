def add_num(a, b):
    """计算两个数的和，返回结果"""
    result = a + b
    return result

sum1 = add_num(5, 8)
sum2 = add_num(10.5, 20.3)

print('5+8 = ', sum1)
print('10.5+20.3 = ', sum2)

def process_text(text):
    new_text = text.strip().lower()
    return new_text

raw_text = '  我要学AI开发  '
clean_text = process_text(raw_text)
print("处理后的文本:", clean_text)