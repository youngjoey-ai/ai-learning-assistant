try:
    a = float(input("输入第一个数："))
    b = float(input("输入第二个数："))
    print("结果：", a + b)
except ValueError:
    print("请输入有效的数字")
except Exception as e:
    print(f"发生未知错误:{e}")
else:
    print("计算成功！")
finally:
    print("程序结束")
    