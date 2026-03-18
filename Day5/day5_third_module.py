import requests


try:
    response = requests.get("https://www.baidu.com")
    # 状态码200表示成功
    if response.status_code == 200:
        print("✅ 网络请求成功！")
        # 打印响应内容的前100个字符
        print("响应内容：", response.text[:100])
except requests.exceptions.RequestException as e:
    print(f"❌ 网络请求失败：{e}")