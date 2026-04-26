import requests

url = (
    "https://push2delay.eastmoney.com/api/qt/clist/get"
    "?pn=1&pz=200&po=1&np=1&fltt=2&invt=2&fid=f3"
    "&fs=m:90+t:2+f:!50&fields=f12,f14"
)
headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://quote.eastmoney.com/",
}
r = requests.get(url, headers=headers, timeout=30)
data = r.json()
diff = data.get("data", {}).get("diff", [])

# 更宽泛的关键词
keywords = [
    "智驾", "驾驶", "汽车", "整车", "零部件",
    "机器人", "自动化", "智能",
    "通信", "电信", "网络",
    "电子", "半导体", "芯片", "集成电路", "分立器件",
    "人工智能", "AI", "算力"
]
print("=== 相关板块 ===")
for item in diff:
    name = item.get("f14", "")
    code = item.get("f12", "")
    for kw in keywords:
        if kw in name:
            print(f"  {code}: {name}")
            break

print("\n=== 全部板块（共{}个）===".format(len(diff)))
for item in diff:
    print(f"  {item.get('f12')}: {item.get('f14')}")
