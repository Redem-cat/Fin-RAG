"""
获取全部 A 股列表（约 5800+ 家）
东方财富分页接口，每页 100 条
"""
import requests
import json

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://quote.eastmoney.com/",
}

all_stocks = []
page = 1
max_pages = 60  # 5800 / 100 = 58 pages

while page <= max_pages:
    url = (
        f"https://push2delay.eastmoney.com/api/qt/clist/get"
        f"?pn={page}&pz=100&po=1&np=1&fltt=2&invt=2&fid=f12"
        f"&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
        f"&fields=f12,f14,f20,f21,f33,f43,f44,f46,f47,f48,f49,f50"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        data = r.json()
        diff = data.get("data", {}).get("diff", [])
        total = data.get("data", {}).get("total", 0)
        if not diff:
            break
        for item in diff:
            all_stocks.append({
                "code": str(item.get("f12", "")),
                "name": item.get("f14", ""),
                "market_cap": item.get("f20", 0),
                "circulating_cap": item.get("f21", 0),
                "pe_ttm": item.get("f33", 0),
                "pb": item.get("f46", 0),
                "turnover": item.get("f47", 0),
                "revenue": item.get("f48", 0),
                "profit": item.get("f50", 0),
            })
        print(f"Page {page}/{max_pages}: +{len(diff)} stocks, total={len(all_stocks)}, api_total={total}")
        if len(diff) < 100:
            break
        page += 1
    except Exception as e:
        print(f"Error on page {page}: {e}")
        break

print(f"\nTotal stocks fetched: {len(all_stocks)}")
with open("data/all_stocks.json", "w", encoding="utf-8") as f:
    json.dump(all_stocks, f, ensure_ascii=False, indent=2)
print("Saved to data/all_stocks.json")
