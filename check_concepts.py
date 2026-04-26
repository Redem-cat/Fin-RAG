import requests

headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://quote.eastmoney.com/'}

for fs in ['m:90+t:3', 'm:90+t:3+f:!50']:
    url = f'https://push2delay.eastmoney.com/api/qt/clist/get?pn=1&pz=200&po=1&np=1&fltt=2&invt=2&fid=f3&fs={fs}&fields=f12,f14'
    r = requests.get(url, timeout=15, headers=headers)
    data = r.json()
    diff = data.get('data', {}).get('diff', [])
    print(f'fs={fs}: {len(diff)} items')
    if diff:
        keywords = ['智驾', '驾驶', '汽车', '机器人', '通信', '电子', '半导体', '芯片', '人工智能', '算力', '无人驾驶']
        found = []
        for item in diff:
            name = item.get('f14', '')
            for kw in keywords:
                if kw in name:
                    found.append((item.get('f12'), name))
                    break
        if found:
            print('  Related:')
            for code, name in found:
                print(f'    {code}: {name}')
        else:
            print('  No related found, first 10:')
            for item in diff[:10]:
                print(f'    {item.get("f12")}: {item.get("f14")}')
    print()
