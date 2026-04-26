"""
定向导入：围绕智驾/机器人/通信/电子建立领域知识图谱

板块组合：
- 汽车芯片 (BK0969)
- 电子后视镜 (BK1125)
- 汽车一体化压铸 (BK1093)
- 机器人执行器 (BK1145)
- AI芯片 (BK1127)
- 存储芯片 (BK1137)
- 半导体概念 (BK0917)
- 第四代半导体 (BK1121)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
from src.knowledge_graph import get_neo4j_connection
from src.kg_builder.kg_writer import get_kg_writer

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://quote.eastmoney.com/",
}

FOCUS_SECTORS = {
    "BK0969": "汽车芯片",
    "BK1125": "电子后视镜",
    "BK1093": "汽车一体化压铸",
    "BK1145": "机器人执行器",
    "BK1127": "AI芯片",
    "BK1137": "存储芯片",
    "BK0917": "半导体概念",
    "BK1121": "第四代半导体",
}


def fetch_sector_companies(sector_code: str):
    """获取板块成分股"""
    url = (
        f"https://push2delay.eastmoney.com/api/qt/clist/get"
        f"?pn=1&pz=200&po=1&np=1&fltt=2&invt=2&fid=f12"
        f"&fs=b:{sector_code}&fields=f12,f14,f20,f21"
    )
    r = requests.get(url, headers=HEADERS, timeout=30)
    data = r.json()
    diff = data.get("data", {}).get("diff", [])
    companies = []
    for item in diff:
        companies.append({
            "code": str(item.get("f12", "")),
            "name": item.get("f14", ""),
            "market_cap": item.get("f20", 0),
        })
    return companies


def main():
    print("=" * 60)
    print("定向导入：智驾/机器人/通信/电子 领域知识图谱")
    print("=" * 60)

    conn = get_neo4j_connection()
    writer = get_kg_writer()

    # 1. 清理旧数据
    print("\n[1/3] 清理旧数据...")
    conn.execute_write("MATCH (c:Company) DETACH DELETE c")
    conn.execute_write("MATCH (s:Sector) DETACH DELETE s")
    print("  已删除旧 Company 和 Sector 节点")

    # 2. 导入板块和公司
    print("\n[2/3] 导入板块和公司...")
    total_companies = 0
    for sector_code, sector_name in FOCUS_SECTORS.items():
        companies = fetch_sector_companies(sector_code)
        print(f"  [{sector_name}] 获取到 {len(companies)} 家公司")

        # 写入板块节点
        writer.write_sector({
            "name": sector_name,
            "code": sector_code,
            "description": f"东方财富概念板块 {sector_code}"
        })

        # 写入公司节点 + 建立 BELONGS_TO 关系
        for c in companies:
            writer.write_company({
                "code": c["code"],
                "name": c["name"],
                "sector": sector_name,
                "market_cap": c["market_cap"],
            })
            # 建立关系
            conn.execute_write("""
                MATCH (c:Company {code: $code})
                MATCH (s:Sector {name: $sector})
                MERGE (c)-[:BELONGS_TO]->(s)
            """, {"code": c["code"], "sector": sector_name})

        total_companies += len(companies)

    print(f"\n  总计导入 {len(FOCUS_SECTORS)} 个板块, {total_companies} 家公司")

    # 3. 验证结果
    print("\n[3/3] 验证结果...")
    node_res = conn.execute_query("MATCH (n) RETURN labels(n)[0] as label, count(n) as cnt")
    for r in node_res:
        print(f"  {r['label']}: {r['cnt']} 个")

    rel_res = conn.execute_query("MATCH ()-[r]->() RETURN type(r) as t, count(r) as c")
    if rel_res:
        for r in rel_res:
            print(f"  关系 {r['t']}: {r['c']} 条")
    else:
        print("  关系: 0 条")

    # 示例查询
    print("\n  示例公司（AI芯片板块）:")
    sample = conn.execute_query("""
        MATCH (c:Company)-[:BELONGS_TO]->(s:Sector {name: 'AI芯片'})
        RETURN c.name as name, c.code as code LIMIT 5
    """)
    for r in sample:
        print(f"    {r['name']} ({r['code']})")

    print("\n" + "=" * 60)
    print("导入完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
