from src.knowledge_graph import get_neo4j_connection
conn = get_neo4j_connection()

print("=== 节点分布 ===")
for label in ['Company','Sector','Person','Event','Asset']:
    res = conn.execute_query(f"MATCH (n:{label}) RETURN n.name as name LIMIT 5")
    names = [r["name"] for r in res]
    cnt = conn.execute_query(f"MATCH (n:{label}) RETURN count(n) as c")
    total = cnt[0]["c"] if cnt else 0
    print(f"{label} ({total}): {names}")

print("\n=== 关系分布 ===")
res = conn.execute_query("MATCH ()-[r]->() RETURN type(r) as t, count(r) as c")
if res:
    for r in res:
        print(f"  {r['t']}: {r['c']}")
else:
    print("  无关系")

print("\n=== 最近创建的公司 ===")
res = conn.execute_query("MATCH (c:Company) RETURN c.name as name, c.code as code ORDER BY c.updated_at DESC LIMIT 10")
for r in res:
    print(f"  {r['name']} ({r['code']})")

print("\n=== 最近创建的行业 ===")
res = conn.execute_query("MATCH (s:Sector) RETURN s.name as name ORDER BY s.updated_at DESC LIMIT 10")
for r in res:
    print(f"  {r['name']}")
