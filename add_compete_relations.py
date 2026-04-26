from src.knowledge_graph import get_neo4j_connection

conn = get_neo4j_connection()

# 为同一板块内的公司建立 COMPETES_WITH 关系
query = """
MATCH (s:Sector)<-[:BELONGS_TO]-(c1:Company)
MATCH (s)<-[:BELONGS_TO]-(c2:Company)
WHERE c1.code < c2.code
MERGE (c1)-[r:COMPETES_WITH]->(c2)
SET r.competition_field = s.name, r.source = 'auto_same_sector'
"""
conn.execute_write(query)

# 统计
result = conn.execute_query("MATCH ()-[r:COMPETES_WITH]->() RETURN count(r) as c")
count = result[0]["c"]
print(f"COMPETES_WITH relations created: {count}")

# 查看总关系分布
rels = conn.execute_query("MATCH ()-[r]->() RETURN type(r) as t, count(r) as c ORDER BY c DESC")
print("\nAll relations:")
for r in rels:
    print(f"  {r['t']}: {r['c']}")

# 节点统计
nodes = conn.execute_query("MATCH (n) RETURN labels(n)[0] as label, count(n) as cnt ORDER BY cnt DESC")
print("\nAll nodes:")
for n in nodes:
    print(f"  {n['label']}: {n['cnt']}")
