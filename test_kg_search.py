from src.knowledge_graph import get_kg_retriever, get_neo4j_connection

print("=== 测试模糊搜索 ===\n")

retriever = get_kg_retriever()

# 测试 1: 搜公司名的一部分
print("1. 搜索 '中芯' (应命中 '中芯国际'):")
result = retriever.query("中芯的关系有哪些？", query_type="related")
print(f"   找到 {len(result.entities)} 个实体, {len(result.relations)} 个关系")
if result.entities:
    print(f"   示例: {result.entities[0]}")

# 测试 2: 搜板块
print("\n2. 搜索 'AI芯片' 板块内的公司:")
conn = get_neo4j_connection()
res = conn.execute_query("""
    MATCH (s:Sector)<-[:BELONGS_TO]-(c:Company)
    WHERE toLower(s.name) CONTAINS toLower('AI芯片')
    RETURN c.name as name LIMIT 5
""")
for r in res:
    print(f"   - {r['name']}")

# 测试 3: 关系查询
print("\n3. 搜索 'TCL' 的关系:")
res = conn.execute_query("""
    MATCH (c:Company)-[r:BELONGS_TO]->(s:Sector)
    WHERE toLower(c.name) CONTAINS toLower('TCL')
    RETURN c.name as company, s.name as sector
""")
for r in res:
    print(f"   - {r['company']} -> {r['sector']}")

print("\n=== 测试完成 ===")
