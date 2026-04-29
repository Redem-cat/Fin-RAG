"""快速检查 Neo4j KG 数据状态"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from neo4j import GraphDatabase

driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
s = driver.session()

print("=== Nodes ===")
nodes = s.run('MATCH (n) RETURN labels(n)[0] as type, count(n) as cnt ORDER BY cnt DESC')
total_n = 0
for r in nodes:
    print(f"  {r['type']}: {r['cnt']}")
    total_n += r['cnt']
print(f"  TOTAL: {total_n}")

print("\n=== Relations ===")
rels = s.run('MATCH ()-[r]->() RETURN type(r) as type, count(r) as cnt ORDER BY cnt DESC')
total_r = 0
for r in rels:
    print(f"  {r['type']}: {r['cnt']}")
    total_r += r['cnt']
print(f"  TOTAL: {total_r}")

if total_n == 0:
    print("\n[!] 图谱为空，需要运行数据导入任务")

s.close()
driver.close()
