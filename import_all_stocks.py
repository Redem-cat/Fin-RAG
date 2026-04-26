"""
全量导入 5800+ A 股公司到 Neo4j
同时创建行业节点和 BELONGS_TO 关系
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.knowledge_graph import get_neo4j_connection

print("Loading stock data...")
with open("data/all_stocks.json", "r", encoding="utf-8") as f:
    stocks = json.load(f)

# 尝试加载行业映射
try:
    with open("data/stock_industry_map.json", "r", encoding="utf-8") as f:
        industry_map = json.load(f)
except:
    industry_map = {}

print(f"Stocks: {len(stocks)}, Industry map: {len(industry_map)}")

conn = get_neo4j_connection()

# 1. 清理旧 Company 数据（保留其他类型节点）
print("\nCleaning old Company nodes...")
conn.execute_write("MATCH (c:Company) DETACH DELETE c")

# 2. 批量导入公司（使用 UNWIND 提高效率）
print("Importing companies...")
batch_size = 500
for i in range(0, len(stocks), batch_size):
    batch = stocks[i:i+batch_size]
    query = """
    UNWIND $batch AS row
    MERGE (c:Company {code: row.code})
    SET c.name = row.name,
        c.market_cap = row.market_cap,
        c.circulating_cap = row.circulating_cap,
        c.pe_ttm = row.pe_ttm,
        c.pb = row.pb,
        c.revenue = row.revenue,
        c.profit = row.profit,
        c.updated_at = datetime()
    """
    conn.execute_write(query, {"batch": batch})
    print(f"  Batch {i//batch_size + 1}: {i+1}-{min(i+batch_size, len(stocks))}")

# 3. 导入行业映射关系
if industry_map:
    print("\nImporting industry relations...")
    # 先创建行业节点
    industries = list(set(industry_map.values()))
    for ind in industries:
        conn.execute_write("""
            MERGE (s:Sector {name: $name})
            SET s.type = 'industry', s.updated_at = datetime()
        """, {"name": ind})
    
    # 再建立关系
    rel_batch = [{"code": code, "industry": ind} for code, ind in industry_map.items()]
    for i in range(0, len(rel_batch), 500):
        batch = rel_batch[i:i+500]
        conn.execute_write("""
            UNWIND $batch AS row
            MATCH (c:Company {code: row.code})
            MATCH (s:Sector {name: row.industry})
            MERGE (c)-[:BELONGS_TO]->(s)
        """, {"batch": batch})
    print(f"  Created {len(industry_map)} BELONGS_TO relations")

# 4. 为没有行业的公司创建一个"未分类"板块
uncategorized = [s["code"] for s in stocks if s["code"] not in industry_map]
if uncategorized:
    print(f"\n{len(uncategorized)} stocks without industry, tagging as '未分类'")
    conn.execute_write("""
        MERGE (s:Sector {name: '未分类'})
        SET s.type = 'industry'
    """)
    for i in range(0, len(uncategorized), 500):
        batch = uncategorized[i:i+500]
        conn.execute_write("""
            UNWIND $batch AS code
            MATCH (c:Company {code: code})
            MATCH (s:Sector {name: '未分类'})
            MERGE (c)-[:BELONGS_TO]->(s)
        """, {"batch": batch})

# 5. 验证
print("\nVerification:")
node_res = conn.execute_query("MATCH (n) RETURN labels(n)[0] as label, count(n) as cnt")
for r in node_res:
    print(f"  {r['label']}: {r['cnt']}")

rel_res = conn.execute_query("MATCH ()-[r]->() RETURN type(r) as t, count(r) as c")
for r in rel_res:
    print(f"  Relation {r['t']}: {r['c']}")

print("\nDone!")
