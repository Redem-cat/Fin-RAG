# Neo4j 知识图谱集成文档

## 概述

本文档描述如何将 Neo4j 知识图谱层集成到 FinRAG 系统中，实现结构化推理和多跳检索。

## 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                       User Query                        │
└─────────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Vector   │   │  Neo4j   │   │  LLM     │
    │ Search   │   │   KG     │   │ Context  │
    │ (ES)     │   │(Cypher)  │   │ Augment  │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
         └──────────────┼──────────────┘
                        ▼
              ┌─────────────────┐
              │  Fusion Layer   │
              │   (RRF/BM25)   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │      LLM       │
              │  (Final Gen)   │
              └─────────────────┘
```

## 金融本体论

### 节点类型

| 类型 | 说明 | 属性 |
|------|------|------|
| Company | 上市公司 | name, code, sector, market_cap |
| Person | 高管/分析师 | name, title, company |
| Sector | 行业 | name, description |
| Asset | 股票/债券 | name, code, type, price |
| Event | 财报/并购/政策 | name, type, date, impact |
| Indicator | 经济指标 | name, value, unit, date |

### 关系类型

| 关系 | 说明 | 方向 |
|------|------|------|
| BELONGS_TO | 属于行业 | Company → Sector |
| CEO_OF | CEO 关系 | Company → Person |
| COMPETES_WITH | 竞争 | Company ↔ Company |
| AFFECTED_BY | 受影响 | Company → Event |
| IMPACTS | 影响 | Event → Sector |
| ISSUED_BY | 发行 | Asset → Company |
| REPORTED | 报告 | Company → Event |

## 快速开始

### 1. 安装 Neo4j

#### 方式一：Docker（推荐）

```bash
docker run \
    --name neo4j \
    -p 7474:7474 \
    -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    -e NEO4J_PLUGINS='["apoc"]' \
    -v neo4j-data:/data \
    -d neo4j:5.15
```

#### 方式二：本地安装

1. 下载 [Neo4j Desktop](https://neo4j.com/download/)
2. 创建数据库
3. 记下连接 URI (bolt://localhost:7687)

### 2. 配置环境变量

```bash
# Neo4j 连接
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# KG 功能开关
KG_ENABLED=true
KG_MAX_HOPS=3
KG_MIN_CONFIDENCE=0.7
```

### 3. 安装依赖

```bash
pip install neo4j
```

### 4. 启动应用

```bash
streamlit run src/streamlit_app.py
```

## 使用方式

### 1. KG 页面

在 Streamlit 导航中选择 **KG** 页面：

- **图谱概览**: 查看当前图谱统计
- **关系查询**: 按公司/行业查询关系网络
- **影响分析**: 分析事件影响链
- **实体搜索**: 搜索特定实体

### 2. Chat 页面集成

在 Chat 页面中，系统会自动：

1. 检测是否需要触发 KG 检索
2. 执行 Cypher 查询
3. 融合 KG 结果与向量检索结果
4. 增强 LLM 上下文

触发关键词：
- "关系"、"竞争"、"合作"
- "影响"、"冲击"
- "产业链"、"上下游"

### 3. 导入数据

#### 从文档导入

```python
from src.knowledge_graph import import_from_documents

documents = [
    "苹果公司发布 Q4 财报，营收增长 8%...",
    "特斯拉在上海建厂，月产能达 2 万辆..."
]

result = import_from_documents(documents)
print(f"导入完成: {result['total_entities']} 实体, {result['total_relations']} 关系")
```

#### 手动导入

```cypher
// 在 Neo4j Browser 中执行
CREATE (apple:Company {name: "Apple", code: "AAPL", sector: "Technology"})
CREATE (ms:Company {name: "Microsoft", code: "MSFT", sector: "Technology"})
CREATE (apple)-[:COMPETES_WITH]->(ms)
CREATE (apple)-[:CEO_OF]->(:Person {name: "Tim Cook"})
```

## Cypher 查询示例

### 查找公司关系网络

```cypher
MATCH (c:Company {name: "Apple"})-[r]-(other)
RETURN c.name AS company,
       type(r) AS relation_type,
       labels(other)[0] AS related_type,
       other.name AS related_entity
```

### 查找影响链

```cypher
MATCH path = (event:Event)-[:IMPACTS*1..3]->(sector:Sector)
WHERE event.name CONTAINS "加息"
RETURN path,
       [n IN nodes(path) | n.name] AS chain
```

### 查找行业公司

```cypher
MATCH (s:Sector {name: "Technology"})<-[:BELONGS_TO]-(c:Company)
RETURN c.name, c.market_cap
ORDER BY c.market_cap DESC
```

## API 参考

### `KnowledgeGraphRetriever`

```python
from src.knowledge_graph import KnowledgeGraphRetriever

retriever = KnowledgeGraphRetriever()

# 查询
result = retriever.query("苹果公司的竞争对手有哪些？")
print(result.explanation)
print(result.entities)
```

### `hybrid_retrieval`

```python
from src.knowledge_graph import hybrid_retrieval

# 混合检索
result = hybrid_retrieval(
    question="腾讯的影响有哪些？",
    vector_results=es_results,
    top_k=5
)
print(result["context"])
```

## 性能优化

### 1. 索引

```cypher
CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name);
CREATE INDEX company_code IF NOT EXISTS FOR (c:Company) ON (c.code);
CREATE INDEX event_date IF NOT EXISTS FOR (e:Event) ON (e.date);
```

### 2. 缓存

对于频繁查询的实体，可以实现 Redis 缓存层。

### 3. 批量导入

使用 APOC 插件进行批量导入：

```cypher
CALL apoc.periodic.iterate(
  "UNWIND $data AS row RETURN row",
  "MERGE (c:Company {name: row.name}) SET c += row.props",
  {batchSize: 1000, parallel: true, params: {data: [...]}}
)
```

## 常见问题

### Q: Neo4j 连接失败？

检查：
1. Docker 容器是否运行 `docker ps`
2. 端口是否正确映射
3. 认证信息是否正确

### Q: 如何查看图谱？

打开 http://localhost:7474 使用 Neo4j Browser。

### Q: 支持哪些图数据库？

当前支持 Neo4j。如需其他数据库（如 TuGraph、 NebulaGraph），可扩展 `GraphConnection` 类。

## 扩展功能

### 图机器学习

使用 Neo4j GDS 库进行：

```cypher
CALL gds.pageRank.stream('myGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC
LIMIT 10
```

### 可视化

集成 pyvis 进行交互式图谱可视化。

## 参考资料

- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)
- [APOC Plugin](https://neo4j.com/docs/apoc/current/)
- [LangChain Neo4j Integration](https://python.langchain.com/docs/integrations/graph/)
- [GraphRAG Paper](https://arxiv.org/abs/2304.03442)
