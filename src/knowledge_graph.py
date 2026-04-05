"""
Neo4j 知识图谱模块 - 为金融 RAG 系统提供结构化知识检索

功能：
1. 金融本体论定义（公司、人员、行业、资产、事件、指标）
2. LLM 驱动的实体/关系抽取
3. Cypher 查询构建
4. 混合检索（KG + Vector）
5. 多跳推理

参考架构：
https://python.langchain.com/docs/integrations/graph/
"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
base_path = Path(__file__).parent.parent.resolve()
dotenv_path = base_path / "elastic-start-local/.env"
load_dotenv(dotenv_path=dotenv_path)


# =========================
# 🔹 配置
# =========================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# KG 检索配置
KG_ENABLED = os.getenv("KG_ENABLED", "true").lower() == "true"
KG_MAX_HOPS = int(os.getenv("KG_MAX_HOPS", "3"))
KG_MIN_CONFIDENCE = float(os.getenv("KG_MIN_CONFIDENCE", "0.7"))


# =========================
# 🔹 金融本体论定义
# =========================
class EntityType(Enum):
    """实体类型枚举"""
    COMPANY = "Company"           # 公司
    PERSON = "Person"              # 人物
    SECTOR = "Sector"             # 行业
    ASSET = "Asset"               # 资产（股票、债券）
    EVENT = "Event"               # 事件（财报、并购、政策）
    INDICATOR = "Indicator"        # 指标（CPI、利率）


class RelationType(Enum):
    """关系类型枚举"""
    # 公司关系
    CEO_OF = "CEO_OF"
    BELONGS_TO = "BELONGS_TO"           # 公司属于行业
    COMPETES_WITH = "COMPETES_WITH"     # 竞争关系
    PARTNER_OF = "PARTNER_OF"           # 合作关系
    SUBSIDIARY_OF = "SUBSIDIARY_OF"     # 子公司关系
    INVESTED_BY = "INVESTED_BY"         # 被投资

    # 资产关系
    ISSUED_BY = "ISSUED_BY"             # 资产发行方
    TRACKED_BY = "TRACKED_BY"           # 被指数追踪

    # 事件影响
    AFFECTED_BY = "AFFECTED_BY"         # 受影响
    IMPACTS = "IMPACTS"                 # 影响
    REPORTED = "REPORTED"               # 报告了
    DRIVEN_BY = "DRIVEN_BY"            # 由...驱动

    # 指标关系
    MEASURED_BY = "MEASURED_BY"         # 由...衡量
    CORRELATES_WITH = "CORRELATES_WITH" # 与...相关


# 实体类型中文映射
ENTITY_TYPE_NAMES = {
    EntityType.COMPANY: "公司",
    EntityType.PERSON: "人物",
    EntityType.SECTOR: "行业",
    EntityType.ASSET: "资产",
    EntityType.EVENT: "事件",
    EntityType.INDICATOR: "指标",
}


# =========================
# 🔹 数据结构
# =========================
@dataclass
class ExtractedEntity:
    """抽取的实体"""
    name: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class ExtractedRelation:
    """抽取的关系"""
    source: str
    relation_type: str
    target: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class KGQueryResult:
    """知识图谱查询结果"""
    entities: List[Dict]
    relations: List[Dict]
    paths: List[List[Dict]]
    explanation: str
    confidence: float = 1.0


# =========================
# 🔹 Neo4j 连接管理
# =========================
class Neo4jConnection:
    """Neo4j 连接管理器"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.driver = None
        self._initialized = True
        self._connect()

    def _connect(self):
        """建立连接"""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
            )
            print(f"[Neo4j] 连接成功: {NEO4J_URI}")
        except ImportError:
            print("[Neo4j] 警告: neo4j 驱动未安装，请运行: pip install neo4j")
            self.driver = None
        except Exception as e:
            print(f"[Neo4j] 连接失败: {e}")
            self.driver = None

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            self.driver = None

    def is_connected(self) -> bool:
        """检查连接状态"""
        if not self.driver:
            return False
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                session.run("RETURN 1")
            return True
        except Exception:
            return False

    def execute_query(self, query: str, params: Dict = None) -> List[Dict]:
        """执行 Cypher 查询"""
        if not self.driver:
            return []

        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(query, params or {})
                return [dict(record) for record in result]
        except Exception as e:
            print(f"[Neo4j] 查询执行失败: {e}")
            return []

    def execute_write(self, query: str, params: Dict = None) -> bool:
        """执行写操作"""
        if not self.driver:
            return False

        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                session.run(query, params or {})
            return True
        except Exception as e:
            print(f"[Neo4j] 写操作失败: {e}")
            return False


# 单例访问函数
def get_neo4j_connection() -> Neo4jConnection:
    """获取 Neo4j 连接实例"""
    return Neo4jConnection()


# =========================
# 🔹 Schema 管理
# =========================
FINANCIAL_SCHEMA = """
节点类型:
- Company: 公司 (name, code, sector, market_cap, description)
- Person: 人物 (name, title, company, biography)
- Sector: 行业 (name, description, trend)
- Asset: 资产 (name, code, type, price, currency)
- Event: 事件 (name, type, date, impact, description)
- Indicator: 经济指标 (name, value, unit, date, region)

关系类型:
- (Company)-[:CEO_OF]->(Person)
- (Company)-[:BELONGS_TO]->(Sector)
- (Company)-[:COMPETES_WITH]->(Company)
- (Company)-[:PARTNER_OF]->(Company)
- (Company)-[:SUBSIDIARY_OF]->(Company)
- (Company)-[:AFFECTED_BY]->(Event)
- (Company)-[:REPORTED]->(Event)
- (Asset)-[:ISSUED_BY]->(Company)
- (Event)-[:IMPACTS]->(Sector)
- (Indicator)-[:AFFECTS]->(Sector)
"""


def create_schema_constraints() -> bool:
    """创建 Schema 约束和索引"""
    conn = get_neo4j_connection()

    constraints = [
        "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT sector_name IF NOT EXISTS FOR (s:Sector) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT asset_code IF NOT EXISTS FOR (a:Asset) REQUIRE a.code IS UNIQUE",
    ]

    indexes = [
        "CREATE INDEX company_code IF NOT EXISTS FOR (c:Company) ON (c.code)",
        "CREATE INDEX event_date IF NOT EXISTS FOR (e:Event) ON (e.date)",
        "CREATE INDEX indicator_date IF NOT EXISTS FOR (i:Indicator) ON (i.date)",
    ]

    success = True
    for c in constraints + indexes:
        try:
            conn.execute_write(c)
        except Exception as e:
            print(f"[Neo4j] Schema 创建跳过: {e}")

    return success


# =========================
# 🔹 实体/关系抽取
# =========================
EXTRACTION_PROMPT = """你是一个专业的金融知识图谱抽取系统。请从文本中抽取实体和关系。

## 金融本体论
节点类型: Company(公司), Person(人物), Sector(行业), Asset(资产), Event(事件), Indicator(指标)

关系类型:
- CEO_OF: 公司CEO关系
- BELONGS_TO: 归属关系
- COMPETES_WITH: 竞争关系
- AFFECTED_BY: 受影响
- IMPACTS: 影响
- REPORTED: 报告了
- DRIVEN_BY: 由...驱动
- ISSUED_BY: 发行

## 输入文本
{text}

## 要求
1. 抽取所有金融相关的实体
2. 识别实体之间的关系
3. 输出标准 JSON 格式

## 输出格式
```json
{{
    "entities": [
        {{"name": "实体名", "type": "类型", "properties": {{"key": "value"}}}}
    ],
    "relations": [
        {{"source": "实体A", "type": "关系类型", "target": "实体B", "properties": {{}}}}
    ]
}}
```

请只输出 JSON，不要有其他内容："""


class EntityExtractor:
    """LLM 驱动的实体/关系抽取器"""

    def __init__(self):
        self.llm = None

    def _get_llm(self):
        """延迟加载 LLM"""
        if self.llm is None:
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "my-qwen25"),
                temperature=0.1
            )
        return self.llm

    def extract(self, text: str) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """
        从文本中抽取实体和关系

        Args:
            text: 输入文本

        Returns:
            (实体列表, 关系列表)
        """
        llm = self._get_llm()
        prompt = EXTRACTION_PROMPT.format(text=text)

        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # 解析 JSON
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())

                entities = [
                    ExtractedEntity(
                        name=e["name"],
                        entity_type=e["type"],
                        properties=e.get("properties", {}),
                        confidence=0.9
                    )
                    for e in data.get("entities", [])
                ]

                relations = [
                    ExtractedRelation(
                        source=r["source"],
                        relation_type=r["type"],
                        target=r["target"],
                        properties=r.get("properties", {}),
                        confidence=0.9
                    )
                    for r in data.get("relations", [])
                ]

                return entities, relations

        except Exception as e:
            print(f"[EntityExtractor] 抽取失败: {e}")

        return [], []


# =========================
# 🔹 Cypher 查询构建
# =========================
class CypherQueryBuilder:
    """Cypher 查询构建器"""

    @staticmethod
    def find_related_entities(entity_name: str, relation_type: str = None,
                              max_depth: int = 2) -> str:
        """
        查找相关实体

        Args:
            entity_name: 实体名称
            relation_type: 关系类型（可选）
            max_depth: 最大跳数

        Returns:
            Cypher 查询语句
        """
        if relation_type:
            return f"""
            MATCH path = (start {{name: $entity}})-[r:{relation_type}*1..{max_depth}]->(end)
            RETURN path,
                   [n IN nodes(path) | {{name: n.name, type: labels(n)[0], properties: properties(n)}}] AS nodes,
                   [r IN relationships(path) | {{type: type(r), properties: properties(r)}}] AS edges
            LIMIT 20
            """
        else:
            return f"""
            MATCH path = (start {{name: $entity}})-[r*1..{max_depth}]->(end)
            RETURN path,
                   [n IN nodes(path) | {{name: n.name, type: labels(n)[0], properties: properties(n)}}] AS nodes,
                   [r IN relationships(path) | {{type: type(r), properties: properties(r)}}] AS edges
            LIMIT 20
            """

    @staticmethod
    def find_entities_by_type(entity_type: str, limit: int = 50) -> str:
        """按类型查找实体"""
        return f"""
        MATCH (e:{entity_type})
        RETURN e.name AS name, properties(e) AS properties
        LIMIT {limit}
        """

    @staticmethod
    def find_affected_entities(event_name: str) -> str:
        """查找受事件影响的实体"""
        return """
        MATCH (e:Event {name: $event_name})-[r]-(affected)
        RETURN e.name AS event,
               type(r) AS relation,
               labels(affected)[0] AS entity_type,
               affected.name AS entity_name,
               properties(r) AS relation_properties
        """

    @staticmethod
    def find_company_relationships(company_name: str) -> str:
        """查找公司关系网络"""
        return """
        MATCH (c:Company {name: $company})-[r]-(other)
        RETURN c.name AS company,
               type(r) AS relation_type,
               labels(other)[0] AS related_type,
               other.name AS related_entity,
               properties(r) AS properties
        """

    @staticmethod
    def find_impact_chain(start_entity: str, end_entity: str, max_hops: int = 3) -> str:
        """查找影响链"""
        return f"""
        MATCH path = (start {{name: $start}})-[r*1..{max_hops}]-(end {{name: $end}})
        WHERE all(rel IN relationships(path) WHERE type(rel) IN ['AFFECTED_BY', 'IMPACTS', 'DRIVEN_BY', 'BELONGS_TO'])
        RETURN path,
               [n IN nodes(path) | n.name] AS chain,
               [r IN relationships(path) | type(r)] AS relation_types
        LIMIT 10
        """

    @staticmethod
    def find_sector_companies(sector_name: str) -> str:
        """查找行业内的公司"""
        return """
        MATCH (s:Sector {name: $sector})<-[:BELONGS_TO]-(c:Company)
        RETURN c.name AS company, c.code AS code, c.market_cap AS market_cap
        ORDER BY c.market_cap DESC
        """

    @staticmethod
    def find_similar_entities(entity_name: str, limit: int = 10) -> str:
        """查找相似实体（基于名称相似度）"""
        return f"""
        MATCH (e)
        WHERE toLower(e.name) CONTAINS toLower($name)
           OR toLower($name) CONTAINS toLower(e.name)
        RETURN e.name AS name,
               labels(e)[0] AS type,
               properties(e) AS properties
        LIMIT {limit}
        """


# =========================
# 🔹 知识图谱检索器
# =========================
class KnowledgeGraphRetriever:
    """知识图谱检索器"""

    def __init__(self):
        self.connection = get_neo4j_connection()
        self.query_builder = CypherQueryBuilder()
        self.extractor = EntityExtractor()

    def is_available(self) -> bool:
        """检查 KG 是否可用"""
        return KG_ENABLED and self.connection.is_connected()

    def query(self, question: str, query_type: str = "auto") -> KGQueryResult:
        """
        执行知识图谱查询

        Args:
            question: 用户问题
            query_type: 查询类型 (auto/related/affected/chain/companies)

        Returns:
            KGQueryResult 查询结果
        """
        if not self.is_available():
            return KGQueryResult(
                entities=[],
                relations=[],
                paths=[],
                explanation="知识图谱不可用",
                confidence=0.0
            )

        # 自动识别查询类型
        if query_type == "auto":
            query_type = self._identify_query_type(question)

        entities = []
        relations = []
        paths = []

        try:
            if "关系" in question or "竞争" in question or "合作" in question:
                # 提取公司名
                company = self._extract_company_name(question)
                if company:
                    result = self.connection.execute_query(
                        self.query_builder.find_company_relationships(company),
                        {"company": company}
                    )
                    entities, relations = self._parse_relationship_result(result)

            elif "影响" in question or "冲击" in question:
                event = self._extract_event_name(question)
                if event:
                    result = self.connection.execute_query(
                        self.query_builder.find_affected_entities(event),
                        {"event_name": event}
                    )
                    entities, relations = self._parse_affected_result(result)

            elif "哪些公司" in question or "行业" in question:
                sector = self._extract_sector_name(question)
                if sector:
                    result = self.connection.execute_query(
                        self.query_builder.find_sector_companies(sector),
                        {"sector": sector}
                    )
                    entities = self._parse_companies_result(result)

            elif "为什么" in question or "原因" in question:
                entity = self._extract_key_entity(question)
                if entity:
                    result = self.connection.execute_query(
                        self.query_builder.find_related_entities(entity),
                        {"entity": entity}
                    )
                    entities, relations, paths = self._parse_path_result(result)

            else:
                # 默认：查找相关实体
                entity = self._extract_key_entity(question)
                if entity:
                    result = self.connection.execute_query(
                        self.query_builder.find_similar_entities(entity),
                        {"name": entity}
                    )
                    entities = self._parse_similar_result(result)

            explanation = self._generate_explanation(question, entities, relations, query_type)

        except Exception as e:
            print(f"[KGRetriever] 查询失败: {e}")
            explanation = f"查询失败: {e}"

        return KGQueryResult(
            entities=entities,
            relations=relations,
            paths=paths,
            explanation=explanation,
            confidence=0.8 if entities else 0.0
        )

    def _identify_query_type(self, question: str) -> str:
        """识别查询类型"""
        if "关系" in question or "竞争" in question:
            return "relationships"
        elif "影响" in question or "冲击" in question:
            return "affected"
        elif "哪些" in question or "行业" in question:
            return "companies"
        elif "为什么" in question or "原因" in question:
            return "chain"
        return "related"

    def _extract_company_name(self, text: str) -> Optional[str]:
        """提取公司名称"""
        # 简单模式匹配
        companies = ["苹果", "Apple", "微软", "Microsoft", "谷歌", "Google",
                     "腾讯", "阿里", "阿里巴巴", "百度", "茅台", "宁德时代", "比亚迪"]
        for c in companies:
            if c in text:
                return c
        return None

    def _extract_event_name(self, text: str) -> Optional[str]:
        """提取事件名称"""
        events = ["加息", "降息", "财报", "并购", "制裁", "政策", "疫情"]
        for e in events:
            if e in text:
                return e
        return None

    def _extract_sector_name(self, text: str) -> Optional[str]:
        """提取行业名称"""
        sectors = ["科技", "金融", "医疗", "能源", "消费", "工业", "房地产"]
        for s in sectors:
            if s in text:
                return s
        return None

    def _extract_key_entity(self, text: str) -> Optional[str]:
        """提取关键实体"""
        # 使用第一个识别到的实体
        for pattern in [r'([\u4e00-\u9fa5]{2,6}(公司|银行|集团|基金))',
                        r'([A-Z][a-z]+(Inc|Corp|Ltd)?)']:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    def _parse_relationship_result(self, result: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """解析关系查询结果"""
        entities = []
        relations = []

        for record in result:
            rel_type = record.get("relation_type", "")
            entity_type = record.get("related_type", "")
            entity_name = record.get("related_entity", "")

            if entity_name:
                entities.append({
                    "name": entity_name,
                    "type": entity_type,
                    "relation": rel_type
                })

            relations.append({
                "from": record.get("company", ""),
                "type": rel_type,
                "to": entity_name,
                "properties": record.get("properties", {})
            })

        return entities, relations

    def _parse_affected_result(self, result: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """解析受影响实体结果"""
        entities = []
        relations = []

        for record in result:
            entities.append({
                "name": record.get("entity_name", ""),
                "type": record.get("entity_type", ""),
                "event": record.get("event", "")
            })

            relations.append({
                "from": record.get("event", ""),
                "type": record.get("relation", ""),
                "to": record.get("entity_name", "")
            })

        return entities, relations

    def _parse_companies_result(self, result: List[Dict]) -> List[Dict]:
        """解析公司查询结果"""
        return [
            {
                "name": r.get("company", ""),
                "code": r.get("code", ""),
                "market_cap": r.get("market_cap", 0)
            }
            for r in result
        ]

    def _parse_path_result(self, result: List[Dict]) -> Tuple[List[Dict], List[Dict], List[List[Dict]]]:
        """解析路径查询结果"""
        entities = []
        relations = []
        paths = []

        for record in result:
            path_entities = record.get("nodes", [])
            path_relations = record.get("edges", [])

            paths.append(path_entities)

            for e in path_entities:
                entities.append({"name": e.get("name", ""), "type": e.get("type", "")})

            for r in path_relations:
                relations.append({"type": r.get("type", ""), "properties": r.get("properties", {})})

        return entities, relations, paths

    def _parse_similar_result(self, result: List[Dict]) -> List[Dict]:
        """解析相似实体结果"""
        return [
            {
                "name": r.get("name", ""),
                "type": r.get("type", ""),
                "properties": r.get("properties", {})
            }
            for r in result
        ]

    def _generate_explanation(self, question: str, entities: List[Dict],
                              relations: List[Dict], query_type: str) -> str:
        """生成结果解释"""
        if not entities:
            return "未找到相关实体"

        entity_count = len(entities)
        relation_count = len(relations)

        explanations = {
            "relationships": f"找到 {entity_count} 个相关实体和 {relation_count} 条关系",
            "affected": f"找到 {entity_count} 个受影响的实体",
            "companies": f"找到 {entity_count} 家公司",
            "chain": f"找到 {len(relations)} 条关联路径",
            "related": f"找到 {entity_count} 个相关实体"
        }

        return explanations.get(query_type, f"找到 {entity_count} 个结果")


# =========================
# 🔹 混合检索
# =========================
def hybrid_retrieval(question: str, vector_results: List[Dict] = None,
                      top_k: int = 5) -> Dict[str, Any]:
    """
    混合检索：结合知识图谱和向量检索

    Args:
        question: 用户问题
        vector_results: 向量检索结果
        top_k: 返回结果数量

    Returns:
        混合检索结果
    """
    retriever = KnowledgeGraphRetriever()

    # 1. KG 检索
    kg_result = retriever.query(question)

    # 2. 结果融合
    combined_context = []
    sources = []

    # 添加 KG 结果
    if kg_result.entities:
        kg_context = "【知识图谱检索结果】\n"
        for entity in kg_result.entities[:top_k]:
            kg_context += f"- {entity['name']} ({entity.get('type', '未知')})"
            if 'relation' in entity:
                kg_context += f" - {entity['relation']}"
            kg_context += "\n"

        if kg_result.explanation:
            kg_context += f"\n分析: {kg_result.explanation}\n"

        combined_context.append(kg_context)
        sources.append({"type": "knowledge_graph", "confidence": kg_result.confidence})

    # 添加向量检索结果
    if vector_results:
        vector_context = "【向量检索结果】\n"
        for i, doc in enumerate(vector_results[:top_k], 1):
            content = doc.get('content', doc.page_content if hasattr(doc, 'page_content') else '')
            source = doc.get('source', doc.metadata.get('source', 'unknown') if hasattr(doc, 'metadata') else 'unknown')
            vector_context += f"\n{i}. {content[:200]}...\n(来源: {source})\n"
        combined_context.append(vector_context)
        sources.append({"type": "vector_search", "confidence": 0.7})

    return {
        "question": question,
        "context": "\n\n".join(combined_context),
        "kg_result": kg_result,
        "sources": sources,
        "has_kg_results": len(kg_result.entities) > 0
    }


# =========================
# 🔹 导入/导出
# =========================
def import_from_file(file_path: str, chunk_size: int = 1500) -> Dict[str, Any]:
    """
    从单个文件导入知识图谱

    Args:
        file_path: 文件路径
        chunk_size: 分块大小（字符数）

    Returns:
        导入统计
    """
    path = Path(file_path)
    if not path.exists():
        return {"total_entities": 0, "total_relations": 0, "success": False, "error": "文件不存在"}

    content = path.read_text(encoding="utf-8", errors="ignore")
    if not content.strip():
        return {"total_entities": 0, "total_relations": 0, "success": False, "error": "文件为空"}

    # 将文档分块，每块约 chunk_size 字符
    chunks = []
    paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 50]
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para
    if current_chunk:
        chunks.append(current_chunk)

    return import_from_documents(chunks)


def import_from_directory(dir_path: str, chunk_size: int = 1500) -> Dict[str, Any]:
    """
    从目录批量导入知识图谱

    Args:
        dir_path: 目录路径
        chunk_size: 分块大小（字符数）

    Returns:
        导入统计
    """
    directory = Path(dir_path)
    if not directory.exists():
        return {"total_files": 0, "total_entities": 0, "total_relations": 0, "success": False, "error": "目录不存在"}

    supported_extensions = {".md", ".txt", ".csv"}
    files = [f for f in directory.iterdir() if f.suffix.lower() in supported_extensions and f.is_file()]

    if not files:
        return {"total_files": 0, "total_entities": 0, "total_relations": 0, "success": False, "error": "目录中没有支持的文件 (.md, .txt, .csv)"}

    total_entities = 0
    total_relations = 0
    processed_files = 0
    errors = []

    for file_path in files:
        result = import_from_file(str(file_path), chunk_size)
        total_entities += result.get("total_entities", 0)
        total_relations += result.get("total_relations", 0)
        if result.get("success"):
            processed_files += 1
        else:
            errors.append(f"{file_path.name}: {result.get('error', 'unknown')}")

    return {
        "total_files": len(files),
        "processed_files": processed_files,
        "total_entities": total_entities,
        "total_relations": total_relations,
        "success": processed_files > 0,
        "errors": errors
    }


def import_from_documents(documents: List[str], batch_size: int = 10) -> Dict[str, Any]:
    """
    从文档导入到知识图谱

    Args:
        documents: 文档列表
        batch_size: 批处理大小

    Returns:
        导入统计
    """
    extractor = EntityExtractor()
    conn = get_neo4j_connection()

    total_entities = 0
    total_relations = 0

    for doc in documents:
        entities, relations = extractor.extract(doc)

        # 导入实体
        for entity in entities:
            query = f"""
            MERGE (e:{entity.entity_type} {{name: $name}})
            SET e += $properties
            """
            conn.execute_write(query, {
                "name": entity.name,
                "properties": entity.properties
            })
            total_entities += 1

        # 导入关系
        for relation in relations:
            query = f"""
            MATCH (source {{name: $source}})
            MATCH (target {{name: $target}})
            MERGE (source)-[r:{relation.relation_type}]->(target)
            SET r += $properties
            """
            conn.execute_write(query, {
                "source": relation.source,
                "target": relation.target,
                "properties": relation.properties
            })
            total_relations += 1

    return {
        "total_entities": total_entities,
        "total_relations": total_relations,
        "success": True
    }


def export_to_cypher(output_path: str) -> bool:
    """导出图谱为 Cypher 格式"""
    conn = get_neo4j_connection()

    query = """
    MATCH (n)
    OPTIONAL MATCH (n)-[r]->(m)
    RETURN
        labels(n)[0] AS start_label,
        n.name AS start_name,
        type(r) AS relation,
        labels(m)[0] AS end_label,
        m.name AS end_name
    """

    results = conn.execute_query(query)

    cypher_lines = []
    for r in results:
        if r.get("relation"):
            line = f"MATCH (a:{r['start_label']}), (b:{r['end_label']}) "
            line += f"WHERE a.name='{r['start_name']}' AND b.name='{r['end_name']}' "
            line += f"CREATE (a)-[:{r['relation']}]->(b)"
        else:
            line = f"CREATE (:{r['start_label']} {{name: '{r['start_name']}'}})"

        cypher_lines.append(line)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(cypher_lines))
        return True
    except Exception as e:
        print(f"[KG] 导出失败: {e}")
        return False


# =========================
# 🔹 触发关键词
# =========================
KG_TRIGGER_KEYWORDS = [
    "关系", "影响", "关联", "竞争", "合作", "产业链",
    "上下游", "关联方", "股权", "股东", "高管",
    "为什么", "原因", "分析", "比较", "对比",
    "行业", "公司", "板块", "概念股"
]


def should_trigger_kg(question: str) -> bool:
    """检查是否应该触发知识图谱"""
    question_lower = question.lower()
    return any(kw in question_lower for kw in KG_TRIGGER_KEYWORDS)


# =========================
# 🔹 便捷函数
# =========================
def get_kg_retriever() -> KnowledgeGraphRetriever:
    """获取 KG 检索器实例"""
    return KnowledgeGraphRetriever()


def check_kg_status() -> Tuple[bool, str]:
    """检查知识图谱状态"""
    if not KG_ENABLED:
        return False, "KG 功能未启用 (KG_ENABLED=false)"

    try:
        conn = get_neo4j_connection()
        if not conn.is_connected():
            return False, f"Neo4j 未连接 ({NEO4J_URI})"

        # 查询节点数量
        result = conn.execute_query("MATCH (n) RETURN count(n) AS count")
        node_count = result[0].get("count", 0) if result else 0

        result = conn.execute_query("MATCH ()-[r]->() RETURN count(r) AS count")
        rel_count = result[0].get("count", 0) if result else 0

        return True, f"Neo4j 已连接: {node_count} 节点, {rel_count} 关系"

    except Exception as e:
        return False, f"检查失败: {e}"


if __name__ == "__main__":
    # 测试代码
    print("=" * 50)
    print("知识图谱模块测试")
    print("=" * 50)

    status, msg = check_kg_status()
    print(f"状态: {status}, {msg}")

    if status:
        retriever = get_kg_retriever()

        # 测试查询
        test_questions = [
            "苹果公司的竞争对手有哪些？",
            "降息对哪些行业有影响？",
            "科技行业有哪些公司？"
        ]

        for q in test_questions:
            print(f"\n问题: {q}")
            result = retriever.query(q)
            print(f"结果: {result.explanation}")
            print(f"实体数: {len(result.entities)}")
