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
    """实体类型枚举 - 金融领域芯片供应链"""
    COMPANY = "Company"           # 公司（设计公司、整机厂、设备商）
    PERSON = "Person"             # 人物
    SECTOR = "Sector"             # 行业/概念板块
    PRODUCT = "Product"           # 产品/部件（GPU/CPU/存储芯片/光刻机）
    FOUNDRY = "Foundry"           # 代工厂/产线
    MATERIAL = "Material"         # 原材料/设备（硅片/光刻胶/刻蚀机）
    LOCATION = "Location"         # 地点（国家/城市/园区）
    EVENT = "Event"               # 事件（停产/事故/政策）
    ASSET = "Asset"               # 资产（股票）
    INDICATOR = "Indicator"       # 指标


class RelationType(Enum):
    """关系类型枚举 - 芯片供应链"""
    # 公司与产品
    DESIGNS = "DESIGNS"               # (公司)-[设计]->(产品)
    PURCHASES = "PURCHASES"           # (公司)-[采购]->(产品)
    MANUFACTURES_FOR = "MANUFACTURES_FOR"  # (代工厂)-[代工]->(公司/产品)

    # 公司与代工厂
    OUTSOURCES_TO = "OUTSOURCES_TO"   # (公司)-[委托代工]->(代工厂)

    # 依赖关系
    DEPENDS_ON = "DEPENDS_ON"         # (代工厂)-[依赖]->(原材料/设备)
    SUPPLIES = "SUPPLIES"             # (公司)-[供应]->(公司)

    # 地点
    LOCATED_IN = "LOCATED_IN"         # (代工厂/公司)-[位于]->(地点)

    # 竞争
    COMPETES_WITH = "COMPETES_WITH"   # (公司)-[竞争]->(公司)

    # 行业归属
    BELONGS_TO = "BELONGS_TO"         # (公司)-[属于]->(板块)

    # 事件影响
    AFFECTED_BY = "AFFECTED_BY"
    IMPACTS = "IMPACTS"

    # 资产关系（保留）
    ISSUED_BY = "ISSUED_BY"

    # 传统关系（兼容）
    CEO_OF = "CEO_OF"
    PARTNER_OF = "PARTNER_OF"
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    INVESTED_BY = "INVESTED_BY"
    REPORTED = "REPORTED"
    DRIVEN_BY = "DRIVEN_BY"
    MEASURED_BY = "MEASURED_BY"
    CORRELATES_WITH = "CORRELATES_WITH"


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
- Company: 公司 (name, code, sector, market_cap, country, description)
- Product: 产品/部件 (name, type, 工艺节点, description)
- Foundry: 代工厂/产线 (name, location, 工艺节点, 月产能)
- Material: 原材料/设备 (name, type, supplier, description)
- Location: 地点 (name, type, country)
- Sector: 概念板块 (name, code, description)
- Event: 事件 (name, type, date, impact, description)
- Person: 人物 (name, title, company)

关系类型（芯片供应链）:
- (Company)-[:DESIGNS]->(Product)            公司设计某产品
- (Company)-[:PURCHASES]->(Product)          公司采购某产品
- (Company)-[:OUTSOURCES_TO {工艺节点, 占比}]->(Foundry)   委托代工
- (Foundry)-[:MANUFACTURES_FOR {工艺节点}]->(Company/Product) 代工生产
- (Foundry)-[:DEPENDS_ON {依赖程度, 是否有替代}]->(Material) 依赖材料/设备
- (Foundry/Company)-[:LOCATED_IN]->(Location) 位于某地
- (Company)-[:COMPETES_WITH {竞争领域}]->(Company) 竞争关系
- (Company)-[:SUPPLIES]->(Company)           供应关系
- (Company)-[:BELONGS_TO]->(Sector)          属于板块
- (Company/Foundry)-[:AFFECTED_BY]->(Event)  受事件影响
- (Asset)-[:ISSUED_BY]->(Company)            资产发行
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
EXTRACTION_PROMPT = """你是一个专业的半导体/芯片供应链知识图谱抽取系统。请从文本中抽取实体和关系，用于金融领域的上下游风险分析。

## 核心任务
从新闻/公告中识别：哪些公司依赖哪些代工厂/原材料，哪些公司之间存在竞争/供应关系。

## 本体论
节点类型:
- Company(公司): 芯片设计公司、整机厂（手机/汽车/服务器厂商）、设备商
- Product(产品): 具体芯片产品（如"A100 GPU"、"麒麟9000"）、部件类型（"GPU/CPU/存储芯片/光刻机"）
- Foundry(代工厂): 台积电、中芯国际、三星等，含工艺节点属性
- Material(原材料/设备): 硅片、光刻胶、光刻机、特种气体
- Location(地点): 国家、城市、园区（如"台湾台南"、"上海张江"）
- Sector(板块): 概念板块（如"AI芯片"、"汽车芯片"）
- Event(事件): 停产、事故、政策变化

关系类型（重点抽取）:
- DESIGNS: (公司)-[设计]->(产品) — 某公司设计某芯片
- PURCHASES: (公司)-[采购]->(产品) — 某公司采购某芯片/部件
- MANUFACTURES_FOR: (代工厂)-[代工]->(公司/产品) — 代工厂为某公司/产品生产
- OUTSOURCES_TO: (公司)-[委托代工]->(代工厂) — 某公司委托某代工厂
- DEPENDS_ON: (代工厂)-[依赖]->(原材料/设备) — 代工厂依赖某种材料/设备
- LOCATED_IN: (代工厂/公司)-[位于]->(地点) — 位于某地
- COMPETES_WITH: (公司)-[竞争]->(公司) — 同一细分领域竞争
- SUPPLIES: (公司)-[供应]->(公司) — 上游供应商
- BELONGS_TO: (公司)-[属于]->(板块) — 属于某概念板块
- AFFECTED_BY: (公司/代工厂)-[受影响]->(事件)

## 关系属性（如文本中有相关信息，务必抽取）
- 工艺节点: "5nm"、"7nm"、"28nm"
- 占比/依赖程度: "高"/"中"/"低"，或具体百分比
- 时间: 关系开始/结束时间
- 来源/置信度: 信息来源

## 输入文本
{text}

## 要求
1. **只抽取与芯片/半导体/电子供应链相关的实体和关系**，忽略无关内容
2. 优先抽取：代工关系、供应链依赖、竞争关系
3. 关系属性尽量填充（工艺节点、占比、依赖程度）
4. 输出标准 JSON 格式

## 输出格式
```json
{{
    "entities": [
        {{"name": "实体名", "type": "Company|Product|Foundry|Material|Location|Sector|Event", "properties": {{"key": "value"}}}}
    ],
    "relations": [
        {{"source": "实体A", "type": "关系类型", "target": "实体B", "properties": {{"工艺节点": "5nm", "占比": "高", "时间": "2024-01"}}}}
    ]
}}
```

请只输出 JSON，不要有其他内容："""


class EntityExtractor:
    """LLM 驱动的实体/关系抽取器"""

    def __init__(self):
        self.llm = None

    def _get_llm(self):
        """延迟加载 LLM（使用 DeepSeek API）"""
        if self.llm is None:
            from src.llm_client import get_llm
            self.llm = get_llm(temperature=0.1)
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
        查找相关实体（支持模糊匹配）
        """
        if relation_type:
            return f"""
            MATCH (start)
            WHERE toLower(start.name) CONTAINS toLower($entity)
               OR toLower($entity) CONTAINS toLower(start.name)
            MATCH path = (start)-[r:{relation_type}*1..{max_depth}]->(end)
            RETURN path,
                   [n IN nodes(path) | {{name: n.name, type: labels(n)[0], properties: properties(n)}}] AS nodes,
                   [r IN relationships(path) | {{type: type(r), properties: properties(r)}}] AS edges
            LIMIT 20
            """
        else:
            return f"""
            MATCH (start)
            WHERE toLower(start.name) CONTAINS toLower($entity)
               OR toLower($entity) CONTAINS toLower(start.name)
            MATCH path = (start)-[r*1..{max_depth}]->(end)
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
        """查找受事件影响的实体（支持模糊匹配）"""
        return """
        MATCH (e:Event)
        WHERE toLower(e.name) CONTAINS toLower($event_name)
           OR toLower($event_name) CONTAINS toLower(e.name)
        MATCH (e)-[r]-(affected)
        RETURN e.name AS event,
               type(r) AS relation,
               labels(affected)[0] AS entity_type,
               affected.name AS entity_name,
               properties(r) AS relation_properties
        """

    @staticmethod
    def find_company_relationships(company_name: str) -> str:
        """查找公司关系网络（支持模糊匹配）"""
        return """
        MATCH (c:Company)
        WHERE toLower(c.name) CONTAINS toLower($company)
           OR toLower($company) CONTAINS toLower(c.name)
        MATCH (c)-[r]-(other)
        RETURN c.name AS company,
               type(r) AS relation_type,
               labels(other)[0] AS related_type,
               other.name AS related_entity,
               properties(r) AS properties
        """

    @staticmethod
    def find_impact_chain(start_entity: str, end_entity: str, max_hops: int = 3) -> str:
        """查找影响链（支持模糊匹配）"""
        return f"""
        MATCH (start), (end)
        WHERE toLower(start.name) CONTAINS toLower($start)
           OR toLower($start) CONTAINS toLower(start.name)
        WHERE toLower(end.name) CONTAINS toLower($end)
           OR toLower($end) CONTAINS toLower(end.name)
        MATCH path = (start)-[r*1..{max_hops}]-(end)
        WHERE all(rel IN relationships(path) WHERE type(rel) IN ['AFFECTED_BY', 'IMPACTS', 'DRIVEN_BY', 'BELONGS_TO'])
        RETURN path,
               [n IN nodes(path) | n.name] AS chain,
               [r IN relationships(path) | type(r)] AS relation_types
        LIMIT 10
        """

    @staticmethod
    def find_sector_companies(sector_name: str) -> str:
        """查找行业/板块内的公司（支持模糊匹配）"""
        return """
        MATCH (s:Sector)
        WHERE toLower(s.name) CONTAINS toLower($sector)
           OR toLower($sector) CONTAINS toLower(s.name)
        MATCH (s)<-[:BELONGS_TO]-(c:Company)
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
    # 芯片供应链专用查询
    # =========================

    @staticmethod
    def find_supply_chain(company_name: str) -> str:
        """查询某公司的完整供应链（上游代工厂 + 下游客户）"""
        return """
        MATCH (c:Company)
        WHERE toLower(c.name) CONTAINS toLower($company)
           OR toLower($company) CONTAINS toLower(c.name)
        OPTIONAL MATCH upstream = (c)-[:OUTSOURCES_TO]->(f:Foundry)-[:DEPENDS_ON]->(m:Material)
        OPTIONAL MATCH downstream = (customer:Company)-[:PURCHASES]->(p:Product)<-[:DESIGNS]-(c)
        RETURN c.name AS company,
               collect(DISTINCT {foundry: f.name, material: m.name, relation: '上游依赖'}) AS upstream,
               collect(DISTINCT {customer: customer.name, product: p.name, relation: '下游客户'}) AS downstream
        """

    @staticmethod
    def find_risk_impact(foundry_name: str) -> str:
        """查询代工厂停产/事故影响的上市公司"""
        return """
        MATCH (f:Foundry)
        WHERE toLower(f.name) CONTAINS toLower($foundry)
           OR toLower($foundry) CONTAINS toLower(f.name)
        OPTIONAL MATCH (f)<-[:OUTSOURCES_TO]-(c:Company)
        OPTIONAL MATCH (f)-[:DEPENDS_ON]->(m:Material)
        OPTIONAL MATCH (c)-[:DESIGNS]->(p:Product)<-[:PURCHASES]-(customer:Company)
        RETURN f.name AS foundry,
               collect(DISTINCT {company: c.name, code: c.code, relation: '委托代工'}) AS affected_designers,
               collect(DISTINCT {material: m.name, relation: '依赖材料'}) AS dependent_materials,
               collect(DISTINCT {customer: customer.name, product: p.name, relation: '下游客户'}) AS downstream_customers
        """

    @staticmethod
    def find_competitors(company_name: str) -> str:
        """查询某公司的竞争对手（同一细分领域）"""
        return """
        MATCH (c:Company)
        WHERE toLower(c.name) CONTAINS toLower($company)
           OR toLower($company) CONTAINS toLower(c.name)
        OPTIONAL MATCH (c)-[r:COMPETES_WITH]-(competitor:Company)
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(s:Sector)<-[:BELONGS_TO]-(peer:Company)
        WHERE peer <> c
        RETURN c.name AS company,
               collect(DISTINCT {name: competitor.name, field: r.竞争领域, relation: '直接竞争'}) AS direct_competitors,
               collect(DISTINCT {name: peer.name, sector: s.name, relation: '同板块'}) AS peer_companies
        """

    @staticmethod
    def find_foundries_by_process(process_node: str) -> str:
        """查询拥有特定工艺节点的代工厂"""
        return """
        MATCH (f:Foundry)
        WHERE toLower(f.工艺节点) CONTAINS toLower($process)
           OR toLower($process) CONTAINS toLower(f.工艺节点)
        RETURN f.name AS foundry,
               f.location AS location,
               f.工艺节点 AS process_node,
               f.月产能 AS capacity
        ORDER BY f.月产能 DESC
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


# =========================
# 🔹 可视化
# =========================
NODE_COLORS = {
    "Company": "#3b82f6",    # 蓝色
    "Sector": "#10b981",     # 绿色
    "Product": "#f59e0b",    # 橙色
    "Foundry": "#ef4444",    # 红色
    "Material": "#8b5cf6",   # 紫色
    "Location": "#6b7280",   # 灰色
    "Event": "#eab308",      # 黄色
    "Person": "#ec4899",     # 粉色
    "Asset": "#14b8a6",      # 青色
    "Indicator": "#f97316",  # 深橙
}


def visualize_kg(max_nodes: int = 500, focus_entity: str = None) -> str:
    """
    生成知识图谱可视化 HTML（Pyvis）

    Args:
        max_nodes: 最大节点数
        focus_entity: 聚焦某个实体（只显示其 N 跳邻居）

    Returns:
        HTML 字符串
    """
    from pyvis.network import Network

    conn = get_neo4j_connection()
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)

    # 获取节点
    if focus_entity:
        node_query = """
        MATCH path = (focus)-[*1..2]-(neighbor)
        WHERE toLower(focus.name) CONTAINS toLower($name)
           OR toLower($name) CONTAINS toLower(focus.name)
        WITH focus, neighbor
        LIMIT $limit
        RETURN DISTINCT labels(focus)[0] AS label, focus.name AS name, elementId(focus) AS nid
        UNION
        RETURN DISTINCT labels(neighbor)[0] AS label, neighbor.name AS name, elementId(neighbor) AS nid
        """
        node_results = conn.execute_query(node_query, {"name": focus_entity, "limit": max_nodes // 2})
    else:
        node_query = """
        MATCH (n)
        RETURN labels(n)[0] AS label, n.name AS name, elementId(n) AS nid
        LIMIT $limit
        """
        node_results = conn.execute_query(node_query, {"limit": max_nodes})

    added_nodes = set()
    for row in node_results:
        nid = row.get("nid")
        name = row.get("name", "")
        label = row.get("label", "Unknown")
        if not name or nid in added_nodes:
            continue
        color = NODE_COLORS.get(label, "#9ca3af")
        net.add_node(
            n_id=nid,
            label=name,
            title=f"{label}: {name}",
            color=color,
            size=25 if label == "Company" else 18,
            font={"size": 14 if label == "Company" else 12}
        )
        added_nodes.add(nid)

    # 获取关系（只获取已添加节点之间的关系）
    if added_nodes:
        rel_query = """
        MATCH (a)-[r]->(b)
        WHERE elementId(a) IN $node_ids AND elementId(b) IN $node_ids
        RETURN elementId(a) AS src, elementId(b) AS dst, type(r) AS rel_type, properties(r) AS props
        LIMIT 2000
        """
        rel_results = conn.execute_query(rel_query, {"node_ids": list(added_nodes)})
        for row in rel_results:
            src = row.get("src")
            dst = row.get("dst")
            rel_type = row.get("rel_type", "")
            if src in added_nodes and dst in added_nodes:
                net.add_edge(src, dst, title=rel_type, label=rel_type, arrows="to")

    # 配置选项
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 150
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200
      }
    }
    """)

    return net.generate_html()


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
