"""
实体抽取模块

功能：
- 从文本中抽取金融实体
- 支持新闻、公告等多种来源
- 使用 LLM 和规则混合方法
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """抽取的实体"""
    name: str
    entity_type: str  # Company, Person, Sector, Event, Indicator
    properties: Dict = field(default_factory=dict)
    confidence: float = 1.0
    source: str = ""


@dataclass
class ExtractedRelation:
    """抽取的关系"""
    source: str
    target: str
    relation_type: str
    properties: Dict = field(default_factory=dict)
    confidence: float = 1.0


class EntityExtractor:
    """实体抽取器"""
    
    def __init__(self):
        self.llm = None
        
        # 预定义实体模式（规则方法）
        self.patterns = {
            "stock_code": r'(?:股票代码|代码|证券代码)[：:]\s*(\d{6})',
            "company": r'([\u4e00-\u9fa5]{2,10}(?:股份|集团|公司|银行|证券|保险))',
            "person": r'([\u4e00-\u9fa5]{2,4})(?:董事长|总裁|CEO|总经理|高管)',
            "money": r'(\d+(?:\.\d+)?)\s*(?:亿元|万元|元|美元|亿)',
            "percent": r'(\d+(?:\.\d+)?)\s*%',
            "date": r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)',
        }
        
        # 行业关键词
        self.sector_keywords = [
            "科技", "金融", "医药", "消费", "能源", "汽车", "房地产",
            "制造业", "新能源", "半导体", "人工智能", "互联网",
            "银行", "保险", "证券", "基金", "信托"
        ]
    
    def extract_from_text(self, text: str) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """
        从文本中抽取实体和关系
        
        Args:
            text: 输入文本
            
        Returns:
            (实体列表, 关系列表)
        """
        entities = []
        relations = []
        
        # 1. 规则方法：快速抽取明确实体
        rule_entities = self._extract_by_rules(text)
        entities.extend(rule_entities)
        
        # 2. LLM 方法：抽取隐含实体和关系（可选）
        # 这里可以选择是否使用 LLM
        
        return entities, relations
    
    def _extract_by_rules(self, text: str) -> List[ExtractedEntity]:
        """规则方法抽取实体"""
        entities = []
        seen = set()
        
        # 股票代码
        for match in re.finditer(self.patterns["stock_code"], text):
            code = match.group(1)
            if code not in seen:
                entities.append(ExtractedEntity(
                    name=code,
                    entity_type="Asset",
                    properties={"code": code},
                    confidence=0.95,
                    source="rule"
                ))
                seen.add(code)
        
        # 公司名称
        for match in re.finditer(self.patterns["company"], text):
            name = match.group(1)
            if name not in seen and len(name) <= 10:
                entities.append(ExtractedEntity(
                    name=name,
                    entity_type="Company",
                    properties={},
                    confidence=0.9,
                    source="rule"
                ))
                seen.add(name)
        
        # 人物
        for match in re.finditer(self.patterns["person"], text):
            name = match.group(1)
            if name not in seen:
                entities.append(ExtractedEntity(
                    name=name,
                    entity_type="Person",
                    properties={},
                    confidence=0.85,
                    source="rule"
                ))
                seen.add(name)
        
        # 行业
        for sector in self.sector_keywords:
            if sector in text and sector not in seen:
                entities.append(ExtractedEntity(
                    name=sector,
                    entity_type="Sector",
                    properties={},
                    confidence=0.8,
                    source="rule"
                ))
                seen.add(sector)
        
        return entities
    
    def extract_with_llm(self, text: str) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """
        使用 LLM 抽取实体和关系
        
        Args:
            text: 输入文本
            
        Returns:
            (实体列表, 关系列表)
        """
        if self.llm is None:
            try:
                from src.llm_client import get_llm
                self.llm = get_llm(temperature=0.1)
            except Exception as e:
                logger.warning(f"LLM 加载失败: {e}")
                return [], []
        
        prompt = f"""请从以下财经新闻中抽取实体和关系。

文本：
{text}

请抽取：
1. 公司实体（Company）
2. 人物实体（Person）
3. 行业实体（Sector）
4. 事件实体（Event）

并识别它们之间的关系。

输出 JSON 格式：
{{
    "entities": [
        {{"name": "实体名", "type": "类型", "properties": {{}}}}
    ],
    "relations": [
        {{"source": "实体A", "target": "实体B", "type": "关系类型"}}
    ]
}}
"""
        
        try:
            response = self.llm.invoke(prompt)
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
                        confidence=0.9,
                        source="llm"
                    )
                    for e in data.get("entities", [])
                ]
                
                relations = [
                    ExtractedRelation(
                        source=r["source"],
                        target=r["target"],
                        relation_type=r["type"],
                        confidence=0.9
                    )
                    for r in data.get("relations", [])
                ]
                
                return entities, relations
                
        except Exception as e:
            logger.error(f"LLM 抽取失败: {e}")
        
        return [], []


def extract_entities_from_news(news_data: Dict, use_llm: bool = False) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
    """
    从新闻数据中抽取实体
    
    Args:
        news_data: 新闻数据（包含 title, content）
        use_llm: 是否使用 LLM
        
    Returns:
        (实体列表, 关系列表)
    """
    extractor = EntityExtractor()
    
    # 合并标题和内容
    text = f"{news_data.get('title', '')}\n{news_data.get('content', '')}"
    
    if use_llm:
        return extractor.extract_with_llm(text)
    else:
        return extractor.extract_from_text(text)


# =========================
# 🔹 测试代码
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("实体抽取测试")
    print("=" * 60)
    
    extractor = EntityExtractor()
    
    # 测试文本
    test_text = """
    平安银行股份有限公司（股票代码：000001）今日发布公告，
    董事长张三表示，公司将在新能源和科技领域加大投资，
    预计投资金额超过100亿元。
    """
    
    entities, relations = extractor.extract_from_text(test_text)
    
    print("\n抽取的实体:")
    for e in entities:
        print(f"  - {e.name} ({e.entity_type}) 置信度: {e.confidence}")
