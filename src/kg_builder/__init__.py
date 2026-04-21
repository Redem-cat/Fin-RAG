"""
知识图谱构建模块

功能：
- 从 API 数据源构建知识图谱
- 实体和关系抽取
- Neo4j 写入封装
"""

from .kg_writer import KGWriter, get_kg_writer
from .entity_extractor import EntityExtractor, extract_entities_from_news

__all__ = [
    "KGWriter",
    "get_kg_writer",
    "EntityExtractor",
    "extract_entities_from_news",
]
