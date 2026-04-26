"""
知识图谱写入器

功能：
- 将实体和关系写入 Neo4j
- 批量导入支持
- 增量更新机制
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImportStats:
    """导入统计"""
    total_companies: int = 0
    total_persons: int = 0
    total_sectors: int = 0
    total_relations: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class KGWriter:
    """知识图谱写入器"""
    
    def __init__(self):
        self.conn = None
        self._initialized = False
    
    def _ensure_connection(self):
        """确保 Neo4j 连接"""
        if self._initialized:
            return
        
        try:
            from src.knowledge_graph import get_neo4j_connection
            self.conn = get_neo4j_connection()
            self._initialized = True
            logger.info("[KGWriter] Neo4j 连接就绪")
        except Exception as e:
            logger.error(f"[KGWriter] Neo4j 连接失败: {e}")
            self._initialized = False
    
    def write_company(self, company_data: Dict) -> bool:
        """
        写入公司节点
        
        Args:
            company_data: 公司数据
            
        Returns:
            是否成功
        """
        self._ensure_connection()
        
        if not self.conn or not self.conn.is_connected():
            logger.warning("[KGWriter] Neo4j 未连接，跳过写入")
            return False
        
        try:
            query = """
            MERGE (c:Company {code: $code})
            SET c.name = $name,
                c.sector = $sector,
                c.market = $market,
                c.market_cap = $market_cap,
                c.list_date = $list_date,
                c.description = $description,
                c.updated_at = datetime()
            """
            
            params = {
                "code": company_data.get("code", ""),
                "name": company_data.get("name", ""),
                "sector": company_data.get("sector", ""),
                "market": company_data.get("market", ""),
                "market_cap": company_data.get("market_cap", 0),
                "list_date": company_data.get("list_date", ""),
                "description": company_data.get("description", "")
            }
            
            self.conn.execute_write(query, params)
            return True
            
        except Exception as e:
            logger.error(f"[KGWriter] 写入公司 {company_data.get('code')} 失败: {e}")
            return False
    
    def write_sector(self, sector_data: Dict) -> bool:
        """
        写入行业节点
        
        Args:
            sector_data: 行业数据
            
        Returns:
            是否成功
        """
        self._ensure_connection()
        
        if not self.conn or not self.conn.is_connected():
            return False
        
        try:
            query = """
            MERGE (s:Sector {name: $name})
            SET s.code = $code,
                s.parent = $parent,
                s.description = $description,
                s.updated_at = datetime()
            """
            
            params = {
                "name": sector_data.get("name", ""),
                "code": sector_data.get("code", ""),
                "parent": sector_data.get("parent", ""),
                "description": sector_data.get("description", "")
            }
            
            self.conn.execute_write(query, params)
            return True
            
        except Exception as e:
            logger.error(f"[KGWriter] 写入行业 {sector_data.get('name')} 失败: {e}")
            return False
    
    def write_person(self, person_data: Dict) -> bool:
        """
        写入人物节点
        
        Args:
            person_data: 人物数据
            
        Returns:
            是否成功
        """
        self._ensure_connection()
        
        if not self.conn or not self.conn.is_connected():
            return False
        
        try:
            query = """
            MERGE (p:Person {name: $name})
            SET p.title = $title,
                p.company = $company,
                p.age = $age,
                p.education = $education,
                p.updated_at = datetime()
            """
            
            params = {
                "name": person_data.get("name", ""),
                "title": person_data.get("title", ""),
                "company": person_data.get("company", ""),
                "age": person_data.get("age", 0),
                "education": person_data.get("education", "")
            }
            
            self.conn.execute_write(query, params)
            return True
            
        except Exception as e:
            logger.error(f"[KGWriter] 写入人物 {person_data.get('name')} 失败: {e}")
            return False
    
    def write_product(self, product_data: Dict) -> bool:
        """写入产品/部件节点"""
        self._ensure_connection()
        if not self.conn or not self.conn.is_connected():
            return False
        try:
            query = """
            MERGE (p:Product {name: $name})
            SET p.type = $type,
                p.process_node = $process_node,
                p.description = $description,
                p.updated_at = datetime()
            """
            self.conn.execute_write(query, {
                "name": product_data.get("name", ""),
                "type": product_data.get("type", ""),
                "process_node": product_data.get("process_node", ""),
                "description": product_data.get("description", "")
            })
            return True
        except Exception as e:
            logger.error(f"[KGWriter] 写入产品 {product_data.get('name')} 失败: {e}")
            return False

    def write_foundry(self, foundry_data: Dict) -> bool:
        """写入代工厂节点"""
        self._ensure_connection()
        if not self.conn or not self.conn.is_connected():
            return False
        try:
            query = """
            MERGE (f:Foundry {name: $name})
            SET f.location = $location,
                f.process_node = $process_node,
                f.capacity = $capacity,
                f.description = $description,
                f.updated_at = datetime()
            """
            self.conn.execute_write(query, {
                "name": foundry_data.get("name", ""),
                "location": foundry_data.get("location", ""),
                "process_node": foundry_data.get("process_node", ""),
                "capacity": foundry_data.get("capacity", ""),
                "description": foundry_data.get("description", "")
            })
            return True
        except Exception as e:
            logger.error(f"[KGWriter] 写入代工厂 {foundry_data.get('name')} 失败: {e}")
            return False

    def write_material(self, material_data: Dict) -> bool:
        """写入原材料/设备节点"""
        self._ensure_connection()
        if not self.conn or not self.conn.is_connected():
            return False
        try:
            query = """
            MERGE (m:Material {name: $name})
            SET m.type = $type,
                m.supplier = $supplier,
                m.description = $description,
                m.updated_at = datetime()
            """
            self.conn.execute_write(query, {
                "name": material_data.get("name", ""),
                "type": material_data.get("type", ""),
                "supplier": material_data.get("supplier", ""),
                "description": material_data.get("description", "")
            })
            return True
        except Exception as e:
            logger.error(f"[KGWriter] 写入材料 {material_data.get('name')} 失败: {e}")
            return False

    def write_location(self, location_data: Dict) -> bool:
        """写入地点节点"""
        self._ensure_connection()
        if not self.conn or not self.conn.is_connected():
            return False
        try:
            query = """
            MERGE (l:Location {name: $name})
            SET l.type = $type,
                l.country = $country,
                l.updated_at = datetime()
            """
            self.conn.execute_write(query, {
                "name": location_data.get("name", ""),
                "type": location_data.get("type", ""),
                "country": location_data.get("country", "")
            })
            return True
        except Exception as e:
            logger.error(f"[KGWriter] 写入地点 {location_data.get('name')} 失败: {e}")
            return False

    def write_relation(self, source: str, target: str, relation_type: str, 
                       source_type: str = "Company", target_type: str = "Sector",
                       properties: Dict = None) -> bool:
        """
        写入关系（支持属性，如工艺节点、占比、依赖程度、时间）
        """
        self._ensure_connection()
        
        if not self.conn or not self.conn.is_connected():
            return False
        
        try:
            # 使用模糊匹配来定位节点，避免名称微小差异导致关系建立失败
            query = f"""
            MATCH (source:{source_type})
            WHERE toLower(source.name) CONTAINS toLower($source_name)
               OR toLower($source_name) CONTAINS toLower(source.name)
            WITH source
            MATCH (target:{target_type})
            WHERE toLower(target.name) CONTAINS toLower($target_name)
               OR toLower($target_name) CONTAINS toLower(target.name)
            MERGE (source)-[r:{relation_type}]->(target)
            SET r += $properties, r.updated_at = datetime()
            """
            
            props = properties or {}
            # 自动添加时间戳
            if "start_time" not in props:
                from datetime import datetime
                props["start_time"] = datetime.now().isoformat()
            
            params = {
                "source_name": source,
                "target_name": target,
                "properties": props
            }
            
            self.conn.execute_write(query, params)
            return True
            
        except Exception as e:
            logger.error(f"[KGWriter] 写入关系 {source}-{relation_type}->{target} 失败: {e}")
            return False
    
    def import_companies_batch(self, companies: List[Dict]) -> ImportStats:
        """
        批量导入公司
        
        Args:
            companies: 公司列表
            
        Returns:
            ImportStats: 导入统计
        """
        stats = ImportStats()
        
        for company in companies:
            success = self.write_company(company)
            if success:
                stats.total_companies += 1
            else:
                stats.errors.append(f"公司: {company.get('code', 'unknown')}")
        
        logger.info(f"[KGWriter] 批量导入完成: {stats.total_companies} 公司")
        return stats
    
    def import_sectors_batch(self, sectors: List[Dict]) -> ImportStats:
        """
        批量导入行业
        
        Args:
            sectors: 行业列表
            
        Returns:
            ImportStats: 导入统计
        """
        stats = ImportStats()
        
        for sector in sectors:
            success = self.write_sector(sector)
            if success:
                stats.total_sectors += 1
            else:
                stats.errors.append(f"行业: {sector.get('name', 'unknown')}")
        
        logger.info(f"[KGWriter] 批量导入完成: {stats.total_sectors} 行业")
        return stats
    
    def build_company_sector_relations(self, company_sector_map: Dict[str, str]) -> int:
        """
        构建公司-行业关系
        
        Args:
            company_sector_map: {公司名: 行业名}
            
        Returns:
            成功数量
        """
        success_count = 0
        
        for company_name, sector_name in company_sector_map.items():
            if self.write_relation(company_name, sector_name, "BELONGS_TO"):
                success_count += 1
        
        logger.info(f"[KGWriter] 构建关系: {success_count} 条")
        return success_count
    
    def get_stats(self) -> Dict:
        """获取图谱统计"""
        self._ensure_connection()
        
        if not self.conn or not self.conn.is_connected():
            return {"error": "Neo4j 未连接"}
        
        try:
            # 统计各类型节点数量
            node_query = """
            MATCH (n)
            RETURN labels(n)[0] AS type, count(n) AS count
            ORDER BY count DESC
            """
            node_result = self.conn.execute_query(node_query)
            
            # 统计关系数量
            rel_query = """
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(r) AS count
            ORDER BY count DESC
            """
            rel_result = self.conn.execute_query(rel_query)
            
            return {
                "nodes": {r["type"]: r["count"] for r in node_result},
                "relations": {r["type"]: r["count"] for r in rel_result}
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def clear_all(self) -> bool:
        """清空图谱（谨慎使用）"""
        self._ensure_connection()
        
        if not self.conn or not self.conn.is_connected():
            return False
        
        try:
            self.conn.execute_write("MATCH (n) DETACH DELETE n")
            logger.warning("[KGWriter] 图谱已清空")
            return True
        except Exception as e:
            logger.error(f"[KGWriter] 清空图谱失败: {e}")
            return False


# =========================
# 🔹 全局实例
# =========================
_writer: Optional[KGWriter] = None


def get_kg_writer() -> KGWriter:
    """获取 KG 写入器实例（单例）"""
    global _writer
    if _writer is None:
        _writer = KGWriter()
    return _writer


# =========================
# 🔹 测试代码
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("KG 写入器测试")
    print("=" * 60)
    
    writer = get_kg_writer()
    
    # 写入公司
    print("\n[测试] 写入公司...")
    writer.write_company({
        "code": "000001",
        "name": "平安银行",
        "sector": "银行",
        "market": "深圳证券交易所",
        "market_cap": 2500.5
    })
    
    # 写入行业
    print("\n[测试] 写入行业...")
    writer.write_sector({
        "name": "银行",
        "description": "银行业"
    })
    
    # 写入关系
    print("\n[测试] 写入关系...")
    writer.write_relation("平安银行", "银行", "BELONGS_TO")
    
    # 统计
    print("\n[测试] 获取统计...")
    stats = writer.get_stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
