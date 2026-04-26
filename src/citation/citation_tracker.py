"""
引用追踪模块

功能：
- 追踪每次回答引用的数据源
- 分类统计引用来源
- 生成引用报告
- 支持溯源展示

参考：成熟大模型应用（如 Perplexity, ChatGPT with Bing）的引用展示方式
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# 🔹 引用来源类型
# =========================
class SourceType(Enum):
    """引用来源类型"""
    VECTOR_SEARCH = "vector_search"      # 向量检索（本地文档）
    KNOWLEDGE_GRAPH = "knowledge_graph"  # 知识图谱
    NEWS_CRAWLER = "news_crawler"        # 新闻爬虫
    FINANCE_API = "finance_api"          # 金融 API（AKShare/Tushare）
    LLM_KNOWLEDGE = "llm_knowledge"      # LLM 内置知识
    WEB_SEARCH = "web_search"            # 网络搜索（预留）


@dataclass
class CitationSource:
    """引用来源"""
    source_type: SourceType
    source_name: str          # 来源名称
    title: str                # 标题
    content: str              # 内容片段
    url: str = ""             # 链接（如果有）
    confidence: float = 0.0   # 置信度
    metadata: Dict = field(default_factory=dict)  # 额外元数据
    
    def to_dict(self) -> Dict:
        return {
            "source_type": self.source_type.value,
            "source_name": self.source_name,
            "title": self.title,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "url": self.url,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class CitationRecord:
    """引用记录"""
    question: str
    answer: str
    sources: List[CitationSource]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_sources: int = 0
    source_breakdown: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        self.total_sources = len(self.sources)
        self.source_breakdown = self._calculate_breakdown()
    
    def _calculate_breakdown(self) -> Dict[str, int]:
        """计算各类型来源数量"""
        breakdown = {}
        for source in self.sources:
            key = source.source_type.value
            breakdown[key] = breakdown.get(key, 0) + 1
        return breakdown
    
    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "timestamp": self.timestamp,
            "total_sources": self.total_sources,
            "source_breakdown": self.source_breakdown,
            "sources": [s.to_dict() for s in self.sources]
        }


class CitationTracker:
    """引用追踪器"""
    
    def __init__(self, log_dir: str = None, max_records: int = 100):
        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent / "memory" / "citations"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_records = max_records
        self.history_file = self.log_dir / "citation_history.json"
        self.history: List[Dict] = self._load_history()
        
        # 统计信息
        self.stats = {
            "total_queries": 0,
            "total_citations": 0,
            "source_type_counts": {},
            "avg_sources_per_query": 0.0
        }
    
    def _load_history(self) -> List[Dict]:
        """加载历史记录"""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载引用历史失败: {e}")
        return []
    
    def _save_history(self):
        """保存历史记录"""
        try:
            # 限制记录数量
            if len(self.history) > self.max_records:
                self.history = self.history[-self.max_records:]
            
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存引用历史失败: {e}")
    
    def record(self, question: str, answer: str, sources: List[CitationSource]) -> CitationRecord:
        """
        记录一次查询的引用
        
        Args:
            question: 用户问题
            answer: 回答内容
            sources: 引用来源列表
            
        Returns:
            CitationRecord: 引用记录
        """
        record = CitationRecord(
            question=question,
            answer=answer,
            sources=sources
        )
        
        # 添加到历史
        self.history.append(record.to_dict())
        self._save_history()
        
        # 更新统计
        self._update_stats(record)
        
        logger.info(f"[引用追踪] 记录 {record.total_sources} 个引用来源")
        return record
    
    def _update_stats(self, record: CitationRecord):
        """更新统计信息"""
        self.stats["total_queries"] += 1
        self.stats["total_citations"] += record.total_sources
        
        # 更新来源类型统计
        for source_type, count in record.source_breakdown.items():
            self.stats["source_type_counts"][source_type] = \
                self.stats["source_type_counts"].get(source_type, 0) + count
        
        # 计算平均引用数
        self.stats["avg_sources_per_query"] = \
            self.stats["total_citations"] / self.stats["total_queries"]
    
    def get_report(self, hours: int = 24) -> Dict:
        """
        生成引用报告
        
        Args:
            hours: 时间范围（小时）
            
        Returns:
            报告数据
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # 过滤时间范围内的记录
        recent_records = []
        for record in self.history:
            try:
                record_time = datetime.fromisoformat(record["timestamp"])
                if record_time > cutoff_time:
                    recent_records.append(record)
            except Exception:
                continue
        
        # 统计
        total_sources = sum(r.get("total_sources", 0) for r in recent_records)
        source_breakdown = {}
        
        for record in recent_records:
            for source_type, count in record.get("source_breakdown", {}).items():
                source_breakdown[source_type] = source_breakdown.get(source_type, 0) + count
        
        return {
            "time_range_hours": hours,
            "total_queries": len(recent_records),
            "total_sources": total_sources,
            "source_breakdown": source_breakdown,
            "avg_sources_per_query": total_sources / len(recent_records) if recent_records else 0,
            "recent_records": recent_records[-10:]  # 最近10条
        }
    
    def format_summary(self, sources: List[CitationSource]) -> str:
        """
        格式化引用摘要（用于显示在回答末尾）
        
        Args:
            sources: 引用来源列表
            
        Returns:
            格式化的摘要字符串
        """
        if not sources:
            return ""
        
        # 按类型分组
        type_counts = {}
        for source in sources:
            type_name = self._get_type_display_name(source.source_type)
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # 格式化输出
        lines = [f"\n📊 **数据来源（共 {len(sources)} 条）**"]
        for type_name, count in type_counts.items():
            lines.append(f"- {type_name}：{count} 条")
        
        return "\n".join(lines)
    
    def format_detail(self, sources: List[CitationSource], max_display: int = 5) -> str:
        """
        格式化详细引用信息
        
        Args:
            sources: 引用来源列表
            max_display: 最大显示数量
            
        Returns:
            格式化的详细信息
        """
        if not sources:
            return ""
        
        lines = [f"\n📚 **引用详情**\n"]
        
        for i, source in enumerate(sources[:max_display], 1):
            type_name = self._get_type_display_name(source.source_type)
            lines.append(f"**{i}. {source.title}** ({type_name})")
            
            if source.content:
                content_preview = source.content[:150] + "..." if len(source.content) > 150 else source.content
                lines.append(f"   {content_preview}")
            
            if source.url:
                lines.append(f"   🔗 [来源]({source.url})")
            
            lines.append("")
        
        if len(sources) > max_display:
            lines.append(f"_...还有 {len(sources) - max_display} 条来源_")
        
        return "\n".join(lines)
    
    def _get_type_display_name(self, source_type: SourceType) -> str:
        """获取来源类型显示名称"""
        names = {
            SourceType.VECTOR_SEARCH: "📄 文档检索",
            SourceType.KNOWLEDGE_GRAPH: "🕸️ 知识图谱",
            SourceType.NEWS_CRAWLER: "📰 财经新闻",
            SourceType.FINANCE_API: "📈 实时行情",
            SourceType.LLM_KNOWLEDGE: "🧠 模型知识",
            SourceType.WEB_SEARCH: "🌐 网络搜索",
        }
        return names.get(source_type, source_type.value)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()


# =========================
# 🔹 全局实例
# =========================
_tracker: Optional[CitationTracker] = None


def get_citation_tracker(log_dir: str = None) -> CitationTracker:
    """获取引用追踪器实例（单例）"""
    global _tracker
    if _tracker is None:
        _tracker = CitationTracker(log_dir)
    return _tracker


# =========================
# 🔹 便捷函数
# =========================
def track_citations(
    question: str,
    answer: str,
    vector_results: List[Dict] = None,
    kg_results: List[Dict] = None,
    news_results: List[Dict] = None,
    finance_data: Dict = None
) -> CitationRecord:
    """
    便捷函数：追踪所有来源的引用
    
    Args:
        question: 用户问题
        answer: 回答内容
        vector_results: 向量检索结果
        kg_results: 知识图谱结果
        news_results: 新闻结果
        finance_data: 金融数据
        
    Returns:
        CitationRecord: 引用记录
    """
    sources = []
    
    # 向量检索来源
    if vector_results:
        for result in vector_results:
            sources.append(CitationSource(
                source_type=SourceType.VECTOR_SEARCH,
                source_name=result.get("source", "未知来源"),
                title=result.get("title", "文档片段"),
                content=result.get("content", ""),
                confidence=result.get("similarity", 0.0),
                metadata={"page": result.get("page", "")}
            ))
    
    # 知识图谱来源
    if kg_results:
        for entity in kg_results:
            sources.append(CitationSource(
                source_type=SourceType.KNOWLEDGE_GRAPH,
                source_name="Neo4j 知识图谱",
                title=entity.get("name", "实体"),
                content=f"{entity.get('type', '')}: {entity.get('relation', '')}",
                confidence=entity.get("confidence", 0.8)
            ))
    
    # 新闻来源
    if news_results:
        for news in news_results:
            sources.append(CitationSource(
                source_type=SourceType.NEWS_CRAWLER,
                source_name=news.get("source", "新闻"),
                title=news.get("title", ""),
                content=news.get("content", ""),
                url=news.get("url", ""),
                metadata={"publish_time": news.get("publish_time", "")}
            ))
    
    # 金融 API 来源
    if finance_data:
        sources.append(CitationSource(
            source_type=SourceType.FINANCE_API,
            source_name=finance_data.get("source", "金融 API"),
            title=finance_data.get("title", "实时数据"),
            content=json.dumps(finance_data.get("data", {}), ensure_ascii=False),
            confidence=1.0
        ))
    
    tracker = get_citation_tracker()
    return tracker.record(question, answer, sources)


# =========================
# 🔹 测试代码
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("引用追踪测试")
    print("=" * 60)
    
    tracker = get_citation_tracker()
    
    # 模拟引用
    sources = [
        CitationSource(
            source_type=SourceType.VECTOR_SEARCH,
            source_name="基金招募说明书.pdf",
            title="投资策略",
            content="本基金采用量化投资策略...",
            confidence=0.92
        ),
        CitationSource(
            source_type=SourceType.KNOWLEDGE_GRAPH,
            source_name="Neo4j",
            title="宁德时代",
            content="Company: 新能源汽车",
            confidence=0.85
        ),
        CitationSource(
            source_type=SourceType.FINANCE_API,
            source_name="AKShare",
            title="实时行情",
            content="000001 平安银行 价格: 10.5",
            confidence=1.0
        ),
    ]
    
    # 记录
    record = tracker.record(
        question="平安银行的投资价值如何？",
        answer="根据分析，平安银行...",
        sources=sources
    )
    
    # 显示摘要
    print(tracker.format_summary(sources))
    
    # 显示详情
    print(tracker.format_detail(sources))
    
    # 报告
    print("\n统计报告:")
    print(json.dumps(tracker.get_report(hours=1), ensure_ascii=False, indent=2))
