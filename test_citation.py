"""
测试引用追踪功能
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'f:/py/Bishe/langchain-ollama-elasticsearch')

from src.citation import CitationTracker, CitationSource, SourceType

# 测试引用追踪器
tracker = CitationTracker()

# 模拟检索结果
sources = [
    CitationSource(
        source_type=SourceType.VECTOR_SEARCH,
        source_name="雪球文章",
        title="如何选择优质股票",
        content="选择股票要看基本面...",
        confidence=0.95,
        metadata={"page": "第5页"}
    ),
    CitationSource(
        source_type=SourceType.VECTOR_SEARCH,
        source_name="东方财富研报",
        title="2024年A股投资策略",
        content="预计2024年...",
        confidence=0.88
    ),
    CitationSource(
        source_type=SourceType.KNOWLEDGE_GRAPH,
        source_name="Neo4j 知识图谱",
        title="Company",
        content="...",
        confidence=0.92
    ),
    CitationSource(
        source_type=SourceType.FINANCE_API,
        source_name="AKShare 实时行情",
        title="上证指数",
        content="3200.50",
        confidence=1.0
    )
]

# 模拟问题和回答
question = "如何选择优质的股票进行投资？"
answer = "选择优质股票需要综合考虑多个因素..."

# 记录引用
tracker.record(question, answer, sources)

# 打印摘要
summary = tracker.format_summary(sources)
print("=" * 50)
print("引用追踪测试结果：")
print("=" * 50)
print(summary)
print("=" * 50)

# 测试统计功能
stats = tracker.get_statistics()
print("\n详细统计：")
for key, value in stats.items():
    print(f"  {key}: {value}")
