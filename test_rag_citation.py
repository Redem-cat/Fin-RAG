"""
测试 RAG 系统集成引用追踪
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'f:/py/Bishe/langchain-ollama-elasticsearch')

# 测试 ask_question 函数
from src.rag import ask_question

# 测试问题
questions = [
    "什么是股票？",
    "如何选择优质股票？",
    "A股市场最近怎么样？"
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"问题: {q}")
    print('='*60)
    
    result = ask_question(q)
    
    print(f"\n回答: {result['answer'][:100]}...")
    print(f"\n引用摘要: {result.get('citation_summary', '无')}")
    print(f"来源数量: {len(result.get('source', []))}")