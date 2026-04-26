"""
FinRAG-Agent: 基于 Function Calling 的智能体模式
将检索、金融数据、锐思、知识图谱等能力封装为工具，由 LLM 按需调用
"""

import json
import os
from typing import Dict, List, Any, Optional
from openai import OpenAI

from src.llm_client import get_llm, DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY
from src.config import Config

# 延迟导入工具实现（避免循环依赖）
_vector_db = None
_memory_manager = None
_reranker = None


def _get_vector_db():
    """延迟初始化向量数据库"""
    global _vector_db
    if _vector_db is None:
        from src.rag import _get_vector_db as get_vdb
        _vector_db = get_vdb()
    return _vector_db


def _get_memory_manager():
    """延迟初始化记忆管理器"""
    global _memory_manager
    if _memory_manager is None:
        from src.rag import _get_memory_manager as get_mem
        _memory_manager = get_mem()
    return _memory_manager


# =========================
# 🔹 工具定义
# =========================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "从金融法规知识库中检索与用户问题相关的文档片段。当用户询问金融法规、政策、合规要求、投资规则等问题时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "优化的检索查询，应包含用户问题中的核心关键词"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回的文档数量，默认3",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "akshare_quote",
            "description": "获取股票、指数、基金的实时行情数据（价格、涨跌幅、成交量等）。当用户询问具体股票/指数/基金的行情、走势、涨跌时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "股票/指数代码，如 '000001'（平安银行）、'HSI'（恒生指数）"
                    },
                    "market": {
                        "type": "string",
                        "description": "市场类型: 'sh'(上海)、'sz'(深圳)、'hk'(港股)、'us'(美股)",
                        "enum": ["sh", "sz", "hk", "us"]
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "resset_fetch",
            "description": "从锐思(RESSET)数据库获取财经文本数据（年报、研报、政府报告、股评等）。当用户需要查看上市公司报告、行业研究、政府政策文件、股评资讯时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_type": {
                        "type": "string",
                        "description": "数据类型",
                        "enum": [
                            "cn_report",      # 中国上市公司报告
                            "gov_report",     # 政府工作报告
                            "us_report",      # 美国上市公司报告
                            "financial_news", # 财经新闻
                            "research",       # 研究报告
                            "stock_bar",      # 股吧评论
                            "real_estate"     # 房产拍卖
                        ]
                    },
                    "code": {
                        "type": "string",
                        "description": "股票代码或区域代码。如 '000002'(万科)、'100100'(国务院)"
                    },
                    "subtype": {
                        "type": "string",
                        "description": "子类型。如 '年度报告'、'10K'、'券商研报'等"
                    },
                    "year": {
                        "type": "integer",
                        "description": "年份，如 2023"
                    }
                },
                "required": ["data_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kg_query",
            "description": "从 Neo4j 知识图谱查询公司关系、行业关联、竞争合作等结构化信息。当用户询问公司之间的关系、产业链上下游、竞争格局时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "公司名称，如 '比亚迪'、'腾讯'"
                    },
                    "relation_type": {
                        "type": "string",
                        "description": "关系类型，如 'competes_with'(竞争)、'belongs_to'(属于行业)、'ceo_of'(高管)",
                        "enum": ["competes_with", "belongs_to", "ceo_of", "affected_by", "any"]
                    }
                },
                "required": ["company_name"]
            }
        }
    }
]


# =========================
# 🔹 工具执行实现
# =========================

def execute_rag_search(query: str, top_k: int = 3) -> str:
    """执行向量检索"""
    try:
        vdb = _get_vector_db()
        docs = vdb.similarity_search(query, k=top_k)
        if not docs:
            return "未检索到相关文档。"
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知")
            content = doc.page_content[:500]
            results.append(f"[{i}] 来源: {source}\n{content}")
        return "\n\n".join(results)
    except Exception as e:
        return f"检索失败: {e}"


def execute_akshare_quote(symbol: str, market: Optional[str] = None) -> str:
    """执行 AKShare 行情查询"""
    try:
        from src.finance_data import get_finance_data
        fd = get_finance_data()

        # 尝试识别指数代码
        index_map = {
            "HSI": "hk", "恒生指数": "hk", "恒生科技": "hk",
            "000001": "sh", "上证指数": "sh",
            "399001": "sz", "深证成指": "sz",
            "399006": "sz", "创业板指": "sz",
        }

        # 如果 symbol 是中文名称，尝试查找对应代码
        if symbol in index_map:
            market = index_map[symbol]

        # 获取实时行情
        if market == "hk":
            df = fd.get_index_spot()
            # 筛选港股指数
            matches = [r for r in df if symbol in str(r.get("名称", "")) or symbol in str(r.get("代码", ""))]
        else:
            df = fd.get_stock_spot()
            matches = [r for r in df if symbol in str(r.get("名称", "")) or symbol in str(r.get("代码", ""))]

        if not matches:
            return f"未找到 {symbol} 的行情数据。"

        result = matches[0]
        return (
            f"【{result.get('名称', symbol)} ({result.get('代码', '')})】\n"
            f"最新价: {result.get('最新价', 'N/A')} | "
            f"涨跌幅: {result.get('涨跌幅', 'N/A')}% | "
            f"成交量: {result.get('成交量', 'N/A')} | "
            f"成交额: {result.get('成交额', 'N/A')}"
        )
    except Exception as e:
        return f"行情获取失败: {e}"


def execute_resset_fetch(data_type: str, code: str = "", subtype: str = "", year: int = 0) -> str:
    """执行锐思数据获取"""
    try:
        from src.resset_data import (
            get_cn_company_report, get_government_report,
            get_us_company_report, get_financial_news,
            get_research_report, get_forum_posts, get_real_estate_info,
            format_resset_content
        )

        year_str = str(year) if year else None
        data = []
        label = ""

        if data_type == "cn_report" and code:
            data = get_cn_company_report(code, subtype or "年度报告", year_str)
            label = f"中国上市公司报告-{code}"
        elif data_type == "gov_report":
            data = get_government_report(code or "100100", year_str)
            label = f"政府工作报告-{code or '国务院'}"
        elif data_type == "us_report" and code:
            data = get_us_company_report(code, subtype or "10K", year_str)
            label = f"美股报告-{code}"
        elif data_type == "financial_news":
            data = get_financial_news(year_str)
            label = "财经新闻"
        elif data_type == "research":
            data = get_research_report(subtype or "公司研究", year_str)
            label = f"研究报告-{subtype or '公司研究'}"
        elif data_type == "stock_bar":
            data = get_forum_posts(subtype or "东方财富", year_str)
            label = f"股评-{subtype or '东方财富'}"
        elif data_type == "real_estate":
            data = get_real_estate_info(subtype or "京东拍卖_拍卖公告", year_str)
            label = f"房产拍卖-{subtype or '京东拍卖'}"
        else:
            return f"不支持的数据类型或缺少必要参数: data_type={data_type}, code={code}"

        if data:
            formatted = format_resset_content(data, label)
            return f"【锐思数据 - {label}】\n{formatted[:2000]}"
        else:
            return f"未找到 {label} 的相关数据。"
    except Exception as e:
        return f"锐思数据获取失败: {e}"


def execute_kg_query(company_name: str, relation_type: str = "any") -> str:
    """执行知识图谱查询"""
    try:
        from src.knowledge_graph import get_kg_retriever
        retriever = get_kg_retriever()

        if not retriever.is_available():
            return "知识图谱服务当前不可用（Neo4j 未连接）。"

        # 构建查询问题
        query_map = {
            "competes_with": f"{company_name}的竞争对手有哪些",
            "belongs_to": f"{company_name}属于什么行业",
            "ceo_of": f"{company_name}的高管是谁",
            "affected_by": f"哪些事件影响了{company_name}",
            "any": f"{company_name}的关系网络",
        }
        query_text = query_map.get(relation_type, f"{company_name}的相关信息")

        result = retriever.query(query_text)

        if result.entities:
            lines = [f"【知识图谱 - {company_name}】", result.explanation, ""]
            for e in result.entities[:8]:
                lines.append(f"- {e.get('name', '')} ({e.get('type', '')})")
            return "\n".join(lines)
        else:
            return f"知识图谱中未找到 {company_name} 的相关信息。"
    except Exception as e:
        return f"知识图谱查询失败: {e}"


# =========================
# 🔹 Agent 核心逻辑
# =========================

class FinRAGAgent:
    """FinRAG 智能体：基于 Function Calling 的按需工具调用"""

    def __init__(self):
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.tools = TOOLS

    def run(self, question: str, conversation_history: str = "", top_k: int = 3) -> Dict[str, Any]:
        """
        执行 Agent 流程

        Returns:
            {
                "answer": str,
                "sources": List[Dict],
                "tool_calls": List[Dict],
                "used_context": bool
            }
        """
        # 构建系统提示
        system_prompt = self._build_system_prompt(conversation_history)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        # 第一轮：LLM 判断是否需要工具
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
            temperature=0.3,
        )

        message = response.choices[0].message
        tool_calls = message.tool_calls

        # 如果 LLM 要求调用工具
        if tool_calls:
            # 添加 LLM 的 assistant 消息（包含 tool_calls）
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]
            })

            # 执行工具
            tool_results = []
            for tc in tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                result = self._execute_tool(name, args)
                tool_results.append({"name": name, "args": args, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

            # 第二轮：LLM 基于工具结果生成最终回答
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )
            final_answer = final_response.choices[0].message.content
        else:
            # LLM 直接回答，不需要工具
            final_answer = message.content
            tool_results = []

        return {
            "answer": final_answer,
            "sources": [],  # Agent 模式下 sources 从工具结果中提取
            "tool_calls": tool_results,
            "used_context": len(tool_results) > 0
        }

    def _build_system_prompt(self, conversation_history: str) -> str:
        """构建系统提示"""
        history_section = f"\nRecent conversation:\n{conversation_history}\n" if conversation_history else ""
        return f"""你是一个专业的金融智能助手，名叫 FinRAG-Advisor。

你的职责是：
1. 回答金融法规、投资理财相关问题
2. 提供市场数据分析和投资建议
3. 提示投资风险，确保合规性

你可以使用以下工具来获取信息：
- rag_search: 从金融法规知识库检索文档
- akshare_quote: 获取股票/指数实时行情
- resset_fetch: 获取上市公司报告、研报、政府报告等文本数据
- kg_query: 查询公司关系、产业链等知识图谱信息

规则：
1. 只有当问题确实需要外部数据时才调用工具
2. 对于常识性问题，直接回答，不调用工具
3. 回答要准确、专业，不确定时明确告知
4. 使用与用户问题相同的语言回答
{history_section}"""

    def _execute_tool(self, name: str, args: Dict) -> str:
        """执行指定工具"""
        executors = {
            "rag_search": execute_rag_search,
            "akshare_quote": execute_akshare_quote,
            "resset_fetch": execute_resset_fetch,
            "kg_query": execute_kg_query,
        }
        executor = executors.get(name)
        if executor:
            return executor(**args)
        return f"未知工具: {name}"


# 全局 Agent 实例
_agent_instance: Optional[FinRAGAgent] = None


def get_agent() -> FinRAGAgent:
    """获取 Agent 实例（单例）"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = FinRAGAgent()
    return _agent_instance


def agent_ask(question: str, conversation_history: str = "", top_k: int = 3) -> Dict[str, Any]:
    """便捷的 Agent 调用接口"""
    agent = get_agent()
    return agent.run(question, conversation_history, top_k)
