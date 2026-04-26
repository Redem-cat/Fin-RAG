"""
意图分类模块 - 从 .env 读取关键词配置
支持: INVESTMENT(投资顾问) vs POLICY(政策咨询)
"""

import os
import re
from typing import Tuple, List
from enum import Enum
from dotenv import load_dotenv

# 加载环境变量
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_path, "elastic-start-local/.env"))


class Intent(Enum):
    """用户意图枚举"""
    INVESTMENT = "investment"      # 投资顾问（股票、基金、理财等）
    POLICY = "policy"              # 政策咨询（法律法规、政府文件等）
    GENERAL = "general"            # 通用问答
    MIXED = "mixed"               # 混合意图


def _load_keywords_from_env(env_key: str, default: List[str]) -> List[str]:
    """从环境变量加载关键词列表"""
    keywords_str = os.getenv(env_key, "")
    if keywords_str:
        return [k.strip() for k in keywords_str.split(",") if k.strip()]
    return default


# =========================
# 🔹 投资相关关键词（可从 .env 覆盖）
# =========================
DEFAULT_INVESTMENT_KEYWORDS = [
    # 金融产品
    "股票", "基金", "ETF", "LOF", "FOF", "QDII", "债券", "国债", "理财",
    "私募", "公募", "信托", "保险", "黄金", "原油", "期货", "期权",
    # 投资行为
    "投资", "理财", "收益", "回报", "涨幅", "跌幅", "涨停", "跌停",
    "市值", "估值", "PE", "PB", "股息", "分红", "持仓", "建仓",
    "买入", "卖出", "止损", "止盈", "抄底", "逃顶", "补仓",
    # 市场数据
    "行情", "走势", "K线", "均线", "成交量", "大盘", "指数",
    "上证", "深证", "创业板", "科创板", "北交所", "港股", "美股",
    # 具体金融实体
    "茅台", "宁德", "比亚迪", "苹果", "特斯拉", "马斯克",
    # 机构/人物
    "基金经理", "巴菲特", "索罗斯", "徐翔",
    # 宏观金融
    "利率", "降息", "加息", "LPR", "SHIBOR", "GDP", "CPI", "PPI",
    "货币政策", "财政政策", "MLF", "逆回购", "QE",
]

INVESTMENT_KEYWORDS = _load_keywords_from_env(
    "INTENT_KEYWORDS_INVESTMENT", 
    DEFAULT_INVESTMENT_KEYWORDS
)


# =========================
# 🔹 政策相关关键词（可从 .env 覆盖）
# =========================
DEFAULT_POLICY_KEYWORDS = [
    # 法律类型
    "法律", "法规", "条例", "规章", "办法", "规定", "细则",
    "司法解释", "指导意见", "通知", "公告", "批复", "函",
    # 政策类型
    "政策", "方针", "路线", "战略", "规划", "方案", "计划",
    "改革", "试点", "示范", "创新", "扶持", "补贴", "优惠",
    # 政府机构
    "国务院", "银保监会", "央行", "财政部", "发改委",
    "证监会", "交易所", "金融监管",
    # 文件类型
    "文件", "白皮书", "蓝皮书", "报告", "皮书", "年鉴",
    # 行为词
    "违法", "合规", "违规", "处罚", "追究", "责任", "权利", "义务",
]

POLICY_KEYWORDS = _load_keywords_from_env(
    "INTENT_KEYWORDS_POLICY", 
    DEFAULT_POLICY_KEYWORDS
)


class IntentClassifier:
    """意图分类器"""
    
    def __init__(self):
        self.llm = None
        
    def _get_llm(self):
        """延迟加载 LLM（使用 DeepSeek API）"""
        if self.llm is None:
            from src.llm_client import get_llm
            self.llm = get_llm(temperature=0.1)
        return self.llm
    
    def classify(self, question: str) -> Tuple[Intent, str]:
        """分类用户意图"""
        intent, reason = self._keyword_classify(question)
        return intent, reason
    
    def _keyword_classify(self, question: str) -> Tuple[Intent, str]:
        """基于关键词的快速分类"""
        question_lower = question.lower()
        
        investment_score = sum(1 for kw in INVESTMENT_KEYWORDS if kw in question_lower)
        policy_score = sum(1 for kw in POLICY_KEYWORDS if kw in question_lower)
        
        if investment_score > 0 and policy_score > 0:
            return Intent.MIXED, f"同时涉及投资({investment_score}个关键词)和政策({policy_score}个关键词)"
        elif investment_score > policy_score:
            return Intent.INVESTMENT, f"涉及投资领域({investment_score}个关键词)"
        elif policy_score > investment_score:
            return Intent.POLICY, f"涉及政策法规({policy_score}个关键词)"
        else:
            return Intent.GENERAL, "未识别明确领域"
    
    def _llm_classify(self, question: str) -> Tuple[Intent, str]:
        """使用LLM进行意图分类"""
        prompt = f"""请分析用户问题的意图类型。

问题: {question}

请从以下三类中选择最合适的类型：
1. investment - 投资顾问类（股票、基金、理财、投资建议、市场分析等）
2. policy - 政策咨询类（法律法规、政府文件、政策解读、合规咨询等）
3. general - 通用问答类（不属于上述两类）

请按以下格式回答：
INTENT: [investment/policy/general]
REASON: [简短说明理由，不超过30字]
"""
        
        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            intent_match = re.search(r'INTENT:\s*(\w+)', content, re.IGNORECASE)
            reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
            
            if intent_match:
                intent_str = intent_match.group(1).lower()
                if 'investment' in intent_str:
                    return Intent.INVESTMENT, reason_match.group(1) if reason_match else "LLM判断为投资类"
                elif 'policy' in intent_str:
                    return Intent.POLICY, reason_match.group(1) if reason_match else "LLM判断为政策类"
            
            return Intent.GENERAL, "LLM未能明确分类"
            
        except Exception as e:
            print(f"LLM意图分类失败: {e}")
            return Intent.GENERAL, "LLM调用失败"


# 全局实例
_intent_classifier = None

def get_intent_classifier() -> IntentClassifier:
    """获取意图分类器实例"""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier
