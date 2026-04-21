"""
智能触发系统 - 统一的触发识别和管理
支持两种模式：
1. 关键词模式：从 .env 读取关键词进行匹配
2. LLM 模式：使用大模型自动识别触发条件

触发数据结构:
{
    "trigger_type": "hyde|reranker|finance|tushare",  # 触发类型
    "confidence": 0.95,                                # 置信度
    "params": {},                                       # 传递给处理函数的参数
    "reason": "检测到金融相关关键词"                      # 识别原因
}
"""

import re
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from dotenv import load_dotenv

# 加载环境变量
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_path, "elastic-start-local/.env"))


class TriggerType(Enum):
    """触发类型枚举"""
    HYDE = "hyde"           # HyDE 检索增强
    RERANKER = "reranker"  # 精排重排
    FINANCE_AKSHARE = "finance_akshare"  # AKShare 金融数据
    FINANCE_TUSHARE = "finance_tushare"  # Tushare 财务数据
    MEMORY = "memory"       # 记忆检索
    COMPLIANCE = "compliance"  # 合规审查
    QUANT = "quant"         # AKQuant 量化策略
    RESSET = "resset"       # 锐思文本分析 API
    KG = "kg"               # Neo4j 知识图谱
    GENERAL = "general"     # 通用处理


@dataclass
class TriggerResult:
    """触发结果数据结构"""
    trigger_type: str
    confidence: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    enabled: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)

    def __bool__(self) -> bool:
        return self.enabled and self.confidence >= 0.5


# =========================
# 🔹 触发器基类
# =========================
class BaseTrigger:
    """触发器基类"""
    
    def __init__(self, trigger_type: TriggerType, config_key: str = None):
        self.trigger_type = trigger_type
        self.config_key = config_key
        self._enabled = self._load_enabled()
    
    def _load_enabled(self) -> bool:
        """从环境变量加载启用状态"""
        if self.config_key:
            return os.getenv(self.config_key, "true").lower() == "true"
        return True
    
    def is_enabled(self) -> bool:
        return self._enabled
    
    def check(self, question: str, context: Dict = None) -> Optional[TriggerResult]:
        """
        检查是否触发
        
        Args:
            question: 用户问题
            context: 额外上下文信息
            
        Returns:
            TriggerResult 或 None
        """
        raise NotImplementedError


# =========================
# 🔹 关键词触发器
# =========================
class KeywordTrigger(BaseTrigger):
    """基于关键词的触发器"""
    
    # 默认关键词配置（可从 .env 覆盖）
    DEFAULT_KEYWORDS: Dict[str, List[str]] = {
        # 金融 AKShare 相关
        "finance_akshare": [
            "股票", "股价", "基金", "净值", "ETF", "指数", "大盘", "上证", "深证",
            "行情", "走势", "涨跌", "收盘", "开盘", "涨停", "跌停", "行业", "板块",
            "LPR", "利率", "SHIBOR", "宏观", "GDP", "CPI", "持仓", "分红"
        ],
        # Tushare 财务相关
        "finance_tushare": [
            "财务", "财报", "年报", "季报", "利润表", "资产负债表", "现金流量表",
            "净利润", "营收", "ROE", "ROA", "毛利率", "负债率", "EPS", "财务报表",
            "分红", "股权质押", "股东", "龙虎榜", "大宗交易"
        ],
        # HyDE 适用场景
        "hyde": [
            "分析", "比较", "评估", "建议", "对比", "解释", "说明", "原因", "如何",
            "为什么", "哪个好", "推荐", "预测", "趋势", "怎么样"
        ],
        # 精排适用场景
        "reranker": [
            "详细", "具体", "准确", "权威", "专业", "完整", "全面"
        ],
        # 记忆检索
        "memory": [
            "之前", "上次", "之前说的", "之前提到的", "记得", "我说过"
        ],
        # 合规审查
        "compliance": [
            "推荐", "买", "卖", "投资", "收益", "回报", "风险", "亏损", "保本"
        ],
        # AKQuant 量化策略
        "quant": [
            "量化", "回测", "策略", "均线", "rsi", "macd", "布林带", "金叉", "死叉",
            "交易", "收益率", "夏普比率", "backtest", "strategy", "quantitative",
            "买入信号", "卖出信号", "止盈", "止损", "仓位", "持仓",
            # ML / 热启动 / 多策略 扩展
            "机器学习", "训练模型", "滚动训练", "walk_forward", "walkforward",
            "热启动", "快照恢复", "checkpoint", "warm_start", "warmstart",
            "多策略", "模拟盘", "组合策略", "slot",
            "xgboost", "lightgbm", "randomforest", "lstm", "transformer",
            "特征工程", "pipeline", "交叉验证", "过拟合", "深度学习",
        ],
        # 锐思文本分析
        "resset": [
            "年报", "季报", "年度报告", "季度报告", "问询函", "招股说明书",
            "政府工作报告", "政府报告", "ipo招股", "10k", "10q",
            "财经资讯", "财经新闻", "新闻资讯",
            "研究报告", "行业分析", "宏观分析", "公司研究", "券商研报",
            "股吧", "东方财富", "雪球", "股民评论",
            "房产拍卖", "法拍", "司法拍卖",
            "锐思", "resset",
            "内控报告", "社会责任报告", "审计报告",
            "业绩说明会", "上市公司公告",
        ],
        # Neo4j 知识图谱
        "kg": [
            "关系", "影响", "关联", "竞争", "合作", "产业链", "上下游", "关联方",
            "股权", "股东", "高管", "为什么", "原因", "分析", "比较", "对比",
            "行业", "公司", "板块", "概念股", "graph", "kg", "知识图谱"
        ]
    }
    
    def __init__(self, trigger_type: TriggerType, keywords: List[str] = None):
        super().__init__(trigger_type)
        self.keywords = keywords or self._load_keywords()
    
    def _load_keywords(self) -> List[str]:
        """从 .env 加载关键词"""
        env_key = f"TRIGGER_KEYWORDS_{self.trigger_type.value.upper()}"
        keywords_str = os.getenv(env_key, "")
        
        if keywords_str:
            # 支持逗号分隔的关键词
            return [k.strip() for k in keywords_str.split(",") if k.strip()]
        
        # 返回默认关键词
        return self.DEFAULT_KEYWORDS.get(self.trigger_type.value, [])
    
    def check(self, question: str, context: Dict = None) -> Optional[TriggerResult]:
        """检查关键词匹配"""
        if not self.is_enabled():
            return None
        
        question_lower = question.lower()
        matched_keywords = []
        
        for keyword in self.keywords:
            if keyword.lower() in question_lower:
                matched_keywords.append(keyword)
        
        if matched_keywords:
            confidence = min(0.5 + 0.1 * len(matched_keywords), 1.0)
            return TriggerResult(
                trigger_type=self.trigger_type.value,
                confidence=confidence,
                params={"keywords": matched_keywords, "question": question},
                reason=f"匹配到关键词: {', '.join(matched_keywords[:5])}"
            )
        
        return None


# =========================
# 🔹 LLM 智能触发器
# =========================
LLM_TRIGGER_PROMPT = """你是一个智能触发识别助手。根据用户问题，判断需要触发哪些处理模块。

## 用户问题
{question}

## 可用的触发类型
1. **hyde** - HyDE 检索增强：适用于需要深度语义理解的复杂问题
2. **reranker** - 精排重排：适用于需要高精准度的专业问题
3. **finance_akshare** - AKShare 实时行情：适用于股票、基金、指数等实时数据查询
4. **finance_tushare** - Tushare 财务数据：适用于财务报表、业绩分析等
5. **memory** - 记忆检索：适用于需要参考历史对话的问题
6. **compliance** - 合规审查：适用于投资建议类问题
7. **kg** - Neo4j 知识图谱：适用于关系分析、影响链、多跳推理
8. **general** - 通用处理：普通问答

## 输出格式（JSON）
{{
    "trigger_type": "类型组合，如 "finance_akshare,hyde"",
    "confidence": 0.0-1.0的置信度,
    "params": {{
        "data_type": "需要的具体数据类型,
        "time_range": "时间范围,
        "stock_code": "如需股票数据,
        "fund_code": "如需基金数据
    }},
    "reason": "判断原因"
}}

请只输出JSON，不要有其他内容："""


class LLMTrigger(BaseTrigger):
    """基于 LLM 的智能触发器"""
    
    def __init__(self, trigger_type: TriggerType, llm_model=None):
        super().__init__(trigger_type)
        self.llm_model = llm_model
    
    def _get_llm(self):
        """延迟加载 LLM"""
        if self.llm_model is None:
            try:
                from langchain_ollama import ChatOllama
                model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
                self.llm_model = ChatOllama(model=model_name, temperature=0.1)
            except Exception as e:
                print(f"[LLMTrigger] LLM 加载失败: {e}")
                return None
        return self.llm_model
    
    def check(self, question: str, context: Dict = None) -> Optional[TriggerResult]:
        """使用 LLM 判断触发"""
        if not self.is_enabled():
            return None
        
        llm = self._get_llm()
        if not llm:
            return None
        
        try:
            prompt = LLM_TRIGGER_PROMPT.format(question=question)
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 解析 JSON 响应
            import json
            # 尝试提取 JSON
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group())
                return TriggerResult(
                    trigger_type=result.get("trigger_type", self.trigger_type.value),
                    confidence=result.get("confidence", 0.8),
                    params=result.get("params", {}),
                    reason=result.get("reason", "LLM 识别")
                )
        except Exception as e:
            print(f"[LLMTrigger] 触发识别失败: {e}")
        
        return None


# =========================
# 🔹 统一触发管理器
# =========================
class TriggerManager:
    """统一触发管理器"""
    
    def __init__(self, use_llm: bool = False):
        """
        Args:
            use_llm: 是否启用 LLM 智能识别
        """
        self.use_llm = use_llm and self._check_llm_available()
        
        # 初始化关键词触发器
        self.triggers: Dict[TriggerType, BaseTrigger] = {
            TriggerType.HYDE: KeywordTrigger(TriggerType.HYDE),
            TriggerType.RERANKER: KeywordTrigger(TriggerType.RERANKER),
            TriggerType.FINANCE_AKSHARE: KeywordTrigger(TriggerType.FINANCE_AKSHARE),
            TriggerType.FINANCE_TUSHARE: KeywordTrigger(TriggerType.FINANCE_TUSHARE),
            TriggerType.MEMORY: KeywordTrigger(TriggerType.MEMORY),
            TriggerType.COMPLIANCE: KeywordTrigger(TriggerType.COMPLIANCE),
            TriggerType.QUANT: KeywordTrigger(TriggerType.QUANT),
            TriggerType.RESSET: KeywordTrigger(TriggerType.RESSET),
            TriggerType.KG: KeywordTrigger(TriggerType.KG),
        }
        
        if self.use_llm:
            self.llm_trigger = LLMTrigger(TriggerType.GENERAL)
            print("[TriggerManager] LLM 智能识别已启用")
        else:
            self.llm_trigger = None
            print("[TriggerManager] 仅使用关键词识别")
    
    def _check_llm_available(self) -> bool:
        """检查 LLM 是否可用"""
        return os.getenv("LLM_TRIGGER_ENABLED", "false").lower() == "true"
    
    def analyze(self, question: str, context: Dict = None) -> List[TriggerResult]:
        """
        分析问题，返回所有触发的模块
        
        Args:
            question: 用户问题
            context: 额外上下文
            
        Returns:
            List[TriggerResult]: 所有触发的结果
        """
        results = []
        
        # 1. 关键词匹配
        for trigger_type, trigger in self.triggers.items():
            if trigger.is_enabled():
                result = trigger.check(question, context)
                if result:
                    results.append(result)
        
        # 2. LLM 智能识别（可选）
        if self.llm_trigger:
            llm_result = self.llm_trigger.check(question, context)
            if llm_result:
                # 合并到已有结果
                for existing in results:
                    if existing.trigger_type in llm_result.trigger_type:
                        # 提升置信度
                        existing.confidence = max(existing.confidence, llm_result.confidence)
                        existing.params.update(llm_result.params)
                        existing.reason += f" + {llm_result.reason}"
        
        # 按置信度排序
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results
    
    def should_trigger(self, trigger_type: str, question: str, 
                       context: Dict = None, threshold: float = 0.5) -> TriggerResult:
        """
        检查特定类型是否应该触发
        
        Args:
            trigger_type: 触发类型
            question: 问题
            context: 上下文
            threshold: 置信度阈值
            
        Returns:
            TriggerResult
        """
        results = self.analyze(question, context)
        
        for result in results:
            if result.trigger_type == trigger_type:
                if result.confidence >= threshold:
                    return result
        
        # 返回默认结果
        return TriggerResult(
            trigger_type=trigger_type,
            confidence=0.0,
            params={},
            reason="未检测到触发条件"
        )
    
    def get_trigger_config(self) -> Dict[str, Any]:
        """获取当前触发配置"""
        config = {
            "use_llm": self.use_llm,
            "triggers": {}
        }
        
        for trigger_type, trigger in self.triggers.items():
            if isinstance(trigger, KeywordTrigger):
                config["triggers"][trigger_type.value] = {
                    "enabled": trigger.is_enabled(),
                    "keyword_count": len(trigger.keywords),
                    "keywords": trigger.keywords[:10]  # 只显示前10个
                }
        
        return config


# =========================
# 🔹 全局实例
# =========================
_trigger_manager: TriggerManager = None


def get_trigger_manager(use_llm: bool = None) -> TriggerManager:
    """
    获取触发管理器实例
    
    Args:
        use_llm: 是否使用 LLM，None 则从环境变量读取
    """
    global _trigger_manager
    
    if _trigger_manager is None:
        if use_llm is None:
            use_llm = os.getenv("LLM_TRIGGER_ENABLED", "false").lower() == "true"
        _trigger_manager = TriggerManager(use_llm=use_llm)
    
    return _trigger_manager


def analyze_triggers(question: str, context: Dict = None) -> List[TriggerResult]:
    """快捷函数：分析问题触发哪些模块"""
    manager = get_trigger_manager()
    return manager.analyze(question, context)


# =========================
# 🔹 .env 配置示例（已移至 .env 文件）
# =========================
