"""
金融数据自动触发模块
当检测到用户问题涉及金融领域时，自动查询实时数据并补充到RAG上下文中
关键词从 .env 文件读取
"""

import re
import os
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_path, "elastic-start-local/.env"))


def _load_keywords_from_env(env_key: str, default: List[str]) -> List[str]:
    """从环境变量加载关键词列表"""
    keywords_str = os.getenv(env_key, "")
    if keywords_str:
        return [k.strip() for k in keywords_str.split(",") if k.strip()]
    return default


# 金融领域关键词（从 .env 加载，可覆盖）
DEFAULT_FINANCE_KEYWORDS = {
    # 股票相关
    "stock": ["股票", "股价", "股市", "大盘", "上证", "深证", "创业板", "科创板", 
              "涨停", "跌停", "涨幅", "跌幅", "收盘", "开盘", "竞价", "龙虎榜",
              "A股", "港股", "美股", "沪市", "深市", "北交所", "ETF", "LOF"],
    # 基金相关
    "fund": ["基金", "净值", "ETF", "LOF", "FOF", "QDII", "公募", "私募",
             "基金经理", "持仓", "重仓", "分红", "申购", "赎回", "定投"],
    # 指数相关
    "index": ["指数", "沪深300", "上证指数", "深证成指", "创业板指", "科创50",
              "中证500", "中证1000", "上证50", "道琼斯", "纳斯达克", "标普"],
    # 宏观相关
    "macro": ["利率", "LPR", "SHIBOR", "GDP", "CPI", "PPI", "降息", "加息",
              "货币政策", "财政政策", "MLF", "逆回购", "正回购"],
    # 通用金融
    "general": ["行情", "走势", "数据", "实时", "交易", "市值", "估值", "PE", "PB",
                "股息", "红利", "ROE", "ROA", "净利润", "营收", "财报", "年报", "季报"]
}

# 从 .env 加载（使用 FINANCE_KEYWORDS_ 前缀）
FINANCE_KEYWORDS = {
    category: _load_keywords_from_env(f"FINANCE_KEYWORDS_{category.upper()}", keywords)
    for category, keywords in DEFAULT_FINANCE_KEYWORDS.items()
}

# 股票代码正则 (支持 sh600519, sz000001, 600519, 000001 等格式)
STOCK_CODE_PATTERN = re.compile(r'(sh|sz)?(\d{6})')
# 基金代码正则 (支持 161039, 710001 等格式)
FUND_CODE_PATTERN = re.compile(r'^(\d{6})$')

# 常用指数代码映射
INDEX_CODE_MAP = {
    "上证指数": "sh000001",
    "深证成指": "sz399001", 
    "创业板指": "sz399006",
    "科创50": "sh000688",
    "沪深300": "sh000300",
    "上证50": "sh000016",
    "中证500": "sh000905",
    "中证1000": "sh000852",
}


class FinanceTrigger:
    """金融数据触发器 - 自动检测问题并补充实时数据"""
    
    def __init__(self):
        self.finance_data = None
        self._enabled = True
        
    def _get_finance_data(self):
        """延迟加载金融数据模块"""
        if self.finance_data is None:
            try:
                from src.finance_data import get_finance_data
                self.finance_data = get_finance_data()
            except Exception as e:
                print(f"警告: 金融数据模块加载失败: {e}")
                self._enabled = False
        return self.finance_data
    
    def is_finance_related(self, question: str) -> bool:
        """判断问题是否涉及金融领域"""
        if not self._enabled:
            return False
            
        question_lower = question.lower()
        
        for category, keywords in FINANCE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in question_lower:
                    return True
        return False
    
    def extract_stock_code(self, question: str) -> Optional[str]:
        """从问题中提取股票代码"""
        match = STOCK_CODE_PATTERN.search(question)
        if match:
            prefix = match.group(1) or ""
            code = match.group(2)
            if prefix:
                return f"{prefix}{code}"
            if code.startswith(("600", "601", "603", "605", "688", "000", "001")):
                if code.startswith(("600", "601", "603", "605", "688")):
                    return f"sh{code}"
                else:
                    return f"sz{code}"
            return code
        
        for name, code in INDEX_CODE_MAP.items():
            if name in question:
                return code
        return None
    
    def extract_fund_code(self, question: str) -> Optional[str]:
        """从问题中提取基金代码"""
        match = FUND_CODE_PATTERN.search(question)
        if match:
            return match.group(1)
        return None
    
    def get_finance_context(self, question: str) -> Tuple[str, List[Dict]]:
        """获取金融上下文数据"""
        if not self.is_finance_related(question):
            return "", []
        
        fd = self._get_finance_data()
        if not fd:
            return "", []
        
        context_parts = []
        sources = []
        
        # 1. 尝试提取股票代码
        stock_code = self.extract_stock_code(question)
        if stock_code:
            try:
                stock_data = fd.get_stock_daily(stock_code)
                if stock_data:
                    recent = stock_data[-5:]
                    data_str = "\n".join([
                        f"{d.get('date', d.get('日期', 'N/A'))}: "
                        f"开盘:{d.get('open', d.get('开盘', 'N/A'))}, "
                        f"收盘:{d.get('close', d.get('收盘', 'N/A'))}, "
                        f"最高:{d.get('high', d.get('最高', 'N/A'))}, "
                        f"最低:{d.get('low', d.get('最低', 'N/A'))}, "
                        f"成交量:{d.get('volume', d.get('成交量', 'N/A'))}"
                        for d in recent
                    ])
                    context_parts.append(f"【股票 {stock_code} 最近交易日行情】\n{data_str}")
                    sources.append({"type": "stock", "code": stock_code})
            except Exception as e:
                print(f"获取股票数据失败: {e}")
        
        # 2. 尝试提取基金代码
        fund_code = self.extract_fund_code(question)
        if fund_code:
            try:
                fund_info = fd.get_fund_info(fund_code)
                if fund_info:
                    recent = fund_info[-5:]
                    data_str = "\n".join([
                        f"{d.get('date', d.get('净值日期', 'N/A'))}: "
                        f"单位净值:{d.get('nav', d.get('单位净值', 'N/A'))}, "
                        f"累计净值:{d.get('acc_nav', d.get('累计净值', 'N/A'))}"
                        for d in recent
                    ])
                    context_parts.append(f"【基金 {fund_code} 净值走势】\n{data_str}")
                    sources.append({"type": "fund", "code": fund_code})
            except Exception as e:
                print(f"获取基金数据失败: {e}")
        
        # 3. 指数数据
        if any(kw in question for kw in ["指数", "大盘", "上证", "深证"]):
            try:
                if "上证" in question:
                    index_code = "sh000001"
                elif "深证" in question or "创业板" in question:
                    index_code = "sz399001"
                elif "沪深300" in question:
                    index_code = "sh000300"
                elif "科创" in question:
                    index_code = "sh000688"
                else:
                    index_code = "sh000001"
                
                index_data = fd.get_index_daily(index_code)
                if index_data:
                    recent = index_data[-5:]
                    data_str = "\n".join([
                        f"{d.get('date', d.get('日期', 'N/A'))}: "
                        f"收盘:{d.get('close', d.get('收盘', 'N/A'))}"
                        for d in recent
                    ])
                    context_parts.append(f"【指数 {index_code} 行情】\n{data_str}")
                    sources.append({"type": "index", "code": index_code})
            except Exception as e:
                print(f"获取指数数据失败: {e}")
        
        # 4. 行业/板块
        if any(kw in question for kw in ["板块", "行业", "概念"]):
            try:
                industry_data = fd.get_industry_spot()
                if industry_data:
                    sorted_data = sorted(industry_data, 
                                      key=lambda x: float(x.get('涨跌幅', x.get('涨跌额', 0))),
                                      reverse=True)[:10]
                    data_str = "\n".join([
                        f"{d.get('板块名称', d.get('name', 'N/A'))}: "
                        f"涨跌幅:{d.get('涨跌幅', d.get('change', 'N/A'))}%"
                        for d in sorted_data
                    ])
                    context_parts.append(f"【行业板块涨跌幅排名】\n{data_str}")
                    sources.append({"type": "industry", "code": "all"})
            except Exception as e:
                print(f"获取板块数据失败: {e}")
        
        # 5. 宏观数据
        if any(kw in question for kw in ["利率", "LPR", "SHIBOR", "降息", "加息"]):
            try:
                lpr_data = fd.get_macro_lpr()
                if lpr_data:
                    recent = lpr_data[-3:]
                    data_str = "\n".join([
                        f"{d.get('日期', d.get('date', 'N/A'))}: "
                        f"1年期:{d.get('1年期', d.get('lpr1y', 'N/A'))}%"
                        for d in recent
                    ])
                    context_parts.append(f"【LPR贷款利率】\n{data_str}")
                    sources.append({"type": "macro", "code": "lpr"})
            except Exception as e:
                print(f"获取LPR数据失败: {e}")
        
        return "\n\n".join(context_parts), sources


_finance_trigger = None

def get_finance_trigger() -> FinanceTrigger:
    """获取金融触发器实例"""
    global _finance_trigger
    if _finance_trigger is None:
        _finance_trigger = FinanceTrigger()
    return _finance_trigger
