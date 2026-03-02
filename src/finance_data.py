"""
金融数据模块 - 基于 AKShare 的开源金融数据接口
支持：股票、基金、指数、宏观数据等
"""

import akshare as ak
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta


class FinanceData:
    """金融数据获取类"""
    
    @staticmethod
    def get_fund_etf_spot() -> List[Dict]:
        """获取ETF实时行情"""
        df = ak.fund_etf_spot_em()
        return df.to_dict('records')
    
    @staticmethod
    def get_fund_etf_hist(symbol: str, period: str = "daily", 
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> List[Dict]:
        """
        获取ETF历史数据
        Args:
            symbol: ETF代码，如 "510300"（沪深300ETF）
            period: "daily" | "weekly" | "monthly"
            start_date: 开始日期，格式 "YYYYMMDD"
            end_date: 结束日期，格式 "YYYYMMDD"
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
            
        df = ak.fund_etf_hist_em(symbol=symbol, period=period, 
                                  start_date=start_date, end_date=end_date)
        return df.to_dict('records')
    
    @staticmethod
    def get_fund_info(fund_code: str, indicator: str = "单位净值走势") -> List[Dict]:
        """
        获取基金净值信息
        Args:
            fund_code: 基金代码，如 "710001"
            indicator: "单位净值走势" | "累计净值走势" | "单位净值"
        """
        df = ak.fund_open_fund_info_em(fund=fund_code, indicator=indicator)
        return df.to_dict('records')
    
    @staticmethod
    def get_stock_daily(symbol: str, adjust: str = "qfq") -> List[Dict]:
        """
        获取A股日线数据
        Args:
            symbol: 股票代码，如 "sh600519"（贵州茅台）
            adjust: "qfq"（前复权）| "hfq"（后复权）| ""
        """
        df = ak.stock_zh_a_daily(symbol=symbol, adjust=adjust)
        return df.to_dict('records')
    
    @staticmethod
    def get_stock_spot() -> List[Dict]:
        """获取A股实时行情"""
        df = ak.stock_zh_a_spot_em()
        return df.to_dict('records')
    
    @staticmethod
    def get_stock_info_a_code_name() -> Dict[str, str]:
        """获取A股代码名称映射"""
        df = ak.stock_info_a_code_name()
        return dict(zip(df['code'], df['name']))
    
    @staticmethod
    def get_index_daily(symbol: str = "sh000001") -> List[Dict]:
        """
        获取指数日线数据
        Args:
            symbol: 指数代码，如 "sh000001"（上证指数）、"sz399001"（深证成指）
        """
        df = ak.stock_zh_index_daily(symbol=symbol)
        return df.to_dict('records')
    
    @staticmethod
    def get_index_spot() -> List[Dict]:
        """获取指数实时行情"""
        df = ak.stock_zh_index_spot_em()
        return df.to_dict('records')
    
    @staticmethod
    def get_industry_spot() -> List[Dict]:
        """获取行业板块实时行情"""
        df = ak.stock_board_industry_name_em()
        return df.to_dict('records')
    
    @staticmethod
    def get_concept_spot() -> List[Dict]:
        """获取概念板块实时行情"""
        df = ak.stock_board_concept_name_em()
        return df.to_dict('records')
    
    @staticmethod
    def get_macro_lpr() -> List[Dict]:
        """获取LPR贷款利率数据"""
        df = ak.macro_lpr()
        return df.to_dict('records')
    
    @staticmethod
    def get_macro_shibor() -> List[Dict]:
        """获取SHIBOR利率数据"""
        df = ak.macro_shibor()
        return df.to_dict('records')
    
    @staticmethod
    def get_fund_portfolio(symbol: str) -> List[Dict]:
        """
        获取基金持仓数据
        Args:
            symbol: 基金代码，如 "161039"
        """
        df = ak.fund_portfolio_hold_em(symbol=symbol)
        return df.to_dict('records')
    
    @staticmethod
    def search_stock(keyword: str) -> List[Dict]:
        """搜索股票"""
        df = ak.stock_info_a_code_name()
        # 模糊匹配
        result = df[df['name'].str.contains(keyword, na=False) | 
                    df['code'].str.contains(keyword, na=False)]
        return result.to_dict('records')
    
    @staticmethod
    def get_fund_manager(fund: str) -> List[Dict]:
        """获取基金经理信息"""
        df = ak.fund_manager_em(fund=fund)
        return df.to_dict('records')


# 全局实例
_finance_data = None

def get_finance_data() -> FinanceData:
    """获取金融数据实例（延迟加载）"""
    global _finance_data
    if _finance_data is None:
        _finance_data = FinanceData()
    return _finance_data
