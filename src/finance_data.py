"""
金融数据模块 - 基于 AKShare 和 Tushare 的开源金融数据接口
支持：股票、基金、指数、宏观数据、财务报表等
"""

import akshare as ak
import tushare as ts
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta

from config import TUSHARE_TOKEN, is_tushare_configured


class TushareData:
    """基于 Tushare Pro 的金融数据接口（需要注册获取 Token）"""
    
    def __init__(self, token: str = None):
        """
        初始化 Tushare API
        Args:
            token: Tushare Pro Token，若为 None 则从配置文件读取
        """
        self.token = token or TUSHARE_TOKEN
        if self.token:
            self.pro = ts.pro_api(self.token)
        else:
            self.pro = None
    
    def is_available(self) -> bool:
        """检查 Tushare 是否可用（是否配置了 Token）"""
        return is_tushare_configured()
    
    def get_stock_basic(self, exchange: str = "", list_status: str = "L") -> List[Dict]:
        """
        获取股票基本信息
        Args:
            exchange: 交易所代码 (SSE/SZSE/BSE) 或空字符串表示全部
            list_status: 上市状态 (L/D/P) L=上市 D=退市 P=暂停上市
        """
        if not self.is_available():
            raise RuntimeError("Tushare 未配置 Token，请先在 .env 文件中设置 TUSHARE_TOKEN")
        df = self.pro.stock_basic(exchange=exchange, list_status=list_status)
        return df.to_dict('records')
    
    def get_daily_data(self, ts_code: str, start_date: str = None, 
                       end_date: str = None) -> List[Dict]:
        """
        获取日线数据
        Args:
            ts_code: 股票代码，如 "000001.SZ"
            start_date: 开始日期，格式 "YYYYMMDD"
            end_date: 结束日期，格式 "YYYYMMDD"
        """
        if not self.is_available():
            raise RuntimeError("Tushare 未配置 Token，请先在 .env 文件中设置 TUSHARE_TOKEN")
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        
        df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        return df.to_dict('records')
    
    def get_fina_indicator(self, ts_code: str) -> List[Dict]:
        """
        获取财务指标数据
        Args:
            ts_code: 股票代码，如 "000001.SZ"
        """
        if not self.is_available():
            raise RuntimeError("Tushare 未配置 Token，请先在 .env 文件中设置 TUSHARE_TOKEN")
        df = self.pro.fina_indicator(ts_code=ts_code)
        return df.to_dict('records')
    
    def get_financial_report(self, ts_code: str, 
                             report_type: str = "annual") -> List[Dict]:
        """
        获取财务报表
        Args:
            ts_code: 股票代码，如 "000001.SZ"
            report_type: "annual" | "quarter" | "all"
        """
        if not self.is_available():
            raise RuntimeError("Tushare 未配置 Token，请先在 .env 文件中设置 TUSHARE_TOKEN")
        
        if report_type == "annual":
            df = self.pro.fina_mainbz(ts_code=ts_code)
        else:
            df = self.pro.fina_audit(ts_code=ts_code)
        return df.to_dict('records')
    
    def get_balance_sheet(self, ts_code: str) -> List[Dict]:
        """获取资产负债表"""
        if not self.is_available():
            raise RuntimeError("Tushare 未配置 Token，请先在 .env 文件中设置 TUSHARE_TOKEN")
        df = self.pro.balancesheet(ts_code=ts_code)
        return df.to_dict('records')
    
    def get_income_statement(self, ts_code: str) -> List[Dict]:
        """获取利润表"""
        if not self.is_available():
            raise RuntimeError("Tushare 未配置 Token，请先在 .env 文件中设置 TUSHARE_TOKEN")
        df = self.pro.income(ts_code=ts_code)
        return df.to_dict('records')
    
    def get_cash_flow(self, ts_code: str) -> List[Dict]:
        """获取现金流量表"""
        if not self.is_available():
            raise RuntimeError("Tushare 未配置 Token，请先在 .env 文件中设置 TUSHARE_TOKEN")
        df = self.pro.cashflow(ts_code=ts_code)
        return df.to_dict('records')
    
    def get_stock_pledge_stat(self, ts_code: str) -> List[Dict]:
        """获取股权质押统计数据"""
        if not self.is_available():
            raise RuntimeError("Tushare 未配置 Token，请先在 .env 文件中设置 TUSHARE_TOKEN")
        df = self.pro.stock_pledged_stat(ts_code=ts_code)
        return df.to_dict('records')
    
    def get_top_list(self, trade_date: str = None) -> List[Dict]:
        """
        获取龙虎榜数据
        Args:
            trade_date: 交易日期，格式 "YYYYMMDD"
        """
        if not self.is_available():
            raise RuntimeError("Tushare 未配置 Token，请先在 .env 文件中设置 TUSHARE_TOKEN")
        df = self.pro.top_list(ts_code=trade_date)
        return df.to_dict('records')
    
    def get_stk_holders(self, ts_code: str) -> List[Dict]:
        """获取股东人数数据"""
        if not self.is_available():
            raise RuntimeError("Tushare 未配置 Token，请先在 .env 文件中设置 TUSHARE_TOKEN")
        df = self.pro.stk_holders(ts_code=ts_code)
        return df.to_dict('records')


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
_tushare_data = None

def get_finance_data() -> FinanceData:
    """获取 AKShare 金融数据实例（延迟加载）"""
    global _finance_data
    if _finance_data is None:
        _finance_data = FinanceData()
    return _finance_data

def get_tushare_data() -> TushareData:
    """获取 Tushare 金融数据实例（延迟加载）"""
    global _tushare_data
    if _tushare_data is None:
        _tushare_data = TushareData()
    return _tushare_data
