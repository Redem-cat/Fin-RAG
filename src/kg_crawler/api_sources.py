"""
金融数据 API 封装模块

整合 AKShare 和 Tushare 数据源，为知识图谱提供结构化数据

数据来源优先级：
1. AKShare（免费、开源、无需注册）
2. Tushare（需注册积分，数据更全面）

功能：
- 上市公司基本信息
- 行业分类数据
- 高管信息
- 实时行情
- 财务指标
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# 🔹 数据结构定义
# =========================
@dataclass
class CompanyInfo:
    """公司信息"""
    code: str  # 股票代码
    name: str  # 公司名称
    sector: str = ""  # 所属行业
    market: str = ""  # 上市板块
    market_cap: float = 0.0  # 市值（亿元）
    list_date: str = ""  # 上市日期
    description: str = ""  # 公司简介
    
    def to_neo4j_node(self) -> Dict:
        """转换为 Neo4j 节点格式"""
        return {
            "name": self.name,
            "code": self.code,
            "sector": self.sector,
            "market": self.market,
            "market_cap": self.market_cap,
            "list_date": self.list_date,
            "description": self.description
        }


@dataclass
class PersonInfo:
    """人物信息"""
    name: str  # 姓名
    title: str = ""  # 职位
    company: str = ""  # 所属公司
    age: int = 0  # 年龄
    education: str = ""  # 学历
    
    def to_neo4j_node(self) -> Dict:
        return {
            "name": self.name,
            "title": self.title,
            "company": self.company,
            "age": self.age,
            "education": self.education
        }


@dataclass
class SectorInfo:
    """行业信息"""
    name: str  # 行业名称
    code: str = ""  # 行业代码
    parent: str = ""  # 上级行业
    description: str = ""  # 行业描述
    trend: str = ""  # 行业趋势
    
    def to_neo4j_node(self) -> Dict:
        return {
            "name": self.name,
            "code": self.code,
            "parent": self.parent,
            "description": self.description,
            "trend": self.trend
        }


@dataclass
class NewsArticle:
    """新闻文章"""
    title: str  # 标题
    content: str  # 内容
    source: str  # 来源
    publish_time: str  # 发布时间
    url: str = ""  # 原文链接
    tags: List[str] = field(default_factory=list)  # 标签
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FinancialIndicator:
    """财务指标"""
    code: str  # 股票代码
    name: str  # 公司名称
    roe: float = 0.0  # 净资产收益率
    pe: float = 0.0  # 市盈率
    pb: float = 0.0  # 市净率
    revenue: float = 0.0  # 营收（亿元）
    profit: float = 0.0  # 净利润（亿元）
    debt_ratio: float = 0.0  # 资产负债率
    date: str = ""  # 报告期


# =========================
# 🔹 AKShare 数据源
# =========================
class AKShareSource:
    """AKShare 数据源封装"""
    
    def __init__(self):
        self.available = False
        self._check_availability()
    
    def _check_availability(self):
        """检查 AKShare 是否可用"""
        try:
            import akshare as ak
            self.ak = ak
            self.available = True
            logger.info("[AKShare] 模块加载成功")
        except ImportError:
            logger.warning("[AKShare] 未安装，请运行: pip install akshare")
            self.ak = None
            self.available = False
    
    def get_all_stocks(self) -> List[Dict]:
        """获取所有 A 股上市公司列表"""
        if not self.available:
            return []
        
        try:
            # 获取 A 股列表
            df = self.ak.stock_zh_a_spot_em()
            
            stocks = []
            for _, row in df.iterrows():
                stocks.append({
                    "code": row.get("代码", ""),
                    "name": row.get("名称", ""),
                    "market": self._get_market(row.get("代码", "")),
                    "price": row.get("最新价", 0),
                    "change_pct": row.get("涨跌幅", 0),
                    "volume": row.get("成交量", 0),
                    "amount": row.get("成交额", 0),
                })
            
            logger.info(f"[AKShare] 获取到 {len(stocks)} 只 A 股")
            return stocks
            
        except Exception as e:
            logger.error(f"[AKShare] 获取股票列表失败: {e}")
            return []
    
    def get_company_info(self, code: str) -> Optional[CompanyInfo]:
        """获取公司详细信息"""
        if not self.available:
            return None
        
        try:
            # 获取公司基本信息
            df = self.ak.stock_individual_info_em(symbol=code)
            
            info_dict = {}
            for _, row in df.iterrows():
                info_dict[row["item"]] = row["value"]
            
            return CompanyInfo(
                code=code,
                name=info_dict.get("股票简称", ""),
                sector=info_dict.get("行业", ""),
                market=info_dict.get("上市板块", ""),
                list_date=info_dict.get("上市时间", ""),
                description=info_dict.get("公司简介", ""),
            )
            
        except Exception as e:
            logger.error(f"[AKShare] 获取公司 {code} 信息失败: {e}")
            return None
    
    def get_industry_classification(self) -> List[SectorInfo]:
        """获取申万行业分类"""
        if not self.available:
            return []
        
        try:
            # 获取申万行业分类
            df = self.ak.stock_board_industry_name_em()
            
            sectors = []
            for _, row in df.iterrows():
                sectors.append(SectorInfo(
                    name=row.get("板块名称", ""),
                    code=row.get("板块代码", ""),
                    description=f"申万行业分类"
                ))
            
            logger.info(f"[AKShare] 获取到 {len(sectors)} 个行业分类")
            return sectors
            
        except Exception as e:
            logger.error(f"[AKShare] 获取行业分类失败: {e}")
            return []
    
    def get_company_by_sector(self, sector_name: str) -> List[str]:
        """获取某行业的所有公司"""
        if not self.available:
            return []
        
        try:
            df = self.ak.stock_board_industry_cons_em(symbol=sector_name)
            
            companies = []
            for _, row in df.iterrows():
                companies.append(row.get("个股名称", ""))
            
            return companies
            
        except Exception as e:
            logger.error(f"[AKShare] 获取行业 {sector_name} 公司失败: {e}")
            return []
    
    def get_realtime_data(self, code: str) -> Dict:
        """获取实时行情数据"""
        if not self.available:
            return {}
        
        try:
            df = self.ak.stock_zh_a_spot_em()
            stock = df[df["代码"] == code]
            
            if not stock.empty:
                row = stock.iloc[0]
                return {
                    "code": code,
                    "name": row.get("名称", ""),
                    "price": row.get("最新价", 0),
                    "change": row.get("涨跌额", 0),
                    "change_pct": row.get("涨跌幅", 0),
                    "volume": row.get("成交量", 0),
                    "amount": row.get("成交额", 0),
                    "high": row.get("最高", 0),
                    "low": row.get("最低", 0),
                    "open": row.get("今开", 0),
                    "prev_close": row.get("昨收", 0),
                }
            return {}
            
        except Exception as e:
            logger.error(f"[AKShare] 获取 {code} 实时数据失败: {e}")
            return {}
    
    def get_financial_news(self, page: int = 1) -> List[NewsArticle]:
        """获取财经新闻"""
        if not self.available:
            return []
        
        try:
            df = self.ak.stock_news_em(symbol="财经新闻")
            
            news_list = []
            for _, row in df.head(20).iterrows():  # 只取前20条
                news_list.append(NewsArticle(
                    title=row.get("新闻标题", ""),
                    content=row.get("新闻内容", ""),
                    source=row.get("新闻来源", ""),
                    publish_time=row.get("发布时间", ""),
                ))
            
            return news_list
            
        except Exception as e:
            logger.error(f"[AKShare] 获取财经新闻失败: {e}")
            return []
    
    def _get_market(self, code: str) -> str:
        """根据代码判断市场"""
        if code.startswith("6"):
            return "上海证券交易所"
        elif code.startswith("0") or code.startswith("3"):
            return "深圳证券交易所"
        elif code.startswith("68"):
            return "科创板"
        elif code.startswith("8") or code.startswith("4"):
            return "北交所"
        return "未知"


# =========================
# 🔹 Tushare 数据源（可选）
# =========================
class TushareSource:
    """Tushare 数据源封装（需要 token）"""
    
    def __init__(self, token: str = None):
        self.token = token or os.getenv("TUSHARE_TOKEN", "")
        self.available = False
        self.pro = None
        self._check_availability()
    
    def _check_availability(self):
        """检查 Tushare 是否可用"""
        if not self.token:
            logger.info("[Tushare] 未配置 token，跳过")
            return
        
        try:
            import tushare as ts
            ts.set_token(self.token)
            self.pro = ts.pro_api()
            self.available = True
            logger.info("[Tushare] 模块加载成功")
        except ImportError:
            logger.warning("[Tushare] 未安装，请运行: pip install tushare")
        except Exception as e:
            logger.warning(f"[Tushare] 初始化失败: {e}")
    
    def get_company_basic(self, code: str) -> Optional[Dict]:
        """获取公司基本信息（Tushare Pro）"""
        if not self.available:
            return None
        
        try:
            # 转换代码格式（如 000001 -> 000001.SZ）
            ts_code = self._convert_code(code)
            
            df = self.pro.daily_basic(ts_code=ts_code, fields=[
                'ts_code', 'name', 'industry', 'market', 'list_date',
                'total_mv', 'circ_mv', 'pe', 'pb', 'turnover_rate'
            ])
            
            if not df.empty:
                row = df.iloc[0]
                return {
                    "code": code,
                    "name": row.get("name", ""),
                    "sector": row.get("industry", ""),
                    "market": row.get("market", ""),
                    "market_cap": row.get("total_mv", 0) / 10000,  # 转为亿元
                    "list_date": row.get("list_date", ""),
                    "pe": row.get("pe", 0),
                    "pb": row.get("pb", 0),
                }
            return None
            
        except Exception as e:
            logger.error(f"[Tushare] 获取公司 {code} 信息失败: {e}")
            return None
    
    def _convert_code(self, code: str) -> str:
        """转换股票代码格式"""
        if "." in code:
            return code
        
        if code.startswith("6"):
            return f"{code}.SH"
        else:
            return f"{code}.SZ"


# =========================
# 🔹 统一数据源接口
# =========================
class FinancialDataSource:
    """金融数据统一接口"""
    
    def __init__(self, tushare_token: str = None):
        self.akshare = AKShareSource()
        self.tushare = TushareSource(tushare_token)
        self._cache = {}
        self._cache_time = {}
    
    def get_all_companies(self, use_cache: bool = True) -> List[Dict]:
        """获取所有上市公司"""
        cache_key = "all_companies"
        
        if use_cache and cache_key in self._cache:
            if self._is_cache_valid(cache_key, hours=24):
                return self._cache[cache_key]
        
        # 优先使用 AKShare
        companies = self.akshare.get_all_stocks()
        
        if companies:
            self._cache[cache_key] = companies
            self._cache_time[cache_key] = datetime.now()
        
        return companies
    
    def get_company_detail(self, code: str) -> Optional[CompanyInfo]:
        """获取公司详细信息"""
        # 先尝试 Tushare（数据更全面）
        if self.tushare.available:
            data = self.tushare.get_company_basic(code)
            if data:
                return CompanyInfo(
                    code=data["code"],
                    name=data["name"],
                    sector=data.get("sector", ""),
                    market=data.get("market", ""),
                    market_cap=data.get("market_cap", 0),
                    list_date=data.get("list_date", ""),
                )
        
        # 降级到 AKShare
        return self.akshare.get_company_info(code)
    
    def get_sectors(self) -> List[SectorInfo]:
        """获取所有行业分类"""
        return self.akshare.get_industry_classification()
    
    def get_sector_companies(self, sector_name: str) -> List[str]:
        """获取行业内的公司"""
        return self.akshare.get_company_by_sector(sector_name)
    
    def get_realtime_quote(self, code: str) -> Dict:
        """获取实时行情"""
        return self.akshare.get_realtime_data(code)
    
    def get_news(self, count: int = 20) -> List[NewsArticle]:
        """获取财经新闻"""
        return self.akshare.get_financial_news()[:count]
    
    def _is_cache_valid(self, cache_key: str, hours: int = 24) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self._cache_time:
            return False
        
        elapsed = datetime.now() - self._cache_time[cache_key]
        return elapsed.total_seconds() < hours * 3600
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self._cache_time.clear()


# =========================
# 🔹 全局实例
# =========================
_data_source: Optional[FinancialDataSource] = None


def get_financial_data_source(tushare_token: str = None) -> FinancialDataSource:
    """获取金融数据源实例（单例）"""
    global _data_source
    if _data_source is None:
        _data_source = FinancialDataSource(tushare_token)
    return _data_source


# =========================
# 🔹 测试代码
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("金融数据 API 测试")
    print("=" * 60)
    
    source = get_financial_data_source()
    
    # 测试获取股票列表
    print("\n[测试] 获取股票列表...")
    stocks = source.get_all_companies()
    if stocks:
        print(f"获取到 {len(stocks)} 只股票")
        print(f"示例: {stocks[0]}")
    
    # 测试获取行业分类
    print("\n[测试] 获取行业分类...")
    sectors = source.get_sectors()
    if sectors:
        print(f"获取到 {len(sectors)} 个行业")
        print(f"示例: {sectors[0].name if sectors else 'N/A'}")
    
    # 测试实时行情
    print("\n[测试] 获取实时行情...")
    quote = source.get_realtime_quote("000001")
    if quote:
        print(f"平安银行: 价格={quote.get('price')}, 涨跌幅={quote.get('change_pct')}%")
