"""
知识图谱数据采集模块

包含：
- API 数据源封装（AKShare, Tushare）
- 新闻爬虫
- 定时调度器
"""

from .api_sources import FinancialDataSource, get_financial_data_source
from .news_crawler import FinancialNewsCrawler, get_news_crawler
from .scheduler import KGScheduler, get_kg_scheduler

__all__ = [
    "FinancialDataSource",
    "get_financial_data_source",
    "FinancialNewsCrawler",
    "get_news_crawler",
    "KGScheduler",
    "get_kg_scheduler",
]
