"""
财经新闻爬虫模块

功能：
- 爬取新浪财经、东方财富等网站新闻
- 支持增量更新
- 自动提取实体和关系
"""

import os
import sys
import re
import time
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

# 修复 Windows 控制台编码
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

_log_handler = logging.StreamHandler(sys.stdout)
_log_handler.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
logging.basicConfig(level=logging.INFO, handlers=[_log_handler], force=True)
logger = logging.getLogger(__name__)


@dataclass
class CrawledNews:
    """爬取的新闻数据"""
    title: str
    content: str
    source: str
    url: str
    publish_time: str
    crawl_time: str = field(default_factory=lambda: datetime.now().isoformat())
    content_hash: str = ""
    entities: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.md5(
                (self.title + self.content).encode()
            ).hexdigest()[:16]


class FinancialNewsCrawler:
    """财经新闻爬虫"""
    
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "data" / "news_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.history_file = self.cache_dir / "crawl_history.json"
        self.history = self._load_history()
        
        # 请求配置
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.timeout = 10
        self.retry_count = 3
    
    def _load_history(self) -> Dict:
        """加载爬取历史"""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载爬取历史失败: {e}")
        return {"urls": {}, "last_crawl": {}}
    
    def _save_history(self):
        """保存爬取历史"""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存爬取历史失败: {e}")
    
    def crawl_sina_finance(self, max_count: int = 50) -> List[CrawledNews]:
        """
        爬取新浪财经新闻
        
        Args:
            max_count: 最大爬取数量
            
        Returns:
            新闻列表
        """
        news_list = []
        
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # 新浪财经滚动新闻 API
            url = "https://feed.mix.sina.com.cn/api/roll/get"
            params = {
                "pageid": "153",
                "lid": "2509",
                "k": "",
                "num": max_count,
                "page": 1,
                "r": str(time.time())
            }
            
            response = requests.get(
                url, params=params, headers=self.headers, timeout=self.timeout
            )
            data = response.json()
            
            if data.get("result") and data["result"].get("data"):
                for item in data["result"]["data"]:
                    news = CrawledNews(
                        title=item.get("title", ""),
                        content=item.get("intro", ""),
                        source="新浪财经",
                        url=item.get("url", ""),
                        publish_time=item.get("ctime", "")
                    )
                    
                    # 检查是否已爬取
                    if news.content_hash not in self.history["urls"]:
                        news_list.append(news)
                        self.history["urls"][news.content_hash] = {
                            "title": news.title,
                            "crawl_time": news.crawl_time
                        }
            
            logger.info(f"[新浪财经] 爬取到 {len(news_list)} 条新新闻")
            
        except ImportError:
            logger.warning("[爬虫] 需要安装 requests 和 beautifulsoup4")
        except Exception as e:
            logger.error(f"[新浪财经] 爬取失败: {e}")
        
        self._save_history()
        return news_list
    
    def crawl_eastmoney_news(self, max_count: int = 50) -> List[CrawledNews]:
        """
        爬取东方财富新闻
        
        Args:
            max_count: 最大爬取数量
            
        Returns:
            新闻列表
        """
        news_list = []
        
        try:
            import requests
            
            # 东方财富财经新闻 API
            url = "https://np-listapi.eastmoney.com/comm/web/getFastNewsList"
            params = {
                "client": "web",
                "biz": "web_724",
                "fastColumn": "102",
                "sortEnd": "",
                "pageSize": max_count,
                "req_trace": str(int(time.time() * 1000))
            }
            
            response = requests.get(
                url, params=params, headers=self.headers, timeout=self.timeout
            )
            data = response.json()
            
            if data.get("data") and data["data"].get("fastNewsList"):
                for item in data["data"]["fastNewsList"]:
                    news = CrawledNews(
                        title=item.get("title", ""),
                        content=item.get("digest", ""),
                        source="东方财富",
                        url=f"https://finance.eastmoney.com/a/{item.get('code', '')}.html",
                        publish_time=item.get("showTime", "")
                    )
                    
                    if news.content_hash not in self.history["urls"]:
                        news_list.append(news)
                        self.history["urls"][news.content_hash] = {
                            "title": news.title,
                            "crawl_time": news.crawl_time
                        }
            
            logger.info(f"[东方财富] 爬取到 {len(news_list)} 条新新闻")
            
        except ImportError:
            logger.warning("[爬虫] 需要安装 requests")
        except Exception as e:
            logger.error(f"[东方财富] 爬取失败: {e}")
        
        self._save_history()
        return news_list
    
    def _is_relevant(self, news: CrawledNews) -> bool:
        """判断新闻是否与芯片/半导体/智能驾驶/电子供应链相关"""
        keywords = [
            "芯片", "半导体", "集成电路", "晶圆", "光刻", "刻蚀",
            "GPU", "CPU", "AI芯片", "存储芯片", "NAND", "DRAM",
            "代工", "台积电", "中芯", "三星", "联电",
            "光刻机", "ASML", "刻蚀机",
            "硅片", "光刻胶", "特种气体", "靶材",
            "5nm", "7nm", "3nm", "14nm", "28nm",
            "封装", "测试", "EDA",
            "智驾", "自动驾驶", "智能驾驶", "激光雷达",
            "车载芯片", "车规级",
            "通信", "5G", "6G", "基站", "射频",
            "机器人", "人形机器人", "伺服电机", "减速器",
            "停产", "断供", "制裁", "出口管制", "供应链",
        ]
        text = f"{news.title} {news.content}".lower()
        for kw in keywords:
            if kw.lower() in text:
                return True
        return False

    def crawl_all(self, max_count_per_source: int = 30, filter_relevant: bool = True) -> List[CrawledNews]:
        """
        爬取所有数据源，可选过滤芯片/半导体相关新闻
        
        Args:
            max_count_per_source: 每个数据源的最大爬取数量
            filter_relevant: 是否只保留相关领域新闻
            
        Returns:
            合并后的新闻列表
        """
        all_news = []
        
        # 爬取新浪财经
        sina_news = self.crawl_sina_finance(max_count_per_source)
        all_news.extend(sina_news)
        
        # 爬取东方财富
        eastmoney_news = self.crawl_eastmoney_news(max_count_per_source)
        all_news.extend(eastmoney_news)
        
        # 按时间排序
        all_news.sort(key=lambda x: x.publish_time, reverse=True)
        
        # 过滤相关新闻
        if filter_relevant:
            filtered = [n for n in all_news if self._is_relevant(n)]
            logger.info(f"[过滤] 从 {len(all_news)} 条中筛选出 {len(filtered)} 条芯片/半导体相关新闻")
            all_news = filtered
        
        logger.info(f"[总计] 爬取到 {len(all_news)} 条新新闻")
        
        # 更新最后爬取时间
        self.history["last_crawl"]["all"] = datetime.now().isoformat()
        self._save_history()
        
        return all_news
    
    def save_news_to_json(self, news_list: List[CrawledNews], output_file: str = None):
        """
        保存新闻到 JSON 文件
        
        Args:
            news_list: 新闻列表
            output_file: 输出文件路径
        """
        if output_file is None:
            output_file = self.cache_dir / f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = [news.__dict__ for news in news_list]
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[保存] 新闻已保存到 {output_file}")
    
    def get_recent_news(self, hours: int = 24) -> List[CrawledNews]:
        """
        获取最近的新闻（从缓存）
        
        Args:
            hours: 时间范围（小时）
            
        Returns:
            新闻列表
        """
        news_files = sorted(
            self.cache_dir.glob("news_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        all_news = []
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        for file_path in news_files[:10]:  # 最多检查最近10个文件
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    news_data = json.load(f)
                
                for item in news_data:
                    crawl_time = datetime.fromisoformat(item.get("crawl_time", ""))
                    if crawl_time > cutoff_time:
                        all_news.append(CrawledNews(**item))
            except Exception as e:
                logger.warning(f"读取新闻文件失败 {file_path}: {e}")
        
        return all_news
    
    def get_statistics(self) -> Dict:
        """获取爬取统计信息"""
        return {
            "total_urls": len(self.history.get("urls", {})),
            "last_crawl": self.history.get("last_crawl", {}),
            "cache_dir": str(self.cache_dir)
        }


# =========================
# 🔹 全局实例
# =========================
_crawler: Optional[FinancialNewsCrawler] = None


def get_news_crawler(cache_dir: str = None) -> FinancialNewsCrawler:
    """获取新闻爬虫实例（单例）"""
    global _crawler
    if _crawler is None:
        _crawler = FinancialNewsCrawler(cache_dir)
    return _crawler


# =========================
# 🔹 测试代码
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("新闻爬虫测试")
    print("=" * 60)
    
    crawler = get_news_crawler()
    
    # 爬取新闻
    print("\n[测试] 爬取新闻...")
    news = crawler.crawl_all(max_count_per_source=10)
    
    if news:
        print(f"爬取到 {len(news)} 条新闻")
        print(f"示例: {news[0].title}")
        
        # 保存
        crawler.save_news_to_json(news)
    
    # 统计
    stats = crawler.get_statistics()
    print(f"\n统计: {stats}")
