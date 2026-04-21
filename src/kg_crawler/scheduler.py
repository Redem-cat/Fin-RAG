"""
定时调度器模块

功能：
- 定时执行数据采集任务
- 增量更新知识图谱
- 任务管理和监控
"""

import os
import time
import json
import schedule
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
import logging
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KGScheduler:
    """知识图谱数据调度器"""
    
    def __init__(self, log_dir: str = None):
        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent / "data" / "scheduler_logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.tasks: Dict[str, Dict] = {}
        self.running = False
        self._thread = None
        
        # 任务日志
        self.task_log_file = self.log_dir / "task_history.json"
        self.task_history: List[Dict] = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """加载任务历史"""
        if self.task_log_file.exists():
            try:
                with open(self.task_log_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []
    
    def _save_history(self):
        """保存任务历史"""
        try:
            # 只保留最近100条
            if len(self.task_history) > 100:
                self.task_history = self.task_history[-100:]
            
            with open(self.task_log_file, "w", encoding="utf-8") as f:
                json.dump(self.task_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存任务历史失败: {e}")
    
    def register_task(self, name: str, task_func: Callable, schedule_type: str, **kwargs):
        """
        注册定时任务
        
        Args:
            name: 任务名称
            task_func: 任务函数
            schedule_type: 调度类型 (daily, hourly, interval_minutes)
            **kwargs: 调度参数
        """
        task_info = {
            "name": name,
            "func": task_func,
            "schedule_type": schedule_type,
            "kwargs": kwargs,
            "last_run": None,
            "next_run": None,
            "run_count": 0,
            "status": "registered"
        }
        
        # 设置调度
        if schedule_type == "daily":
            hour = kwargs.get("hour", 9)
            minute = kwargs.get("minute", 0)
            schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(
                self._wrap_task(name, task_func)
            )
            task_info["next_run"] = f"每天 {hour:02d}:{minute:02d}"
            
        elif schedule_type == "hourly":
            schedule.every().hour.do(self._wrap_task(name, task_func))
            task_info["next_run"] = "每小时"
            
        elif schedule_type == "interval_minutes":
            minutes = kwargs.get("minutes", 30)
            schedule.every(minutes).minutes.do(self._wrap_task(name, task_func))
            task_info["next_run"] = f"每 {minutes} 分钟"
        
        self.tasks[name] = task_info
        logger.info(f"[调度器] 注册任务: {name} ({task_info['next_run']})")
    
    def _wrap_task(self, name: str, task_func: Callable) -> Callable:
        """包装任务函数，添加日志和错误处理"""
        def wrapped():
            start_time = time.time()
            task_info = self.tasks.get(name, {})
            
            try:
                logger.info(f"[调度器] 开始执行任务: {name}")
                
                # 执行任务
                result = task_func()
                
                # 记录成功
                elapsed = time.time() - start_time
                task_info["last_run"] = datetime.now().isoformat()
                task_info["run_count"] += 1
                task_info["status"] = "success"
                
                # 添加到历史
                self.task_history.append({
                    "task": name,
                    "start_time": datetime.fromtimestamp(start_time).isoformat(),
                    "elapsed_seconds": round(elapsed, 2),
                    "status": "success",
                    "result": str(result)[:500] if result else None
                })
                
                logger.info(f"[调度器] 任务完成: {name} (耗时: {elapsed:.2f}s)")
                
            except Exception as e:
                elapsed = time.time() - start_time
                task_info["status"] = f"failed: {str(e)}"
                
                self.task_history.append({
                    "task": name,
                    "start_time": datetime.fromtimestamp(start_time).isoformat(),
                    "elapsed_seconds": round(elapsed, 2),
                    "status": "failed",
                    "error": str(e)
                })
                
                logger.error(f"[调度器] 任务失败: {name} - {e}")
            
            finally:
                self._save_history()
        
        return wrapped
    
    def start(self, blocking: bool = False):
        """
        启动调度器
        
        Args:
            blocking: 是否阻塞运行
        """
        if self.running:
            logger.warning("[调度器] 已在运行中")
            return
        
        self.running = True
        logger.info(f"[调度器] 启动，共 {len(self.tasks)} 个任务")
        
        if blocking:
            self._run_loop()
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
    
    def _run_loop(self):
        """运行循环"""
        while self.running:
            schedule.run_pending()
            time.sleep(1)
    
    def stop(self):
        """停止调度器"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[调度器] 已停止")
    
    def run_task_now(self, name: str):
        """立即执行指定任务"""
        if name not in self.tasks:
            logger.error(f"[调度器] 任务不存在: {name}")
            return
        
        task_info = self.tasks[name]
        wrapped = self._wrap_task(name, task_info["func"])
        wrapped()
    
    def get_status(self) -> Dict:
        """获取调度器状态"""
        return {
            "running": self.running,
            "tasks_count": len(self.tasks),
            "tasks": {
                name: {
                    "schedule_type": info["schedule_type"],
                    "next_run": info["next_run"],
                    "last_run": info["last_run"],
                    "run_count": info["run_count"],
                    "status": info["status"]
                }
                for name, info in self.tasks.items()
            },
            "recent_history": self.task_history[-10:]
        }


# =========================
# 🔹 预定义任务
# =========================
def task_crawl_news():
    """任务：爬取财经新闻"""
    from src.kg_crawler.news_crawler import get_news_crawler
    from src.kg_builder.entity_extractor import extract_entities_from_news
    from src.kg_builder.kg_writer import get_kg_writer
    
    crawler = get_news_crawler()
    writer = get_kg_writer()
    
    # 爬取新闻
    news_list = crawler.crawl_all(max_count_per_source=20)
    
    if not news_list:
        return {"message": "没有新新闻"}
    
    # 抽取实体并写入图谱
    total_entities = 0
    for news in news_list:
        entities, relations = extract_entities_from_news({
            "title": news.title,
            "content": news.content
        })
        
        # 写入实体
        for entity in entities:
            if entity.entity_type == "Company":
                writer.write_company({"name": entity.name, **entity.properties})
            elif entity.entity_type == "Sector":
                writer.write_sector({"name": entity.name, **entity.properties})
        
        total_entities += len(entities)
    
    return {
        "news_count": len(news_list),
        "entities_extracted": total_entities
    }


def task_update_stocks():
    """任务：更新股票基础信息"""
    from src.kg_crawler.api_sources import get_financial_data_source
    from src.kg_builder.kg_writer import get_kg_writer
    
    source = get_financial_data_source()
    writer = get_kg_writer()
    
    # 获取股票列表
    stocks = source.get_all_companies(use_cache=False)
    
    if not stocks:
        return {"message": "获取股票列表失败"}
    
    # 批量写入
    stats = writer.import_companies_batch(stocks[:100])  # 限制数量
    
    return {
        "companies_imported": stats.total_companies,
        "errors": len(stats.errors)
    }


def task_update_sectors():
    """任务：更新行业分类"""
    from src.kg_crawler.api_sources import get_financial_data_source
    from src.kg_builder.kg_writer import get_kg_writer
    
    source = get_financial_data_source()
    writer = get_kg_writer()
    
    # 获取行业分类
    sectors = source.get_sectors()
    
    if not sectors:
        return {"message": "获取行业分类失败"}
    
    # 批量写入
    sector_data = [s.__dict__ for s in sectors]
    stats = writer.import_sectors_batch(sector_data)
    
    return {
        "sectors_imported": stats.total_sectors
    }


# =========================
# 🔹 全局实例
# =========================
_scheduler: Optional[KGScheduler] = None


def get_kg_scheduler() -> KGScheduler:
    """获取调度器实例（单例）"""
    global _scheduler
    if _scheduler is None:
        _scheduler = KGScheduler()
    return _scheduler


def setup_default_tasks():
    """设置默认任务"""
    scheduler = get_kg_scheduler()
    
    # 每小时爬取新闻
    scheduler.register_task(
        "crawl_news",
        task_crawl_news,
        "hourly"
    )
    
    # 每天更新股票信息
    scheduler.register_task(
        "update_stocks",
        task_update_stocks,
        "daily",
        hour=9,
        minute=30
    )
    
    # 每周更新行业分类
    scheduler.register_task(
        "update_sectors",
        task_update_sectors,
        schedule_type="daily",
        hour=9,
        minute=0
    )
    
    return scheduler


# =========================
# 🔹 测试代码
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("调度器测试")
    print("=" * 60)
    
    scheduler = setup_default_tasks()
    
    # 查看状态
    status = scheduler.get_status()
    print(json.dumps(status, ensure_ascii=False, indent=2))
    
    # 立即执行一个任务测试
    print("\n[测试] 立即执行新闻爬取...")
    scheduler.run_task_now("crawl_news")
