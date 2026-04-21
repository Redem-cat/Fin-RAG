"""
AKQuant 多策略模拟盘模块

支持多 slot 策略配置、组合回测、跨策略风控指标计算。
"""

import os
import traceback
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
import pandas as pd


# =========================
# 配置
# =========================
BASE_PATH = Path(__file__).parent.parent
DEFAULT_INITIAL_CASH = 100_000.0
DEFAULT_COMMISSION = 0.0003


@dataclass
class StrategySlot:
    """单个策略槽位配置"""
    slot_id: str
    strategy_class: Any = None          # Strategy 类
    strategy_params: Dict = field(default_factory=dict)
    max_order_size: int = 10
    weight: float = 1.0                 # 资金权重
    description: str = ""

    def to_dict(self) -> Dict:
        result = asdict(self)
        result["strategy_class"] = str(self.strategy_class) if self.strategy_class else None
        return result


class MultiStrategySimulator:
    """多策略模拟盘"""

    def __init__(self):
        self.slots: Dict[str, StrategySlot] = {}
        self.active_snapshot: Optional[str] = None
        self._results_cache: Dict = {}

    def add_slot(
        self,
        slot_id: str,
        strategy_class: Any = None,
        strategy_params: Dict = None,
        max_order_size: int = 10,
        weight: float = 1.0,
        description: str = "",
    ) -> StrategySlot:
        """
        注册策略槽位

        Args:
            slot_id: 槽位 ID
            strategy_class: Strategy 类
            strategy_params: 策略参数
            max_order_size: 最大下单量
            weight: 资金权重
            description: 描述

        Returns:
            StrategySlot
        """
        slot = StrategySlot(
            slot_id=slot_id,
            strategy_class=strategy_class,
            strategy_params=strategy_params or {},
            max_order_size=max_order_size,
            weight=weight,
            description=description,
        )
        self.slots[slot_id] = slot
        return slot

    def remove_slot(self, slot_id: str) -> bool:
        """移除策略槽位"""
        if slot_id in self.slots:
            del self.slots[slot_id]
            return True
        return False

    def get_slot(self, slot_id: str) -> Optional[StrategySlot]:
        """获取策略槽位"""
        return self.slots.get(slot_id)

    def list_slots(self) -> List[Dict]:
        """列出所有槽位"""
        return [slot.to_dict() for slot in self.slots.values()]

    def run_simulation(
        self,
        data: pd.DataFrame,
        symbols: str = "AAPL",
        initial_cash: float = DEFAULT_INITIAL_CASH,
        commission_rate: float = DEFAULT_COMMISSION,
        stamp_tax_rate: float = 0.001,
        transfer_fee_rate: float = 0.00001,
        start_date: str = None,
        end_date: str = None,
    ) -> Dict[str, Any]:
        """
        使用 AKQuant BacktestConfig + StrategyConfig 运行多策略回测

        Args:
            data: 市场数据 DataFrame
            symbols: 交易标的
            initial_cash: 初始资金
            commission_rate: 佣金费率
            stamp_tax_rate: 印花税率
            transfer_fee_rate: 过户费率
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            回测结果字典，包含各 slot 独立指标和组合指标
        """
        if not self.slots:
            return {
                "success": False,
                "error": "没有注册任何策略槽位",
            }

        try:
            import akquant as aq

            # 准备策略配置
            strategies_by_slot = {}
            strategy_max_order_size = {}
            slot_weights = {}

            for slot_id, slot in self.slots.items():
                if slot.strategy_class is not None:
                    strategies_by_slot[slot_id] = slot.strategy_class
                strategy_max_order_size[slot_id] = slot.max_order_size
                slot_weights[slot_id] = slot.weight

            # 构建 BacktestConfig
            config = aq.BacktestConfig(
                strategy_config=aq.StrategyConfig(
                    strategy_id=list(self.slots.keys())[0] if self.slots else "alpha",
                    strategies_by_slot=strategies_by_slot if strategies_by_slot else None,
                    strategy_max_order_size=strategy_max_order_size,
                )
            )

            # 运行回测
            backtest_kwargs = {
                "data": data,
                "strategy": list(self.slots.values())[0].strategy_class,
                "symbols": symbols,
                "initial_cash": initial_cash,
                "commission_rate": commission_rate,
                "stamp_tax_rate": stamp_tax_rate,
                "transfer_fee_rate": transfer_fee_rate,
                "show_progress": False,
                "config": config,
            }

            if start_date:
                backtest_kwargs["start_time"] = start_date
            if end_date:
                backtest_kwargs["end_time"] = end_date

            result = aq.run_backtest(**backtest_kwargs)

            # 提取指标
            metrics = result.metrics
            metrics_dict = {
                "total_return_pct": metrics.total_return_pct,
                "annualized_return": metrics.annualized_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "win_rate": metrics.win_rate,
                "total_trades": len(result.trades_df) if hasattr(result, 'trades_df') else 0,
                "final_value": metrics.final_value if hasattr(metrics, 'final_value') else initial_cash * (1 + metrics.total_return_pct / 100),
            }

            # 转换为可序列化格式
            for key in metrics_dict:
                if metrics_dict[key] is not None and not isinstance(metrics_dict[key], dict):
                    metrics_dict[key] = float(metrics_dict[key])

            # 缓存主策略指标，用于跨策略计算
            primary_slot_id = list(self.slots.keys())[0] if self.slots else "alpha"
            self._results_cache[primary_slot_id] = metrics_dict

            # 计算跨策略指标
            cross_metrics = self.get_cross_slot_metrics({
                "slot_weights": slot_weights,
                "metrics": metrics_dict,
                "result_object": result,
            })

            return {
                "success": True,
                "metrics": metrics_dict,
                "cross_slot_metrics": cross_metrics,
                "trades_count": metrics_dict["total_trades"],
                "result_object": result,
                "slot_count": len(self.slots),
                "slot_ids": list(self.slots.keys()),
            }

        except ImportError:
            return {
                "success": False,
                "error": "AKQuant 未安装",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def run_individual_slot(
        self,
        slot_id: str,
        data: pd.DataFrame,
        symbols: str = "AAPL",
        initial_cash: float = DEFAULT_INITIAL_CASH,
        commission_rate: float = DEFAULT_COMMISSION,
    ) -> Dict[str, Any]:
        """
        运行单个槽位的独立回测

        Args:
            slot_id: 槽位 ID
            data: 市场数据
            symbols: 交易标的
            initial_cash: 初始资金
            commission_rate: 佣金费率

        Returns:
            回测结果字典
        """
        slot = self.slots.get(slot_id)
        if not slot:
            return {"success": False, "error": f"槽位 {slot_id} 不存在"}

        if slot.strategy_class is None:
            return {"success": False, "error": f"槽位 {slot_id} 未配置策略"}

        try:
            import akquant as aq

            result = aq.run_backtest(
                data=data,
                strategy=slot.strategy_class,
                symbols=symbols,
                initial_cash=initial_cash,
                commission_rate=commission_rate,
                show_progress=False,
            )

            metrics = result.metrics
            metrics_dict = {
                "total_return_pct": float(metrics.total_return_pct) if metrics.total_return_pct is not None else 0.0,
                "annualized_return": float(metrics.annualized_return) if metrics.annualized_return is not None else 0.0,
                "sharpe_ratio": float(metrics.sharpe_ratio) if metrics.sharpe_ratio is not None else 0.0,
                "max_drawdown_pct": float(metrics.max_drawdown_pct) if metrics.max_drawdown_pct is not None else 0.0,
                "win_rate": float(metrics.win_rate) if metrics.win_rate is not None else 0.0,
                "total_trades": len(result.trades_df) if hasattr(result, 'trades_df') else 0,
            }

            self._results_cache[slot_id] = metrics_dict

            return {
                "success": True,
                "slot_id": slot_id,
                "metrics": metrics_dict,
                "result_object": result,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def get_cross_slot_metrics(self, results: Dict = None) -> Dict:
        """
        计算跨策略指标

        Args:
            results: 回测结果字典

        Returns:
            跨策略指标字典
        """
        cross_metrics = {
            "correlation_matrix": None,
            "portfolio_sharpe": None,
            "portfolio_max_drawdown": None,
            "slot_weights": {},
            "slot_performances": {},
        }

        # 获取各槽位的权重
        for slot_id, slot in self.slots.items():
            cross_metrics["slot_weights"][slot_id] = slot.weight

        # 如果有缓存的结果，计算组合指标
        if self._results_cache:
            total_weight = sum(slot.weight for slot in self.slots.values())

            if total_weight > 0:
                # 加权平均收益
                weighted_return = 0.0
                weighted_sharpe = 0.0
                max_individual_dd = 0.0

                for slot_id, slot in self.slots.items():
                    if slot_id in self._results_cache:
                        perf = self._results_cache[slot_id]
                        w = slot.weight / total_weight
                        weighted_return += perf.get("total_return_pct", 0) * w
                        weighted_sharpe += perf.get("sharpe_ratio", 0) * w
                        max_individual_dd = max(max_individual_dd, perf.get("max_drawdown_pct", 0))

                        cross_metrics["slot_performances"][slot_id] = {
                            "return_pct": perf.get("total_return_pct", 0),
                            "sharpe_ratio": perf.get("sharpe_ratio", 0),
                            "max_drawdown_pct": perf.get("max_drawdown_pct", 0),
                            "win_rate": perf.get("win_rate", 0),
                        }

                cross_metrics["portfolio_weighted_return"] = weighted_return
                cross_metrics["portfolio_sharpe"] = weighted_sharpe
                cross_metrics["portfolio_max_drawdown"] = max_individual_dd

        return cross_metrics

    def clear(self):
        """清空所有槽位"""
        self.slots.clear()
        self._results_cache.clear()
        self.active_snapshot = None

    def save_config(self, name: str) -> str:
        """保存多策略配置"""
        config_dir = BASE_PATH / "multi_strategy_configs"
        config_dir.mkdir(exist_ok=True)

        config = {
            "name": name,
            "saved_at": datetime.now().isoformat(),
            "slots": self.list_slots(),
            "active_snapshot": self.active_snapshot,
        }

        config_path = config_dir / f"{name}.json"
        with open(config_path, "w", encoding="utf-8") as f:
            import json
            json.dump(config, f, ensure_ascii=False, indent=2, default=str)

        return str(config_path)

    def load_config(self, name: str) -> bool:
        """加载多策略配置"""
        config_path = BASE_PATH / "multi_strategy_configs" / f"{name}.json"

        if not config_path.exists():
            return False

        try:
            import json
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            self.clear()
            self.active_snapshot = config.get("active_snapshot")

            for slot_data in config.get("slots", []):
                slot = StrategySlot(
                    slot_id=slot_data["slot_id"],
                    strategy_params=slot_data.get("strategy_params", {}),
                    max_order_size=slot_data.get("max_order_size", 10),
                    weight=slot_data.get("weight", 1.0),
                    description=slot_data.get("description", ""),
                )
                self.slots[slot.slot_id] = slot

            return True
        except Exception as e:
            print(f"[MultiStrategySimulator] 加载配置失败: {e}")
            return False


# =========================
# 全局实例
# =========================
_simulator: Optional[MultiStrategySimulator] = None


def get_multi_strategy_simulator() -> MultiStrategySimulator:
    """获取多策略模拟器单例"""
    global _simulator
    if _simulator is None:
        _simulator = MultiStrategySimulator()
    return _simulator


def list_saved_configs() -> List[Dict]:
    """列出已保存的多策略配置"""
    config_dir = BASE_PATH / "multi_strategy_configs"
    configs = []

    if not config_dir.exists():
        return configs

    for config_path in config_dir.glob("*.json"):
        try:
            import json
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            configs.append({
                "name": config.get("name", config_path.stem),
                "saved_at": config.get("saved_at", ""),
                "slot_count": len(config.get("slots", [])),
            })
        except Exception:
            pass

    return configs
