"""
AKQuant 热启动快照管理模块

管理 save_snapshot / run_warm_start 的生命周期，
支持快照的保存、恢复、列表展示和删除。
"""

import os
import json
import pickle
import traceback
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
from dataclasses import dataclass, asdict


# =========================
# 配置
# =========================
BASE_PATH = Path(__file__).parent.parent
CHECKPOINTS_DIR = BASE_PATH / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)


@dataclass
class SnapshotInfo:
    """快照元数据"""
    name: str
    created_at: str
    strategy_type: str = "unknown"
    model_type: str = "unknown"
    train_window: int = 0
    test_window: int = 0
    symbols: List[str] = None
    final_cash: float = 0.0
    total_return_pct: float = 0.0
    version: str = "0.2.2"
    file_size_mb: float = 0.0
    description: str = ""

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = []

    def to_dict(self) -> Dict:
        return asdict(self)


class SnapshotManager:
    """热启动快照管理器"""

    def __init__(self, checkpoint_dir: str = None):
        """
        Args:
            checkpoint_dir: 快照存储目录，默认为项目根目录/checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINTS_DIR
        self.checkpoint_dir.mkdir(exist_ok=True)

    def save(
        self,
        engine,
        strategy,
        name: str,
        metadata: Dict = None,
    ) -> str:
        """
        保存快照到 checkpoints/<name>.pkl

        Args:
            engine: AKQuant 回测引擎实例
            strategy: AKQuant 策略实例
            name: 快照名称
            metadata: 额外元数据

        Returns:
            快照文件路径
        """
        # 保存核心状态
        pkl_path = self.checkpoint_dir / f"{name}.pkl"
        meta_path = self.checkpoint_dir / f"{name}_meta.json"

        try:
            from akquant.checkpoint import save_snapshot
            save_snapshot(engine, strategy, str(pkl_path))
        except ImportError:
            # 降级：使用 pickle 手动保存
            snapshot_data = {
                "engine": engine,
                "strategy": strategy,
            }
            try:
                with open(pkl_path, "wb") as f:
                    pickle.dump(snapshot_data, f)
            except (pickle.PicklingError, TypeError, AttributeError) as e:
                # 如果策略包含不可 pickle 的对象，尝试保存可序列化的状态
                print(f"[SnapshotManager] 完整快照保存失败: {e}，尝试保存可序列化状态...")
                try:
                    # 提取可序列化的策略状态
                    strategy_state = {}
                    for attr in ['model_type', 'feature_config', 'train_window',
                                 'test_window', 'rolling_step', 'model_params',
                                 'warmup_period', '_trained', '_bar_count',
                                 'hidden_dim', 'num_layers', 'epochs', 'lr',
                                 '_probability_threshold', '_warmup']:
                        if hasattr(strategy, attr):
                            try:
                                val = getattr(strategy, attr)
                                pickle.dumps(val)  # 测试是否可序列化
                                strategy_state[attr] = val
                            except (pickle.PicklingError, TypeError):
                                pass

                    # 尝试单独序列化 sklearn 模型
                    if hasattr(strategy, '_model') and strategy._model is not None:
                        try:
                            model_bytes = pickle.dumps(strategy._model)
                            strategy_state['_model_bytes'] = model_bytes
                        except (pickle.PicklingError, TypeError):
                            pass

                    snapshot_data = {
                        "strategy_state": strategy_state,
                        "strategy_class_name": type(strategy).__name__,
                        "strategy_module": type(strategy).__module__,
                    }
                    with open(pkl_path, "wb") as f:
                        pickle.dump(snapshot_data, f)
                except Exception as e2:
                    raise RuntimeError(f"快照保存失败: {e2}") from e2

        # 保存元数据
        meta = SnapshotInfo(
            name=name,
            created_at=datetime.now().isoformat(),
            strategy_type=metadata.get("strategy_type", "unknown") if metadata else "unknown",
            model_type=metadata.get("model_type", "unknown") if metadata else "unknown",
            train_window=metadata.get("train_window", 0) if metadata else 0,
            test_window=metadata.get("test_window", 0) if metadata else 0,
            symbols=metadata.get("symbols", []) if metadata else [],
            final_cash=metadata.get("final_cash", 0.0) if metadata else 0.0,
            total_return_pct=metadata.get("total_return_pct", 0.0) if metadata else 0.0,
            description=metadata.get("description", "") if metadata else "",
            file_size_mb=pkl_path.stat().st_size / (1024 * 1024) if pkl_path.exists() else 0.0,
        )

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, ensure_ascii=False, indent=2)

        return str(pkl_path)

    def list_snapshots(self) -> List[Dict]:
        """
        列出所有可用快照

        Returns:
            快照元数据列表
        """
        snapshots = []

        for meta_path in self.checkpoint_dir.glob("*_meta.json"):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                # 检查对应的 pkl 文件是否存在
                pkl_path = meta_path.with_name(meta["name"] + ".pkl")
                if pkl_path.exists():
                    meta["file_size_mb"] = pkl_path.stat().st_size / (1024 * 1024)
                    snapshots.append(meta)
                else:
                    # pkl 不存在，删除孤立的 meta 文件
                    meta_path.unlink()

            except Exception as e:
                print(f"[SnapshotManager] 读取快照元数据失败: {meta_path} - {e}")

        # 按创建时间降序排列
        snapshots.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return snapshots

    def resume(
        self,
        checkpoint_name: str,
        new_data: Any,
        symbols: str = "AAPL",
        commission_rate: float = 0.0003,
        stamp_tax_rate: float = 0.001,
        transfer_fee_rate: float = 0.00001,
        initial_cash: float = None,
        config: Any = None,
        t_plus_one: bool = True,
    ) -> Dict[str, Any]:
        """
        从快照恢复并继续运行

        Args:
            checkpoint_name: 快照名称
            new_data: 新的回测数据 DataFrame
            symbols: 交易标的
            commission_rate: 佣金费率
            stamp_tax_rate: 印花税率
            transfer_fee_rate: 过户费率
            initial_cash: 初始资金（None 则使用快照恢复时的资金）
            config: AKQuant BacktestConfig
            t_plus_one: 是否启用 T+1

        Returns:
            回测结果字典（兼容 quantitative.py 的格式）
        """
        pkl_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"

        if not pkl_path.exists():
            return {
                "success": False,
                "error": f"快照文件不存在: {pkl_path}",
            }

        try:
            import akquant as aq

            # 构建 warm_start 参数
            warm_start_kwargs = {
                "checkpoint_path": str(pkl_path),
                "data": new_data,
                "symbols": symbols,
                "commission_rate": commission_rate,
                "stamp_tax_rate": stamp_tax_rate,
                "transfer_fee_rate": transfer_fee_rate,
                "t_plus_one": t_plus_one,
                "show_progress": False,
            }

            if config is not None:
                warm_start_kwargs["config"] = config

            # 执行热启动
            result = aq.run_warm_start(**warm_start_kwargs)

            # 提取指标
            metrics = result.metrics
            metrics_dict = {
                "total_return_pct": metrics.total_return_pct,
                "annualized_return": metrics.annualized_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "win_rate": metrics.win_rate,
                "total_trades": len(result.trades_df) if hasattr(result, 'trades_df') else 0,
                "final_value": metrics.final_value if hasattr(metrics, 'final_value') else 0,
            }

            # 转换为可序列化的格式
            for key in metrics_dict:
                if metrics_dict[key] is not None and not isinstance(metrics_dict[key], dict):
                    metrics_dict[key] = float(metrics_dict[key])

            return {
                "success": True,
                "metrics": metrics_dict,
                "trades_count": metrics_dict["total_trades"],
                "result_object": result,
                "resumed_from": checkpoint_name,
                "initial_cash_adjusted": initial_cash is None,
            }

        except ImportError:
            return {
                "success": False,
                "error": "AKQuant 未安装，无法执行热启动",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def delete(self, name: str) -> bool:
        """
        删除指定快照

        Args:
            name: 快照名称

        Returns:
            是否删除成功
        """
        pkl_path = self.checkpoint_dir / f"{name}.pkl"
        meta_path = self.checkpoint_dir / f"{name}_meta.json"

        deleted = False

        if pkl_path.exists():
            pkl_path.unlink()
            deleted = True

        if meta_path.exists():
            meta_path.unlink()
            deleted = True

        return deleted

    def get_snapshot_info(self, name: str) -> Optional[Dict]:
        """
        获取指定快照的元数据

        Args:
            name: 快照名称

        Returns:
            元数据字典，不存在则返回 None
        """
        meta_path = self.checkpoint_dir / f"{name}_meta.json"

        if not meta_path.exists():
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None


# =========================
# 全局实例
# =========================
_snapshot_manager: Optional[SnapshotManager] = None


def get_snapshot_manager() -> SnapshotManager:
    """获取快照管理器单例"""
    global _snapshot_manager
    if _snapshot_manager is None:
        _snapshot_manager = SnapshotManager()
    return _snapshot_manager


def save_snapshot(engine, strategy, name: str, metadata: Dict = None) -> str:
    """快捷函数：保存快照"""
    manager = get_snapshot_manager()
    return manager.save(engine, strategy, name, metadata)


def list_snapshots() -> List[Dict]:
    """快捷函数：列出快照"""
    manager = get_snapshot_manager()
    return manager.list_snapshots()


def resume_snapshot(checkpoint_name: str, new_data: Any, **kwargs) -> Dict:
    """快捷函数：从快照恢复"""
    manager = get_snapshot_manager()
    return manager.resume(checkpoint_name, new_data, **kwargs)


def delete_snapshot(name: str) -> bool:
    """快捷函数：删除快照"""
    manager = get_snapshot_manager()
    return manager.delete(name)
