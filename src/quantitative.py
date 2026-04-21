"""
AKQuant 量化策略模块 - 与 FinRAG 系统集成

提供自然语言驱动的量化策略创建和回测功能
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
import json
import traceback

# AKQuant 已通过 uv 安装到虚拟环境，不需要手动添加路径
# 如果需要使用本地开发版本，可以使用 maturin develop 安装
_akquant_path_added = True  # 标记为已处理


def _ensure_akquant_path():
    """akquant 已安装，无需额外处理"""
    pass

# =========================
# 🔹 配置
# =========================
AKQUANT_ENABLED = True
DEFAULT_INITIAL_CASH = 100_000.0
DEFAULT_COMMISSION = 0.0003

# 预定义策略模板
STRATEGY_TEMPLATES = {
    # === 规则策略 ===
    "dual_ma": {
        "name": "双均线策略",
        "description": "金叉买入，死叉卖出",
        "category": "rule",
        "params": {
            "fast_window": {"type": "int", "default": 10, "min": 2, "max": 50},
            "slow_window": {"type": "int", "default": 30, "min": 5, "max": 200}
        }
    },
    "rsi": {
        "name": "RSI 策略",
        "description": "RSI 超卖买入，超买卖出",
        "category": "rule",
        "params": {
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "oversold": {"type": "int", "default": 30, "min": 10, "max": 40},
            "overbought": {"type": "int", "default": 70, "min": 60, "max": 90}
        }
    },
    "macd": {
        "name": "MACD 策略",
        "description": "MACD 金叉死叉信号",
        "category": "rule",
        "params": {
            "fast_period": {"type": "int", "default": 12, "min": 5, "max": 30},
            "slow_period": {"type": "int", "default": 26, "min": 15, "max": 50},
            "signal_period": {"type": "int", "default": 9, "min": 5, "max": 20}
        }
    },
    "bollinger": {
        "name": "布林带策略",
        "description": "价格触及下轨买入，上轨卖出",
        "category": "rule",
        "params": {
            "window": {"type": "int", "default": 20, "min": 10, "max": 50},
            "num_std": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0}
        }
    },
    # === ML 策略 ===
    "ml_logistic": {
        "name": "逻辑回归 Walk-forward",
        "description": "使用逻辑回归预测涨跌，Walk-forward 滚动训练",
        "category": "ml",
        "params": {
            "train_window": {"type": "int", "default": 50, "min": 20, "max": 500},
            "test_window": {"type": "int", "default": 20, "min": 5, "max": 100},
            "rolling_step": {"type": "int", "default": 10, "min": 1, "max": 50},
            "feature_config": {"type": "select", "default": "basic", "options": ["basic", "technical", "extended"]},
        }
    },
    "ml_xgboost": {
        "name": "XGBoost Walk-forward",
        "description": "使用 XGBoost 预测涨跌，支持特征重要性",
        "category": "ml",
        "params": {
            "train_window": {"type": "int", "default": 50, "min": 20, "max": 500},
            "test_window": {"type": "int", "default": 20, "min": 5, "max": 100},
            "rolling_step": {"type": "int", "default": 10, "min": 1, "max": 50},
            "feature_config": {"type": "select", "default": "technical", "options": ["basic", "technical", "extended"]},
            "max_depth": {"type": "int", "default": 5, "min": 2, "max": 10},
            "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500},
        }
    },
    "ml_lightgbm": {
        "name": "LightGBM Walk-forward",
        "description": "使用 LightGBM 预测涨跌，速度快适合大数据集",
        "category": "ml",
        "params": {
            "train_window": {"type": "int", "default": 50, "min": 20, "max": 500},
            "test_window": {"type": "int", "default": 20, "min": 5, "max": 100},
            "rolling_step": {"type": "int", "default": 10, "min": 1, "max": 50},
            "feature_config": {"type": "select", "default": "technical", "options": ["basic", "technical", "extended"]},
            "num_leaves": {"type": "int", "default": 31, "min": 10, "max": 100},
            "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500},
        }
    },
    "ml_random_forest": {
        "name": "随机森林 Walk-forward",
        "description": "使用随机森林预测涨跌，鲁棒性强",
        "category": "ml",
        "params": {
            "train_window": {"type": "int", "default": 50, "min": 20, "max": 500},
            "test_window": {"type": "int", "default": 20, "min": 5, "max": 100},
            "rolling_step": {"type": "int", "default": 10, "min": 1, "max": 50},
            "feature_config": {"type": "select", "default": "basic", "options": ["basic", "technical", "extended"]},
            "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500},
        }
    },
    "ml_lstm": {
        "name": "LSTM 深度学习策略",
        "description": "使用 PyTorch LSTM 进行序列预测",
        "category": "ml",
        "params": {
            "train_window": {"type": "int", "default": 50, "min": 20, "max": 500},
            "test_window": {"type": "int", "default": 20, "min": 5, "max": 100},
            "rolling_step": {"type": "int", "default": 10, "min": 1, "max": 50},
            "feature_config": {"type": "select", "default": "basic", "options": ["basic", "technical", "extended"]},
            "hidden_dim": {"type": "int", "default": 32, "min": 8, "max": 256},
            "num_layers": {"type": "int", "default": 2, "min": 1, "max": 4},
            "epochs": {"type": "int", "default": 20, "min": 5, "max": 100},
            "lr": {"type": "float", "default": 0.001, "min": 0.0001, "max": 0.01},
        }
    },
}


def get_available_strategies() -> List[Dict]:
    """获取可用的策略模板列表"""
    return [
        {"id": k, "name": v["name"], "description": v["description"],
         "category": v.get("category", "rule")}
        for k, v in STRATEGY_TEMPLATES.items()
    ]


def get_available_rule_strategies() -> List[Dict]:
    """获取规则策略模板列表"""
    return [
        {"id": k, "name": v["name"], "description": v["description"]}
        for k, v in STRATEGY_TEMPLATES.items()
        if v.get("category", "rule") == "rule"
    ]


def get_available_ml_strategies() -> List[Dict]:
    """获取 ML 策略模板列表"""
    return [
        {"id": k, "name": v["name"], "description": v["description"]}
        for k, v in STRATEGY_TEMPLATES.items()
        if v.get("category", "rule") == "ml"
    ]


def check_akquant_available() -> Tuple[bool, str]:
    """检查 AKQuant 是否可用"""
    try:
        _ensure_akquant_path()
        from akquant import Strategy, run_backtest
        return True, "AKQuant 已就绪"
    except ImportError as e:
        return False, f"AKQuant 未安装: {e}"
    except Exception as e:
        return False, f"AKQuant 加载失败: {e}"


def create_strategy_instance(strategy_type: str, params: Dict) -> Any:
    """
    根据策略类型和参数创建策略实例
    
    Args:
        strategy_type: 策略类型 (dual_ma, rsi, macd, bollinger, ml_*)
        params: 策略参数
        
    Returns:
        Strategy 类
    """
    # ML 策略走专用工厂
    template = STRATEGY_TEMPLATES.get(strategy_type, {})
    if template.get("category") == "ml":
        return create_ml_strategy_instance(strategy_type, params)
    
    from akquant import Strategy
    import numpy as np
    
    if strategy_type == "dual_ma":
        fast = params.get("fast_window", 10)
        slow = params.get("slow_window", 30)
        
        class DualMAStrategy(Strategy):
            warmup_period = slow + 10
            
            def __init__(self):
                self.fast_window = fast
                self.slow_window = slow
                self.warmup_period = slow + 10
                
            def on_start(self):
                self.subscribe("AAPL")
                
            def on_bar(self, bar):
                closes = self.get_history(count=self.slow_window, symbol=bar.symbol, field="close")
                if len(closes) < self.slow_window:
                    return
                
                fast_ma = np.mean(closes[-self.fast_window:])
                slow_ma = np.mean(closes[-self.slow_window:])
                position = self.get_position(bar.symbol)
                
                if fast_ma > slow_ma and position == 0:
                    self.buy(symbol=bar.symbol, quantity=100)
                elif fast_ma < slow_ma and position > 0:
                    self.sell(symbol=bar.symbol, quantity=position)
        
        return DualMAStrategy
    
    elif strategy_type == "rsi":
        period = params.get("rsi_period", 14)
        oversold = params.get("oversold", 30)
        overbought = params.get("overbought", 70)
        
        class RSIStrategy(Strategy):
            warmup_period = period + 10
            
            def __init__(self):
                self.rsi_period = period
                self.oversold = oversold
                self.overbought = overbought
                self.warmup_period = period + 10
                
            def on_start(self):
                self.subscribe("AAPL")
                
            def on_bar(self, bar):
                closes = self.get_history(count=self.rsi_period + 1, symbol=bar.symbol, field="close")
                if len(closes) < self.rsi_period + 1:
                    return
                
                # 计算 RSI
                deltas = np.diff(closes)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains[-self.rsi_period:])
                avg_loss = np.mean(losses[-self.rsi_period:])
                
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                
                position = self.get_position(bar.symbol)
                
                if rsi < self.oversold and position == 0:
                    self.buy(symbol=bar.symbol, quantity=100)
                elif rsi > self.overbought and position > 0:
                    self.sell(symbol=bar.symbol, quantity=position)
        
        return RSIStrategy
    
    elif strategy_type == "macd":
        fast = params.get("fast_period", 12)
        slow = params.get("slow_period", 26)
        signal = params.get("signal_period", 9)
        
        class MACDStrategy(Strategy):
            warmup_period = slow + signal + 10
            
            def __init__(self):
                self.fast_period = fast
                self.slow_period = slow
                self.signal_period = signal
                self.warmup_period = slow + signal + 10
                
            def on_start(self):
                self.subscribe("AAPL")
                
            def _ema(self, data, period):
                """计算 EMA"""
                alpha = 2 / (period + 1)
                ema = [data[0]]
                for i in range(1, len(data)):
                    ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
                return np.array(ema)
                
            def on_bar(self, bar):
                period = self.slow_period + self.signal_period
                closes = self.get_history(count=period + 1, symbol=bar.symbol, field="close")
                if len(closes) < period + 1:
                    return
                
                # 计算 MACD
                ema_fast = self._ema(closes, self.fast_period)
                ema_slow = self._ema(closes, self.slow_period)
                macd_line = ema_fast - ema_slow
                
                signal_line = self._ema(macd_line, self.signal_period)
                
                position = self.get_position(bar.symbol)
                
                # 金叉
                if macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2] and position == 0:
                    self.buy(symbol=bar.symbol, quantity=100)
                # 死叉
                elif macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2] and position > 0:
                    self.sell(symbol=bar.symbol, quantity=position)
        
        return MACDStrategy
    
    elif strategy_type == "bollinger":
        window = params.get("window", 20)
        num_std = params.get("num_std", 2.0)
        
        class BollingerStrategy(Strategy):
            warmup_period = window + 10
            
            def __init__(self):
                self.window = window
                self.num_std = num_std
                self.warmup_period = window + 10
                
            def on_start(self):
                self.subscribe("AAPL")
                
            def on_bar(self, bar):
                closes = self.get_history(count=self.window, symbol=bar.symbol, field="close")
                if len(closes) < self.window:
                    return
                
                # 计算布林带
                sma = np.mean(closes)
                std = np.std(closes)
                lower_band = sma - self.num_std * std
                upper_band = sma + self.num_std * std
                
                position = self.get_position(bar.symbol)
                
                # 价格触及下轨买入
                if bar.close <= lower_band and position == 0:
                    self.buy(symbol=bar.symbol, quantity=100)
                # 价格触及上轨卖出
                elif bar.close >= upper_band and position > 0:
                    self.sell(symbol=bar.symbol, quantity=position)
        
        return BollingerStrategy
    
    else:
        raise ValueError(f"未知策略类型: {strategy_type}")


def create_ml_strategy_instance(strategy_type: str, params: Dict) -> Any:
    """
    根据 ML 策略类型和参数创建策略类（委托给 ml_strategy 模块）

    Args:
        strategy_type: ML 策略类型 (ml_logistic, ml_xgboost, etc.)
        params: 策略参数

    Returns:
        Strategy 类
    """
    from src.ml_strategy import create_ml_strategy_instance as _create_ml
    return _create_ml(strategy_type, params)


def run_backtest(
    data,
    strategy_type: str,
    strategy_params: Dict,
    symbol: str = "AAPL",
    start_date: str = "20230101",
    end_date: str = "20231231",
    initial_cash: float = DEFAULT_INITIAL_CASH,
    commission_rate: float = DEFAULT_COMMISSION,
    benchmark_data: Any = None,
    on_event: Any = None,
    ml_config: Dict = None,
    checkpoint_path: str = None,
    save_checkpoint: str = None,
) -> Dict[str, Any]:
    """
    运行回测（支持基准对比 + 流式事件 + ML + 热启动）
    
    Args:
        data: DataFrame，包含 date, open, high, low, close, volume, symbol 列
        strategy_type: 策略类型
        strategy_params: 策略参数
        symbol: 交易标的
        start_date: 开始日期
        end_date: 结束日期
        initial_cash: 初始资金
        commission_rate: 佣金费率
        benchmark_data: 基准收益序列（Series），用于基准对比
        on_event: 流式回测回调函数
        ml_config: ML 特有配置（Walk-forward 参数覆盖）
        checkpoint_path: 热启动：从快照恢复路径
        save_checkpoint: 热启动：回测后保存快照的名称
        
    Returns:
        回测结果字典
    """
    try:
        from akquant import run_backtest as ak_run_backtest
        import pandas as pd
        
        # 确保 data 有正确的格式
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            if "symbol" not in df.columns:
                df["symbol"] = symbol
        else:
            raise ValueError("数据必须是 pandas DataFrame")
        
        # 创建策略类
        strategy_class = create_strategy_instance(strategy_type, strategy_params)
        
        # 构建回测参数
        backtest_kwargs = {
            "data": df,
            "strategy": strategy_class,
            "symbols": symbol,
            "initial_cash": initial_cash,
            "commission_rate": commission_rate,
            "show_progress": False,
            "start_time": start_date,
            "end_time": end_date,
            "strict_strategy_params": False
        }
        
        # 添加基准对比（如果提供）
        if benchmark_data is not None:
            backtest_kwargs["benchmark"] = benchmark_data
        
        # 添加流式回调（如果提供）
        if on_event is not None:
            backtest_kwargs["on_event"] = on_event
            backtest_kwargs["stream_progress_interval"] = 10
            backtest_kwargs["stream_equity_interval"] = 10
        
        # 运行回测
        result = ak_run_backtest(**backtest_kwargs)
        
        # 提取指标
        metrics = result.metrics
        metrics_dict = {
            "total_return_pct": metrics.total_return_pct,
            "annualized_return": metrics.annualized_return,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "win_rate": metrics.win_rate,
            "total_trades": len(result.trades_df) if hasattr(result, 'trades_df') else 0,
            "final_value": metrics.final_value if hasattr(metrics, 'final_value') else initial_cash * (1 + metrics.total_return_pct / 100)
        }
        
        # 提取基准对比指标（如果可用）
        if benchmark_data is not None and hasattr(result, 'get_event_stats'):
            try:
                event_stats = result.get_event_stats()
                metrics_dict["event_stats"] = event_stats
            except Exception:
                pass
        
        # 转换为可序列化的格式
        for key in metrics_dict:
            if metrics_dict[key] is not None and not isinstance(metrics_dict[key], dict):
                metrics_dict[key] = float(metrics_dict[key])
        
        return {
            "success": True,
            "metrics": metrics_dict,
            "trades_count": metrics_dict["total_trades"],
            "result_object": result,
            "has_benchmark": benchmark_data is not None
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def create_benchmark_from_data(data: Any, symbol: str = "BENCH") -> Any:
    """
    从市场数据创建基准收益序列
    
    Args:
        data: 市场数据 DataFrame，需包含 date, close 列
        symbol: 基准名称
        
    Returns:
        pandas Series: 每日收益率序列
    """
    import pandas as pd
    
    if isinstance(data, pd.DataFrame):
        if "date" in data.columns:
            df = data.set_index("date")
        else:
            df = data
        
        if "close" in df.columns:
            returns = df["close"].pct_change().fillna(0.0)
            returns.name = symbol
            return returns
    
    return None


def create_streaming_callback() -> callable:
    """
    创建流式回测回调函数
    
    Returns:
        回调函数，用于接收流式事件
    """
    import pandas as pd
    
    class StreamingCallback:
        def __init__(self):
            self.progress_events = 0
            self.equity_events = []
            self.order_events = []
            self.trade_events = []
            
        def __call__(self, event: dict):
            event_type = event.get("event_type", "")
            
            if event_type == "progress":
                self.progress_events += 1
                if self.progress_events % 10 == 0:
                    print(f"[进度] {self.progress_events} 批次完成", flush=True)
                    
            elif event_type == "equity":
                self.equity_events.append({
                    "timestamp": event.get("ts"),
                    "equity": event.get("payload", {}).get("equity", 0)
                })
                
            elif event_type == "order":
                self.order_events.append(event.get("payload", {}))
                
            elif event_type == "trade":
                self.trade_events.append(event.get("payload", {}))
                
            elif event_type == "finished":
                status = event.get("payload", {}).get("status", "unknown")
                print(f"[完成] 状态: {status}", flush=True)
    
    return StreamingCallback()


def generate_report(
    result_dict: Dict,
    strategy_name: str,
    symbol: str,
    market_data=None,
    output_path: str = None,
    benchmark_data: Any = None
) -> Optional[str]:
    """
    生成可视化报告 HTML（支持基准对比）
    
    Args:
        result_dict: run_backtest 返回的结果字典
        strategy_name: 策略名称
        symbol: 交易标的
        market_data: 市场数据 DataFrame
        output_path: 输出文件路径
        benchmark_data: 基准收益序列
        
    Returns:
        HTML 文件路径
    """
    try:
        if not result_dict.get("success"):
            return None
            
        result_obj = result_dict.get("result_object")
        if result_obj is None:
            return None
            
        # 生成报告
        if output_path is None:
            output_path = f"reports/quant_report_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        
        # 准备报告参数
        report_kwargs = {
            "title": f"量化策略报告 - {strategy_name} ({symbol})",
            "filename": output_path,
            "show": False,
            "market_data": market_data,
            "plot_symbol": symbol,
            "include_trade_kline": True
        }
        
        # 添加基准对比（如果提供）
        if benchmark_data is not None:
            report_kwargs["benchmark"] = benchmark_data
        
        result_obj.report(**report_kwargs)
        
        return output_path
        
    except Exception as e:
        print(f"生成报告失败: {e}")
        return None


def get_strategy_metrics_description(metrics: Dict) -> str:
    """
    生成策略指标的中文描述
    
    Args:
        metrics: 指标字典
        
    Returns:
        中文描述字符串
    """
    total_return = metrics.get("total_return_pct", 0)
    annualized = metrics.get("annualized_return", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    max_dd = metrics.get("max_drawdown_pct", 0)
    win_rate = metrics.get("win_rate", 0)
    trades = metrics.get("total_trades", 0)
    
    # 性能评估
    if total_return > 20:
        perf = "表现优秀"
    elif total_return > 0:
        perf = "小幅盈利"
    elif total_return > -10:
        perf = "小幅亏损"
    else:
        perf = "亏损较大"
    
    # 夏普比率评估
    if sharpe > 2:
        sharpe_eval = "风险调整收益极佳"
    elif sharpe > 1:
        sharpe_eval = "风险调整收益良好"
    elif sharpe > 0:
        sharpe_eval = "风险调整收益一般"
    else:
        sharpe_eval = "风险调整收益较差"
    
    return f"""
## 📊 策略绩效报告

| 指标 | 数值 | 评估 |
|------|------|------|
| 总收益率 | {total_return:.2f}% | {perf} |
| 年化收益率 | {annualized:.2f}% | - |
| 夏普比率 | {sharpe:.2f} | {sharpe_eval} |
| 最大回撤 | {max_dd:.2f}% | - |
| 胜率 | {win_rate:.1f}% | - |
| 总交易次数 | {trades} 次 | - |

**总结**: 该策略在回测期间{perf}，{sharpe_eval}，共执行了 {trades} 次交易。
"""


def parse_natural_language_strategy(user_input: str) -> Optional[Dict]:
    """
    解析自然语言策略描述
    
    Args:
        user_input: 用户输入的自然语言
        
    Returns:
        解析结果，包含 strategy_type, params, symbol 等
    """
    user_input_lower = user_input.lower()
    result = {
        "strategy_type": None,
        "params": {},
        "symbol": "AAPL",
        "start_date": "20230101",
        "end_date": "20231231"
    }
    
    # 识别策略类型
    if "均线" in user_input or "ma" in user_input_lower:
        result["strategy_type"] = "dual_ma"
        # 尝试提取参数
        import re
        fast_match = re.search(r'(\d+)[日天]?线|fast.*?(\d+)', user_input)
        slow_match = re.search(r'(\d+)[日天]?.*均线|slow.*?(\d+)', user_input)
        if fast_match:
            result["params"]["fast_window"] = int(fast_match.group(1) or fast_match.group(2))
        if slow_match:
            result["params"]["slow_window"] = int(slow_match.group(1) or slow_match.group(2))
    elif "rsi" in user_input_lower:
        result["strategy_type"] = "rsi"
    elif "macd" in user_input_lower:
        result["strategy_type"] = "macd"
    elif "布林" in user_input or "bollinger" in user_input_lower:
        result["strategy_type"] = "bollinger"
    
    # 识别股票代码
    import re
    stock_match = re.search(r'([0-9]{6})|([a-z]{2}[0-9]{6})', user_input_lower)
    if stock_match:
        result["symbol"] = stock_match.group(0).lower()
    
    # 识别日期范围
    date_match = re.search(r'(\d{4}).*?(\d{4})', user_input)
    if date_match:
        result["start_date"] = date_match.group(1) + "0101"
        result["end_date"] = date_match.group(2) + "1231"
    
    return result if result["strategy_type"] else None


# =========================
# 🔹 触发关键词
# =========================
QUANT_TRIGGER_KEYWORDS = [
    "量化", "回测", "策略", "交易策略", "均线", "macd", "rsi", "布林带",
    "金叉", "死叉", "买入", "卖出", "交易", "收益率", "夏普比率", "回测",
    "backtest", "strategy", "quantitative",
    # ML / 热启动 / 多策略
    "机器学习", "训练模型", "滚动训练", "walk_forward", "walkforward",
    "热启动", "快照恢复", "checkpoint", "warm_start", "warmstart",
    "多策略", "模拟盘", "组合策略", "slot",
    "xgboost", "lightgbm", "randomforest", "lstm", "transformer",
    "特征工程", "pipeline", "交叉验证", "过拟合", "深度学习",
]


def should_trigger_quant(user_input: str) -> bool:
    """检查是否应该触发量化功能"""
    user_input_lower = user_input.lower()
    return any(keyword in user_input_lower for keyword in QUANT_TRIGGER_KEYWORDS)
