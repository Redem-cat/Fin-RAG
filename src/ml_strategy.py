"""
AKQuant ML 策略模块 - Walk-forward Validation + SklearnAdapter / PyTorchAdapter

提供 ML 策略工厂、特征工程基类、模型持久化等功能。
与 FinRAG 系统集成，支持自然语言驱动的 ML 策略创建和回测。
"""

import os
import pickle
import json
import traceback
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Callable, TypedDict
from datetime import datetime

import numpy as np
import pandas as pd


# =========================
# 配置
# =========================
ML_MODELS_DIR = Path(__file__).parent.parent / "ml_models"
ML_MODELS_DIR.mkdir(exist_ok=True)


class MLFeatureConfig(TypedDict, total=False):
    """ML 特征配置"""
    features: List[str]          # 特征列表，如 ["ret1", "ret2", "rsi", "macd"]
    label_shift: int             # 标签 shift，默认 -1（下一期涨跌）
    history_depth: int           # 计算特征所需历史 bar 数
    threshold: float             # 涨跌分类阈值，默认 0.0


# =========================
# 默认特征配置
# =========================
DEFAULT_FEATURE_CONFIGS = {
    "basic": {
        "features": ["ret1", "ret2", "ret3", "ret5"],
        "label_shift": -1,
        "history_depth": 10,
        "threshold": 0.0,
    },
    "technical": {
        "features": ["ret1", "ret2", "rsi_14", "macd_diff", "bollinger_pos"],
        "label_shift": -1,
        "history_depth": 30,
        "threshold": 0.0,
    },
    "extended": {
        "features": ["ret1", "ret2", "ret3", "ret5", "rsi_14", "macd_diff",
                     "bollinger_pos", "vol_ratio", "ma5_ma20_ratio"],
        "label_shift": -1,
        "history_depth": 30,
        "threshold": 0.0,
    },
}

# ML 策略模板定义
ML_STRATEGY_TEMPLATES = {
    "ml_logistic": {
        "name": "逻辑回归 Walk-forward",
        "description": "使用逻辑回归预测涨跌，Walk-forward 滚动训练",
        "category": "ml",
        "model_type": "logistic_regression",
        "params": {
            "train_window": {"type": "int", "default": 50, "min": 20, "max": 500, "label": "训练窗口"},
            "test_window": {"type": "int", "default": 20, "min": 5, "max": 100, "label": "测试窗口"},
            "rolling_step": {"type": "int", "default": 10, "min": 1, "max": 50, "label": "滚动步长"},
            "feature_config": {"type": "select", "default": "basic",
                               "options": list(DEFAULT_FEATURE_CONFIGS.keys()), "label": "特征集"},
        }
    },
    "ml_xgboost": {
        "name": "XGBoost Walk-forward",
        "description": "使用 XGBoost 预测涨跌，支持特征重要性分析",
        "category": "ml",
        "model_type": "xgboost",
        "params": {
            "train_window": {"type": "int", "default": 50, "min": 20, "max": 500, "label": "训练窗口"},
            "test_window": {"type": "int", "default": 20, "min": 5, "max": 100, "label": "测试窗口"},
            "rolling_step": {"type": "int", "default": 10, "min": 1, "max": 50, "label": "滚动步长"},
            "feature_config": {"type": "select", "default": "technical",
                               "options": list(DEFAULT_FEATURE_CONFIGS.keys()), "label": "特征集"},
            "max_depth": {"type": "int", "default": 5, "min": 2, "max": 10, "label": "最大深度"},
            "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500, "label": "树数量"},
        }
    },
    "ml_lightgbm": {
        "name": "LightGBM Walk-forward",
        "description": "使用 LightGBM 预测涨跌，速度快适合大数据集",
        "category": "ml",
        "model_type": "lightgbm",
        "params": {
            "train_window": {"type": "int", "default": 50, "min": 20, "max": 500, "label": "训练窗口"},
            "test_window": {"type": "int", "default": 20, "min": 5, "max": 100, "label": "测试窗口"},
            "rolling_step": {"type": "int", "default": 10, "min": 1, "max": 50, "label": "滚动步长"},
            "feature_config": {"type": "select", "default": "technical",
                               "options": list(DEFAULT_FEATURE_CONFIGS.keys()), "label": "特征集"},
            "num_leaves": {"type": "int", "default": 31, "min": 10, "max": 100, "label": "叶子数"},
            "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500, "label": "树数量"},
        }
    },
    "ml_random_forest": {
        "name": "随机森林 Walk-forward",
        "description": "使用随机森林预测涨跌，鲁棒性强",
        "category": "ml",
        "model_type": "random_forest",
        "params": {
            "train_window": {"type": "int", "default": 50, "min": 20, "max": 500, "label": "训练窗口"},
            "test_window": {"type": "int", "default": 20, "min": 5, "max": 100, "label": "测试窗口"},
            "rolling_step": {"type": "int", "default": 10, "min": 1, "max": 50, "label": "滚动步长"},
            "feature_config": {"type": "select", "default": "basic",
                               "options": list(DEFAULT_FEATURE_CONFIGS.keys()), "label": "特征集"},
            "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500, "label": "树数量"},
        }
    },
    "ml_lstm": {
        "name": "LSTM 深度学习策略",
        "description": "使用 PyTorch LSTM 进行序列预测",
        "category": "ml",
        "model_type": "lstm",
        "params": {
            "train_window": {"type": "int", "default": 50, "min": 20, "max": 500, "label": "训练窗口"},
            "test_window": {"type": "int", "default": 20, "min": 5, "max": 100, "label": "测试窗口"},
            "rolling_step": {"type": "int", "default": 10, "min": 1, "max": 50, "label": "滚动步长"},
            "feature_config": {"type": "select", "default": "basic",
                               "options": list(DEFAULT_FEATURE_CONFIGS.keys()), "label": "特征集"},
            "hidden_dim": {"type": "int", "default": 32, "min": 8, "max": 256, "label": "隐藏层维度"},
            "num_layers": {"type": "int", "default": 2, "min": 1, "max": 4, "label": "LSTM 层数"},
            "epochs": {"type": "int", "default": 20, "min": 5, "max": 100, "label": "训练轮数"},
            "lr": {"type": "float", "default": 0.001, "min": 0.0001, "max": 0.01, "label": "学习率"},
        }
    },
}


def get_available_ml_strategies() -> List[Dict]:
    """获取可用的 ML 策略模板列表"""
    return [
        {"id": k, "name": v["name"], "description": v["description"],
         "category": v["category"], "model_type": v["model_type"]}
        for k, v in ML_STRATEGY_TEMPLATES.items()
    ]


# =========================
# 特征工程
# =========================
def compute_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """
    根据特征名称列表计算特征列

    Args:
        df: 包含 OHLCV 数据的 DataFrame
        feature_names: 特征名称列表

    Returns:
        添加了特征列的 DataFrame
    """
    result = df.copy()

    for feat in feature_names:
        if feat in result.columns:
            continue

        if feat.startswith("ret") and feat[3:].isdigit():
            # 收益率特征: ret1, ret2, ret3, ret5
            n = int(feat[3:])
            result[feat] = result["close"].pct_change(n)

        elif feat == "rsi_14":
            delta = result["close"].diff()
            gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            result[feat] = 100 - (100 / (1 + rs))

        elif feat == "macd_diff":
            ema12 = result["close"].ewm(span=12).mean()
            ema26 = result["close"].ewm(span=26).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9).mean()
            result[feat] = dif - dea

        elif feat == "bollinger_pos":
            # 布林带位置 (0~1 之间，0.5 为中轨)
            sma = result["close"].rolling(window=20).mean()
            std = result["close"].rolling(window=20).std()
            upper = sma + 2 * std
            lower = sma - 2 * std
            result[feat] = (result["close"] - lower) / (upper - lower + 1e-10)

        elif feat == "vol_ratio":
            # 成交量比率
            vol_ma = result["volume"].rolling(window=20).mean()
            result[feat] = result["volume"] / (vol_ma + 1e-10)

        elif feat == "ma5_ma20_ratio":
            ma5 = result["close"].rolling(window=5).mean()
            ma20 = result["close"].rolling(window=20).mean()
            result[feat] = ma5 / (ma20 + 1e-10)

    return result


def prepare_features(df: pd.DataFrame, feature_config: Dict, mode: str = "training") -> Any:
    """
    特征工程：根据配置生成特征矩阵和标签

    Args:
        df: 包含 OHLCV 数据的 DataFrame
        feature_config: 特征配置字典
        mode: "training" 返回 (X, y), "inference" 返回 X

    Returns:
        training: (X, y) 元组
        inference: X (最后一行)
    """
    feature_names = feature_config.get("features", ["ret1", "ret2"])
    label_shift = feature_config.get("label_shift", -1)
    threshold = feature_config.get("threshold", 0.0)

    # 计算特征
    df = compute_features(df, feature_names)

    # 提取特征矩阵
    feature_cols = [c for c in feature_names if c in df.columns]
    X = df[feature_cols].copy()

    # 填充 NaN
    X = X.fillna(0)

    if mode == "training":
        # 计算标签：下一期涨跌
        future_return = df["close"].pct_change().shift(label_shift)
        y = (future_return > threshold).astype(int)
        y = y.fillna(0)
        return X.values, y.values
    else:
        # 推理：返回最后一行
        return X.iloc[[-1]].values if len(X) > 0 else np.zeros((1, len(feature_cols)))


# =========================
# ML 策略工厂
# =========================
# =========================
# ML 策略类（模块级，支持 pickle 序列化）
# =========================
try:
    from akquant import Strategy
except ImportError:
    Strategy = object


class WalkForwardSklearnStrategy(Strategy):
    """Walk-forward Sklearn 策略（模块级类，支持 pickle）

    工厂通过类属性 _cfg_* 注入配置，__init__() 无参数，
    以兼容 AKQuant 的 run_backtest(strategy=cls) 调用方式。
    """

    # 类级配置（由工厂注入）
    _cfg_model_type = "logistic_regression"
    _cfg_feature_config = None  # 将使用 DEFAULT_FEATURE_CONFIGS["basic"]
    _cfg_train_window = 50
    _cfg_test_window = 20
    _cfg_rolling_step = 10
    _cfg_model_params = None
    _cfg_feature_fn = None

    def __init__(self):
        self.model_type = self.__class__._cfg_model_type
        self.feature_config = self.__class__._cfg_feature_config or DEFAULT_FEATURE_CONFIGS["basic"]
        self.train_window = self.__class__._cfg_train_window
        self.test_window = self.__class__._cfg_test_window
        self.rolling_step = self.__class__._cfg_rolling_step
        self.model_params = self.__class__._cfg_model_params or {}
        self._feature_fn = self.__class__._cfg_feature_fn or prepare_features
        self._warmup = self.feature_config.get("history_depth", 30) + 10

        # 内部状态
        self._model = None
        self._scaler = None
        self._trained = False
        self._bar_count = 0
        self._history_buffer = []
        self._probability_threshold = 0.55
        self.warmup_period = self._warmup + self.train_window

    def on_start(self):
        if not self.is_restored:
            self._init_model()
            self._init_scaler()
        self.subscribe("AAPL")

    def on_resume(self):
        """热启动恢复时调用"""
        pass

    def _init_model(self):
        """初始化模型"""
        if self.model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            self._model = LogisticRegression(max_iter=1000, **self.model_params)
        elif self.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                self._model = XGBClassifier(
                    max_depth=self.model_params.get("max_depth", 5),
                    n_estimators=self.model_params.get("n_estimators", 100),
                    use_label_encoder=False,
                    eval_metric="logloss",
                    verbosity=0,
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                self._model = GradientBoostingClassifier(
                    max_depth=self.model_params.get("max_depth", 5),
                    n_estimators=self.model_params.get("n_estimators", 100),
                )
        elif self.model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
                self._model = LGBMClassifier(
                    num_leaves=self.model_params.get("num_leaves", 31),
                    n_estimators=self.model_params.get("n_estimators", 100),
                    verbose=-1,
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                self._model = GradientBoostingClassifier(
                    n_estimators=self.model_params.get("n_estimators", 100),
                )
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            self._model = RandomForestClassifier(
                n_estimators=self.model_params.get("n_estimators", 100),
            )
        else:
            from sklearn.linear_model import LogisticRegression
            self._model = LogisticRegression(max_iter=1000)

    def _init_scaler(self):
        """初始化标准化器"""
        try:
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
        except ImportError:
            self._scaler = None

    def _train_model(self, X_train, y_train):
        """训练模型"""
        if self._scaler is not None:
            X_train = self._scaler.fit_transform(X_train)

        try:
            self._model.fit(X_train, y_train)
            self._trained = True
        except Exception as e:
            print(f"[ML] 模型训练失败: {e}")

    def on_bar(self, bar):
        self._bar_count += 1

        closes = self.get_history(
            count=self.train_window + self._warmup,
            symbol=bar.symbol,
            field="close"
        )
        volumes = self.get_history(
            count=self.train_window + self._warmup,
            symbol=bar.symbol,
            field="volume"
        )

        if len(closes) < self.train_window + self._warmup:
            return

        n = len(closes)
        temp_df = pd.DataFrame({
            "close": closes,
            "volume": volumes if volumes is not None else [10000] * n,
            "open": closes,
            "high": closes,
            "low": closes,
        })

        if not self._trained or (self._bar_count % self.rolling_step == 0):
            X, y = self._feature_fn(temp_df, self.feature_config, mode="training")

            valid_mask = ~np.isnan(y)
            X_valid = X[valid_mask][-self.train_window:]
            y_valid = y[valid_mask][-self.train_window:]

            if len(X_valid) >= 20 and len(np.unique(y_valid)) >= 2:
                self._train_model(X_valid, y_valid)

        if self._trained:
            X_infer = self._feature_fn(temp_df, self.feature_config, mode="inference")
            if self._scaler is not None:
                X_infer = self._scaler.transform(X_infer)

            try:
                proba = self._model.predict_proba(X_infer)[0]
                signal = proba[1] if len(proba) > 1 else proba[0]
            except Exception:
                signal = 0.5

            position = self.get_position(bar.symbol)

            if signal > self._probability_threshold and position == 0:
                self.buy(symbol=bar.symbol, quantity=100)
            elif signal < (1 - self._probability_threshold) and position > 0:
                self.sell(symbol=bar.symbol, quantity=position)


class WalkForwardPyTorchStrategy(Strategy):
    """Walk-forward PyTorch 策略（模块级类，支持 pickle）

    工厂通过类属性 _cfg_* 注入配置，__init__() 无参数，
    以兼容 AKQuant 的 run_backtest(strategy=cls) 调用方式。
    """

    # 类级配置（由工厂注入）
    _cfg_feature_config = None
    _cfg_train_window = 50
    _cfg_test_window = 20
    _cfg_rolling_step = 10
    _cfg_hidden_dim = 32
    _cfg_num_layers = 2
    _cfg_epochs = 20
    _cfg_lr = 0.001
    _cfg_batch_size = 64
    _cfg_device = "cpu"
    _cfg_feature_fn = None
    _cfg_network_cls = None

    def __init__(self):
        self.feature_config = self.__class__._cfg_feature_config or DEFAULT_FEATURE_CONFIGS["basic"]
        self.train_window = self.__class__._cfg_train_window
        self.test_window = self.__class__._cfg_test_window
        self.rolling_step = self.__class__._cfg_rolling_step
        self.hidden_dim = self.__class__._cfg_hidden_dim
        self.num_layers = self.__class__._cfg_num_layers
        self.epochs = self.__class__._cfg_epochs
        self.lr = self.__class__._cfg_lr
        self.batch_size = self.__class__._cfg_batch_size
        self.device = self.__class__._cfg_device
        self._feature_fn = self.__class__._cfg_feature_fn or prepare_features
        self._network_cls = self.__class__._cfg_network_cls
        self._warmup = self.feature_config.get("history_depth", 30) + 10

        # 内部状态
        self._model = None
        self._optimizer = None
        self._trained = False
        self._bar_count = 0
        self._probability_threshold = 0.55
        self.warmup_period = self._warmup + self.train_window

    def on_start(self):
        if not self.is_restored:
            self._init_model()
        self.subscribe("AAPL")

    def on_resume(self):
        """热启动恢复"""
        pass

    def _init_model(self):
        """初始化 PyTorch 模型"""
        try:
            import torch
            import torch.nn as nn

            feature_names = self.feature_config.get("features", ["ret1", "ret2"])
            feature_dim = len(feature_names)

            if self._network_cls is not None:
                self._model = self._network_cls(
                    input_dim=feature_dim,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                ).to(self.device)
            else:
                self._model = _DefaultLSTM(
                    input_dim=feature_dim,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                ).to(self.device)

            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
            self._criterion = nn.BCELoss()

        except ImportError:
            print("[ML] PyTorch 未安装，LSTM 策略不可用。请安装: pip install torch")
            self._model = None

    def _train_model(self, X_train, y_train):
        """训练 PyTorch 模型"""
        import torch

        if self._model is None:
            return

        self._model.train()

        seq_len = min(10, len(X_train) - 1)
        if seq_len < 2:
            return

        X_seq = []
        y_seq = []
        for i in range(seq_len, len(X_train)):
            X_seq.append(X_train[i - seq_len:i])
            y_seq.append(y_train[i])

        if not X_seq:
            return

        X_tensor = torch.FloatTensor(np.array(X_seq)).to(self.device)
        y_tensor = torch.FloatTensor(np.array(y_seq)).unsqueeze(1).to(self.device)

        for epoch in range(self.epochs):
            self._optimizer.zero_grad()
            output = self._model(X_tensor)
            loss = self._criterion(output, y_tensor)
            loss.backward()
            self._optimizer.step()

        self._trained = True

    def on_bar(self, bar):
        self._bar_count += 1

        closes = self.get_history(
            count=self.train_window + self._warmup,
            symbol=bar.symbol,
            field="close"
        )
        volumes = self.get_history(
            count=self.train_window + self._warmup,
            symbol=bar.symbol,
            field="volume"
        )

        if len(closes) < self.train_window + self._warmup:
            return

        n = len(closes)
        temp_df = pd.DataFrame({
            "close": closes,
            "volume": volumes if volumes is not None else [10000] * n,
            "open": closes,
            "high": closes,
            "low": closes,
        })

        if not self._trained or (self._bar_count % self.rolling_step == 0):
            X, y = self._feature_fn(temp_df, self.feature_config, mode="training")

            valid_mask = ~np.isnan(y)
            X_valid = X[valid_mask][-self.train_window:]
            y_valid = y[valid_mask][-self.train_window:]

            if len(X_valid) >= 20 and len(np.unique(y_valid)) >= 2:
                self._train_model(X_valid, y_valid)

        if self._trained and self._model is not None:
            import torch

            X_infer = self._feature_fn(temp_df, self.feature_config, mode="inference")

            seq_len = min(10, len(X_infer))
            if seq_len < 2:
                X_seq = np.tile(X_infer, (10, 1))[-10:]
            else:
                X_seq = X_infer[-seq_len:]

            X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)

            self._model.eval()
            with torch.no_grad():
                signal = self._model(X_tensor).item()

            position = self.get_position(bar.symbol)

            if signal > self._probability_threshold and position == 0:
                self.buy(symbol=bar.symbol, quantity=100)
            elif signal < (1 - self._probability_threshold) and position > 0:
                self.sell(symbol=bar.symbol, quantity=position)


# =========================
# 默认 LSTM 模型（模块级，支持 pickle）
# =========================
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None


if _TORCH_AVAILABLE:
    class _DefaultLSTM(nn.Module):
        """默认 LSTM 模型（模块级类，支持 pickle）"""

        def __init__(self, input_dim, hidden_dim, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            out = self.fc(lstm_out[:, -1, :])
            return self.sigmoid(out)
else:
    _DefaultLSTM = None


class MLStrategyFactory:
    """ML 策略工厂：创建 Walk-forward ML 策略类

    通过向模块级策略类注入类属性配置，使其可被 pickle 序列化。
    """

    @staticmethod
    def create_sklearn_strategy(
        model_type: str,
        feature_config: Dict,
        train_window: int = 50,
        test_window: int = 20,
        rolling_step: int = 10,
        use_pipeline: bool = True,
        model_params: Dict = None,
        feature_engineering_fn: Callable = None,
    ) -> type:
        """
        创建基于 SklearnAdapter 的 Walk-forward 策略类

        Args:
            model_type: "logistic_regression" | "xgboost" | "lightgbm" | "random_forest"
            feature_config: 特征工程配置
            train_window: 训练窗口大小
            test_window: 测试窗口大小
            rolling_step: 滚动步长
            use_pipeline: 是否使用 Pipeline 封装预处理
            model_params: 模型超参数
            feature_engineering_fn: 自定义特征工程函数

        Returns:
            Strategy 类
        """
        # 注入配置到类属性
        WalkForwardSklearnStrategy._cfg_model_type = model_type
        WalkForwardSklearnStrategy._cfg_feature_config = feature_config
        WalkForwardSklearnStrategy._cfg_train_window = train_window
        WalkForwardSklearnStrategy._cfg_test_window = test_window
        WalkForwardSklearnStrategy._cfg_rolling_step = rolling_step
        WalkForwardSklearnStrategy._cfg_model_params = model_params or {}
        WalkForwardSklearnStrategy._cfg_feature_fn = feature_engineering_fn or prepare_features

        return WalkForwardSklearnStrategy

    @staticmethod
    def create_pytorch_strategy(
        network_cls=None,
        feature_config: Dict = None,
        train_window: int = 50,
        test_window: int = 20,
        rolling_step: int = 10,
        hidden_dim: int = 32,
        num_layers: int = 2,
        epochs: int = 20,
        lr: float = 0.001,
        batch_size: int = 64,
        device: str = "cpu",
        feature_engineering_fn: Callable = None,
    ) -> type:
        """
        创建基于 PyTorchAdapter 的深度学习策略类

        Args:
            network_cls: nn.Module 子类（如果为 None，使用默认 LSTM）
            feature_config: 特征配置
            train_window: 训练窗口
            test_window: 测试窗口
            rolling_step: 滚动步长
            hidden_dim: LSTM 隐藏层维度
            num_layers: LSTM 层数
            epochs: 训练轮数
            lr: 学习率
            batch_size: 批大小
            device: 计算设备
            feature_engineering_fn: 自定义特征工程函数

        Returns:
            Strategy 类
        """
        # 注入配置到类属性
        WalkForwardPyTorchStrategy._cfg_feature_config = feature_config or DEFAULT_FEATURE_CONFIGS["basic"]
        WalkForwardPyTorchStrategy._cfg_train_window = train_window
        WalkForwardPyTorchStrategy._cfg_test_window = test_window
        WalkForwardPyTorchStrategy._cfg_rolling_step = rolling_step
        WalkForwardPyTorchStrategy._cfg_hidden_dim = hidden_dim
        WalkForwardPyTorchStrategy._cfg_num_layers = num_layers
        WalkForwardPyTorchStrategy._cfg_epochs = epochs
        WalkForwardPyTorchStrategy._cfg_lr = lr
        WalkForwardPyTorchStrategy._cfg_batch_size = batch_size
        WalkForwardPyTorchStrategy._cfg_device = device
        WalkForwardPyTorchStrategy._cfg_feature_fn = feature_engineering_fn or prepare_features
        WalkForwardPyTorchStrategy._cfg_network_cls = network_cls

        return WalkForwardPyTorchStrategy


# =========================
# 模型持久化
# =========================
def save_model(model: Any, name: str, metadata: Dict = None) -> str:
    """
    保存训练好的模型到磁盘

    Args:
        model: 模型对象
        name: 模型名称
        metadata: 元数据

    Returns:
        保存路径
    """
    model_path = ML_MODELS_DIR / f"{name}.pkl"
    meta_path = ML_MODELS_DIR / f"{name}_meta.json"

    # 保存模型
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # 保存元数据
    meta = {
        "name": name,
        "saved_at": datetime.now().isoformat(),
        "model_type": type(model).__name__,
    }
    if metadata:
        meta.update(metadata)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return str(model_path)


def load_model(name: str) -> Any:
    """
    从磁盘加载模型

    Args:
        name: 模型名称

    Returns:
        模型对象
    """
    model_path = ML_MODELS_DIR / f"{name}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


def list_saved_models() -> List[Dict]:
    """列出所有已保存的模型"""
    models = []
    for meta_path in ML_MODELS_DIR.glob("*_meta.json"):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            pkl_path = meta_path.with_name(meta["name"] + ".pkl")
            if pkl_path.exists():
                meta["file_size_kb"] = pkl_path.stat().st_size / 1024
                models.append(meta)
        except Exception:
            pass
    return models


def delete_model(name: str) -> bool:
    """删除已保存的模型"""
    pkl_path = ML_MODELS_DIR / f"{name}.pkl"
    meta_path = ML_MODELS_DIR / f"{name}_meta.json"

    deleted = False
    if pkl_path.exists():
        pkl_path.unlink()
        deleted = True
    if meta_path.exists():
        meta_path.unlink()
        deleted = True

    return deleted


# =========================
# ML 回测辅助函数
# =========================
def create_ml_strategy_instance(
    strategy_type: str,
    params: Dict,
    feature_engineering_fn: Callable = None,
) -> type:
    """
    根据 ML 策略类型和参数创建策略类

    Args:
        strategy_type: ML 策略类型 ID（如 "ml_xgboost"）
        params: 策略参数
        feature_engineering_fn: 自定义特征工程函数

    Returns:
        Strategy 类
    """
    template = ML_STRATEGY_TEMPLATES.get(strategy_type)
    if not template:
        raise ValueError(f"未知 ML 策略类型: {strategy_type}")

    model_type = template["model_type"]
    feature_config_name = params.get("feature_config", template["params"].get("feature_config", {}).get("default", "basic"))
    feature_config = DEFAULT_FEATURE_CONFIGS.get(feature_config_name, DEFAULT_FEATURE_CONFIGS["basic"])

    # 提取 walk-forward 参数
    train_window = params.get("train_window", 50)
    test_window = params.get("test_window", 20)
    rolling_step = params.get("rolling_step", 10)

    if model_type == "lstm":
        return MLStrategyFactory.create_pytorch_strategy(
            feature_config=feature_config,
            train_window=train_window,
            test_window=test_window,
            rolling_step=rolling_step,
            hidden_dim=params.get("hidden_dim", 32),
            num_layers=params.get("num_layers", 2),
            epochs=params.get("epochs", 20),
            lr=params.get("lr", 0.001),
            feature_engineering_fn=feature_engineering_fn,
        )
    else:
        # Sklearn 系列
        model_params = {}
        if model_type == "xgboost":
            model_params = {
                "max_depth": params.get("max_depth", 5),
                "n_estimators": params.get("n_estimators", 100),
            }
        elif model_type == "lightgbm":
            model_params = {
                "num_leaves": params.get("num_leaves", 31),
                "n_estimators": params.get("n_estimators", 100),
            }
        elif model_type == "random_forest":
            model_params = {
                "n_estimators": params.get("n_estimators", 100),
            }

        return MLStrategyFactory.create_sklearn_strategy(
            model_type=model_type,
            feature_config=feature_config,
            train_window=train_window,
            test_window=test_window,
            rolling_step=rolling_step,
            model_params=model_params,
            feature_engineering_fn=feature_engineering_fn,
        )


def get_ml_strategy_metrics_description(metrics: Dict, strategy_type: str) -> str:
    """
    生成 ML 策略指标的中文描述

    Args:
        metrics: 指标字典
        strategy_type: ML 策略类型

    Returns:
        中文描述字符串
    """
    template = ML_STRATEGY_TEMPLATES.get(strategy_type, {})
    strategy_name = template.get("name", strategy_type)

    total_return = metrics.get("total_return_pct", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    max_dd = metrics.get("max_drawdown_pct", 0)
    win_rate = metrics.get("win_rate", 0)
    trades = metrics.get("total_trades", 0)

    # 性能评估
    if total_return > 15:
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
## ML 策略绩效报告 - {strategy_name}

| 指标 | 数值 | 评估 |
|------|------|------|
| 总收益率 | {total_return:.2f}% | {perf} |
| 夏普比率 | {sharpe:.2f} | {sharpe_eval} |
| 最大回撤 | {max_dd:.2f}% | - |
| 胜率 | {win_rate:.1f}% | - |
| 总交易次数 | {trades} 次 | - |

**总结**: 该 ML 策略在回测期间{perf}，{sharpe_eval}，共执行了 {trades} 次交易。
"""


def should_trigger_ml(user_input: str) -> bool:
    """检查是否应该触发 ML 策略功能"""
    ml_keywords = [
        "机器学习", "训练模型", "滚动训练", "walk_forward", "walkforward",
        "热启动", "快照恢复", "checkpoint", "warm_start", "warmstart",
        "多策略", "模拟盘", "组合策略", "slot",
        "xgboost", "lightgbm", "randomforest", "lstm", "transformer",
        "特征工程", "pipeline", "交叉验证", "过拟合", "深度学习",
    ]
    user_input_lower = user_input.lower()
    return any(keyword in user_input_lower for keyword in ml_keywords)
