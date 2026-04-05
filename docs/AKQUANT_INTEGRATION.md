# AKQuant 量化策略集成说明

## 概述

本项目已成功将 AKQuant 量化策略框架与 FinRAG 系统有机整合，实现：
- **自然语言驱动的量化策略创建**
- **一键回测与可视化报告**
- **Web 界面无缝集成**

---

## 集成架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web 界面                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                 │
│  │   CHAT   │   │   QUANT  │   │   EVAL   │                 │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘                 │
│       │              │              │                         │
│       ▼              ▼              ▼                         │
│  ┌─────────────────────────────────────────┐                  │
│  │         RAG + LLM + 触发系统              │                  │
│  └─────────────────────────────────────────┘                  │
│                         │                                     │
└─────────────────────────┼─────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              量化模块 (src/quantitative.py)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  双均线策略  │  │   RSI 策略   │  │  MACD 策略   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │  布林带策略  │  │  自定义策略  │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   AKQuant 回测引擎                           │
│  run_backtest() → 回测结果 → generate_report() → HTML报告   │
└─────────────────────────────────────────────────────────────┘
```

---

## 新增文件

| 文件 | 说明 |
|------|------|
| `src/quantitative.py` | 量化策略模块（封装 AKQuant） |
| `src/trigger_examples.md` | 触发系统使用示例 |

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/streamlit_app.py` | 添加 QUANT 页面 |
| `src/trigger_system.py` | 添加量化触发类型 |
| `elastic-start-local/.env` | 添加 AKQuant 配置 |
| `requirements.txt` | 添加 streamlit, pandas, plotly |

---

## 功能特性

### 1. 预定义策略模板

| 策略 | 说明 | 参数 |
|------|------|------|
| **双均线** | 金叉买入，死叉卖出 | fast_window, slow_window |
| **RSI** | 超卖买入，超买卖出 | rsi_period, oversold, overbought |
| **MACD** | MACD 金叉死叉信号 | fast_period, slow_period, signal_period |
| **布林带** | 价格触及下轨买入，上轨卖出 | window, num_std |

### 2. Web 界面功能

- **策略选择**: 下拉菜单选择预定义策略
- **参数调整**: 滑块调整策略参数
- **数据源选择**: AKShare 实时数据 / 模拟数据
- **快速模板**: 一键加载预设配置
- **绩效指标**: 总收益、夏普比率、最大回撤、胜率
- **可视化报告**: K线图 + 交易标记 +  Equity 曲线

### 3. 触发系统集成

当用户输入包含以下关键词时，自动触发量化功能：

```
量化, 回测, 策略, 均线, rsi, macd, 布林带, 金叉, 死叉,
交易, 收益率, 夏普比率, backtest, strategy
```

---

## 使用方式

### 方式一：Web 界面

1. 启动应用：`streamlit run src/streamlit_app.py`
2. 导航到 **QUANT** 页面
3. 选择策略和参数
4. 点击 **RUN BACKTEST**

### 方式二：Python API

```python
from src.quantitative import (
    run_backtest,
    generate_report,
    get_available_strategies
)

# 获取可用策略
strategies = get_available_strategies()
print(strategies)

# 运行回测
result = run_backtest(
    data=df,
    strategy_type="dual_ma",
    strategy_params={"fast_window": 10, "slow_window": 30},
    symbol="sh600000",
    start_date="20210101",
    end_date="20231231",
    initial_cash=100000.0
)

# 生成报告
if result["success"]:
    report_path = generate_report(
        result,
        strategy_name="双均线策略",
        symbol="sh600000",
        market_data=df
    )
    print(f"报告已生成: {report_path}")
```

### 方式三：触发系统自动识别

在 CHAT 页面提问：
```
帮我用双均线策略回测一下茅台股票2022年的表现
```

系统会自动：
1. 识别量化触发类型
2. 解析策略参数
3. 获取数据
4. 运行回测
5. 返回结果

---

## 依赖安装

```bash
# 安装项目依赖
pip install -r requirements.txt

# 安装 AKQuant（开发模式）
cd akquant-main/python
pip install -e .

# 或者
pip install -e ./akquant-main/python
```

---

## .env 配置

```env
# AKQuant 量化配置
AKQUANT_PATH=./akquant-main/python
QUANT_INITIAL_CASH=100000
QUANT_COMMISSION_RATE=0.0003
DEFAULT_FAST_WINDOW=10
DEFAULT_SLOW_WINDOW=30

# 量化触发关键词
TRIGGER_KEYWORDS_QUANT=量化,回测,策略,均线,rsi,macd,布林带,金叉,死叉
```

---

## 注意事项

1. **AKQuant 安装**: 必须先安装 AKQuant 才能使用量化功能
2. **数据依赖**: AKShare 用于获取真实股票数据
3. **性能**: 复杂的回测可能需要较长时间
4. **风险提示**: 回测结果仅供参考，不构成投资建议

---

## 未来扩展

- [ ] 支持更多策略类型
- [ ] 参数自动优化
- [ ] 实盘交易接口
- [ ] 组合策略回测
- [ ] 风险分析模块
