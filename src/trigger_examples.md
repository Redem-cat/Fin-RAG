# 触发系统使用示例

## 📋 .env 配置总览

所有关键词触发都通过 `.env` 文件配置：

```env
# =========================
# 🔹 意图分类关键词
# =========================
INTENT_KEYWORDS_INVESTMENT=股票,基金,ETF,投资,收益...
INTENT_KEYWORDS_POLICY=法律,法规,证监会,国务院...

# =========================
# 🔹 触发系统关键词
# =========================
LLM_TRIGGER_ENABLED=false
TRIGGER_KEYWORDS_HYDE=分析,比较,评估,建议...
TRIGGER_KEYWORDS_FINANCE_AKSHARE=股票,股价,基金,净值...
TRIGGER_KEYWORDS_FINANCE_TUSHARE=财务,财报,年报,季报...
TRIGGER_KEYWORDS_MEMORY=之前,上次,记得...
TRIGGER_KEYWORDS_COMPLIANCE=推荐,买,卖,投资...
TRIGGER_KEYWORDS_RERANKER=详细,具体,准确...

# =========================
# 🔹 金融数据关键词
# =========================
FINANCE_KEYWORDS_STOCK=股票,股价,涨停...
FINANCE_KEYWORDS_FUND=基金,净值,持仓...
FINANCE_KEYWORDS_INDEX=指数,沪深300,上证指数...
FINANCE_KEYWORDS_MACRO=利率,LPR,GDP,CPI...
```

---

## 🔄 触发流程

```
用户问题
    │
    ├─→ 意图分类器 (intent_classifier)
    │      ├─ INTENT_KEYWORDS_INVESTMENT
    │      └─ INTENT_KEYWORDS_POLICY
    │
    ├─→ 触发系统 (trigger_system)
    │      ├─ TRIGGER_KEYWORDS_HYDE
    │      ├─ TRIGGER_KEYWORDS_FINANCE_AKSHARE
    │      ├─ TRIGGER_KEYWORDS_FINANCE_TUSHARE
    │      ├─ TRIGGER_KEYWORDS_MEMORY
    │      ├─ TRIGGER_KEYWORDS_COMPLIANCE
    │      └─ TRIGGER_KEYWORDS_RERANKER
    │
    └─→ 金融触发器 (finance_trigger)
           ├─ FINANCE_KEYWORDS_STOCK
           ├─ FINANCE_KEYWORDS_FUND
           ├─ FINANCE_KEYWORDS_INDEX
           └─ FINANCE_KEYWORDS_MACRO
```

---

## 💬 示例输入输出

### 示例 1: 股票行情查询

**输入:**
```
"帮我查一下贵州茅台今天涨了多少"
```

**触发分析:**
```
[意图分类] INVESTMENT - 涉及投资领域(1个关键词)
[触发] finance_akshare: 匹配到关键词: 股票, 涨 (置信度: 0.70)
[触发] hyde: 匹配到关键词: 帮, 查 (置信度: 0.60)
[触发] compliance: 匹配到关键词: 涨 (置信度: 0.60)
```

**输出:**
```json
{
  "intent": "investment",
  "triggers": [
    {"trigger_type": "finance_akshare", "confidence": 0.70, "reason": "匹配到关键词: 股票, 涨"},
    {"trigger_type": "hyde", "confidence": 0.60, "reason": "匹配到关键词: 帮, 查"},
    {"trigger_type": "compliance", "confidence": 0.60, "reason": "匹配到关键词: 涨"}
  ],
  "answer": "贵州茅台今日涨幅约2.3%，最新股价1780元..."
}
```

---

### 示例 2: 基金财务分析

**输入:**
```
"帮我分析一下510300这只ETF的财务数据，包括净利润和ROE"
```

**触发分析:**
```
[意图分类] INVESTMENT - 涉及投资领域(3个关键词)
[触发] finance_akshare: 匹配到关键词: ETF (置信度: 0.80)
[触发] finance_tushare: 匹配到关键词: 净利润, ROE, 财务数据 (置信度: 0.90)
[触发] hyde: 匹配到关键词: 分析 (置信度: 0.70)
```

**输出:**
```json
{
  "intent": "investment",
  "triggers": [
    {"trigger_type": "finance_tushare", "confidence": 0.90, "reason": "匹配到关键词: 净利润, ROE"},
    {"trigger_type": "finance_akshare", "confidence": 0.80, "reason": "匹配到关键词: ETF"},
    {"trigger_type": "hyde", "confidence": 0.70, "reason": "匹配到关键词: 分析"}
  ],
  "answer": "510300（沪深300ETF）最新财务指标..."
}
```

---

### 示例 3: 政策合规咨询

**输入:**
```
"证监会新出台的《私募投资基金监督管理条例》有哪些主要内容？"
```

**触发分析:**
```
[意图分类] POLICY - 涉及政策法规(2个关键词)
[触发] compliance: 匹配到关键词: 条例 (置信度: 0.70)
```

**输出:**
```json
{
  "intent": "policy",
  "triggers": [
    {"trigger_type": "compliance", "confidence": 0.70, "reason": "匹配到关键词: 条例"}
  ],
  "answer": "《私募投资基金监督管理条例》的主要内容包括..."
}
```

---

### 示例 4: 混合意图问题

**输入:**
```
"我想投资基金，有什么政策需要注意？同时帮我看看最近哪个行业表现好"
```

**触发分析:**
```
[意图分类] MIXED - 同时涉及投资(2个关键词)和政策(1个关键词)
[触发] finance_akshare: 匹配到关键词: 基金, 行业 (置信度: 0.80)
[触发] hyde: 匹配到关键词: 哪个, 看看 (置信度: 0.60)
```

**输出:**
```json
{
  "intent": "mixed",
  "triggers": [
    {"trigger_type": "finance_akshare", "confidence": 0.80, "reason": "匹配到关键词: 基金, 行业"},
    {"trigger_type": "hyde", "confidence": 0.60, "reason": "匹配到关键词: 哪个, 看看"}
  ],
  "answer": "关于投资基金的政策需要注意...最近表现最好的行业是..."
}
```

---

### 示例 5: 参考历史对话

**输入:**
```
"之前提到的那个基金，现在还能买吗？"
```

**触发分析:**
```
[意图分类] INVESTMENT - 涉及投资领域(1个关键词)
[触发] memory: 匹配到关键词: 之前, 提到的 (置信度: 0.80)
[触发] finance_akshare: 匹配到关键词: 基金 (置信度: 0.70)
```

**输出:**
```json
{
  "intent": "investment",
  "triggers": [
    {"trigger_type": "memory", "confidence": 0.80, "reason": "匹配到关键词: 之前, 提到的"},
    {"trigger_type": "finance_akshare", "confidence": 0.70, "reason": "匹配到关键词: 基金"}
  ],
  "answer": "根据之前的对话，您提到的是XX基金。当前行情下..."
}
```

---

### 示例 6: 需要精排的复杂问题

**输入:**
```
"请详细介绍一下沪深300指数的编制方法，专业一点"
```

**触发分析:**
```
[意图分类] INVESTMENT - 涉及投资领域(2个关键词)
[触发] finance_akshare: 匹配到关键词: 沪深300, 指数 (置信度: 0.80)
[触发] reranker: 匹配到关键词: 详细, 专业, 介绍 (置信度: 0.90)
[触发] hyde: 匹配到关键词: 介绍 (置信度: 0.70)
```

**输出:**
```json
{
  "intent": "investment",
  "triggers": [
    {"trigger_type": "reranker", "confidence": 0.90, "reason": "匹配到关键词: 详细, 专业"},
    {"trigger_type": "finance_akshare", "confidence": 0.80, "reason": "匹配到关键词: 沪深300, 指数"},
    {"trigger_type": "hyde", "confidence": 0.70, "reason": "匹配到关键词: 介绍"}
  ],
  "answer": "沪深300指数是由中证指数有限公司编制..."
}
```

---

## 🧪 测试脚本

```python
# test_triggers.py
from src.trigger_system import get_trigger_manager
from src.intent_classifier import get_intent_classifier

def test_question(question: str):
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print('='*60)
    
    # 意图分类
    classifier = get_intent_classifier()
    intent, reason = classifier.classify(question)
    print(f"[意图] {intent.value} - {reason}")
    
    # 触发分析
    manager = get_trigger_manager()
    results = manager.analyze(question)
    print(f"[触发] 共 {len(results)} 个模块被触发:")
    for r in results:
        print(f"  - {r.trigger_type}: {r.reason} (置信度: {r.confidence:.2f})")
    
    return {"intent": intent.value, "triggers": results}

# 测试
if __name__ == "__main__":
    questions = [
        "帮我查一下贵州茅台今天涨了多少",
        "分析510300的财务数据和ROE",
        "证监会新出台的政策有哪些",
        "之前提到的基金现在还能买吗",
        "请详细介绍沪深300指数的编制方法",
    ]
    
    for q in questions:
        test_question(q)
```

---

## ⚙️ 自定义关键词

编辑 `.env` 文件即可自定义关键词：

```env
# 添加新的触发关键词
TRIGGER_KEYWORDS_FINANCE_AKSHARE=股票,股价,基金,净值,ETF,指数,大盘,行情,涨跌,行业,转债,可转债

# 添加意图关键词
INTENT_KEYWORDS_INVESTMENT=股票,基金,ETF,投资,收益,回报,涨幅,跌幅,转债,可转债,打新
```
