# TikZ UML 绘图素材集

> 用途：提供画论文 UML 图所需的核心骨架信息
> 包含：系统架构层次关系、类图要素（属性+方法）、时序图调用链

---

## 一、系统整体架构（用于画 Layer / Package 图）

```
┌─────────────────────────────────────────────────────────────┐
│                     Presentation Layer                       │
│   streamlit_app.py (Streamlit Web UI)                       │
│   ├─ chat_page()        对话页面                             │
│   ├─ market_page()      行情数据页                           │
│   ├─ kg_page()          知识图谱可视化页                      │
│   └─ resset_page()      锐思数据查询页                       │
├─────────────────────────────────────────────────────────────┤
│                      Agent Core Layer                        │
│   agent.py (FinRAGAgent)                                     │
│   ├─ run(question, history, top_k) → {answer, sources}     │
│   ├─ _build_system_prompt(history, memories)                │
│   ├─ _execute_tool(name, args) → str                       │
│   └─ tools: [rag_search, akshare_quote, resset_fetch,       │
│              kg_query, quant_strategy]                       │
├─────────────────────────────────────────────────────────────┤
│                   Retrieval & Memory Layer                   │
│   rag.py (RAGPipeline)          memory_manager.py (v2.0)    │
│   ├─ retrieve(query)            ├─ smart_retrieve(query)    │
│   ├─ generate_hyde(query)       ├─ hybrid_search(query)     │
│   ├─ rerank(query, docs)        ├─ mmr_rerank(items)        │
│   └─ generate(state)            ├─ should_retrieve(q)→bool  │
│                                └─ should_update(q,a)→bool  │
├─────────────────────────────────────────────────────────────┤
│                 Knowledge Graph Layer                        │
│   knowledge_graph.py (EntityExtractor, KGQueryEngine)        │
│   kg_builder/kg_writer.py (KGWriter)                         │
│   kg_crawler/scheduler.py (KGScheduler)                      │
│   kg_crawler/news_crawler.py (NewsCrawler)                   │
├─────────────────────────────────────────────────────────────┤
│                    Trigger System Layer                      │
│   trigger_system.py (TriggerManager)                         │
│   ├─ KeywordTrigger × 9 种                                   │
│   ├─ LLMTrigger (可选)                                       │
│   └─ analyze(question) → List[TriggerResult]                │
├─────────────────────────────────────────────────────────────┤
│                  External Data Sources                       │
│   api_sources.py (AKShare/Tushare)  resset_data.py (锐思)   │
│   elasticsearch_client.py           neo4j_graph.py          │
└─────────────────────────────────────────────────────────────┘
```

**TikZ 画法建议**: 用 `fit` 库把每个 Layer 包成矩形框，箭头表示依赖方向（上→下调用）

---

## 二、核心类图要素

### 2.1 FinRAGAgent（智能体核心）

```
class FinRAGAgent {
    # ===== 属性 =====
    - client: ChatOllama / OpenAI         # LLM 客户端
    - model: str                          # 模型名称 (qwen2.5:7b)
    - tools: List[ToolSchema]             # 工具定义列表 (4个)
    - memory_manager: MemoryManagerV2     # 记忆管理器
    - trigger_manager: TriggerManager     # 触发管理器
    - rag_pipeline: RAGPipeline           # RAG 检索管道
    - compliance_checker: ComplianceChecker # 合规检查

    # ===== 公开方法 =====
    + run(question: str, conversation_history: str = "",
         top_k: int = 3) -> Dict[str, Any]

    # ===== 私有方法 =====
    - _build_system_prompt(history, memories) -> str
    - _execute_tool(tool_name: str, args: Dict) -> str
    - _format_tool_results(results) -> str
}
```

**依赖关系**:
- `FinRAGAgent` ——uses──▶ `MemoryManagerV2`
- `FinRAGAgent` ——uses──▶ `TriggerManager`
- `FinRAGAgent` ——uses──▶ `RAGPipeline`
- `FinRAGAgent` ——uses──▶ `ComplianceChecker`

---

### 2.2 RAGPipeline（检索管道）

```
class RAGPipeline {
    # ===== 属性 =====
    - es_client: Elasticsearch           # ES 向量检索客户端
    - embedding_model: OllamaEmbeddings   # 嵌入模型 (nomic-embed-text)
    - reranker: CrossEncoder              # BGE-reranker-v2-m3
    - llm: ChatOllama                     # HyDE 生成用 LLM
    - index_name: str                     # ES 索引名
    - use_hyde: bool                      # 是否启用 HyDE
    - use_reranker: bool                  # 是否启用精排

    # ===== 公开方法 =====
    + retrieve(query: str, top_k: int = 5)
        -> List[Tuple[Document, float]]
    + generate(state: GraphState) -> Dict
    + add_documents(docs: List[Document]) -> None

    # ===== 私有方法 =====
    - _vector_search(query_emb, top_k) -> List[Document]
    - _hyde_generate(question) -> str
    - _rerank_and_filter(query, docs, top_k) -> List[Document]
    - _build_rag_prompt(context, question) -> str
}
```

**依赖关系**:
- `RAGPipeline` ——calls──▶ `Elasticsearch` (向量搜索)
- `RAGPipeline` ——calls──▶ `OllamaEmbeddings` (HyDE嵌入)
- `RAGPipeline` ——calls──▶ `CrossEncoder` (精排)

---

### 2.3 MemoryManagerV2（记忆系统 v2.0）

```
class MemoryManagerV2 {
    # ===== 属性 =====
    - bm25_scorer: BM25Scorer            # BM25 评分引擎
    - embedding_cache: EmbeddingCache    # LRU Embedding 缓存
    - memory_files: Dict[str, Path]      # 记忆文件映射
    - soul_path: Path                    # SOUL.md 路径
    - daily_dir: Path                    # 日志目录 (90天滚动)
    - half_life_days: int = 30           # 时间衰减半衰期

    # ===== 公开方法 =====
    + smart_retrieve(query: str, top_k: int = 5)
        -> List[MemoryResult]
    + store_memory(question: str, answer: str,
                   metadata: Dict = None) -> None
    + compact_memories() -> None         # 压缩过期日志
    + get_stats() -> Dict

    # ===== 私有方法（内部管道）=====
    - _hybrid_search(query, top_k) -> List[Tuple]
    - _bm25_score(query, docs) -> List[float]
    - _vector_similarity(query, docs) -> List[float]
    - _time_decay(days_ago) -> float
    - _mmr_rerank(items, lambda_=0.7, top_k=5) -> List
    - _should_retrieve(query) -> bool     # 三道防线触发
    - _should_update(question, answer) -> tuple  # 四层过滤
}
```

**内部组件关系**:
- `MemoryManagerV2` ——contains──▶ `BM25Scorer`
- `MemoryManagerV2` ——contains──▶ `EmbeddingCache`
- `BM25Scorer` ←(3:7加权)→ 向量相似度 → **混合分数**

---

### 2.4 EntityExtractor & KGQueryEngine（知识图谱）

```
class EntityExtractor {
    # ===== 属性 =====
    - llm: ChatOllama                    # LLM 抽取引擎
    - extraction_prompt: str             # 领域本体论 Prompt

    # ===== 公开方法 =====
    + extract(text: str)
        -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]
}

class KGWriter {
    # ===== 属性 =====
    - driver: Neo4jDriver                # Neo4j 连接驱动

    # ===== 公开方法 =====
    + write_company(data: Dict) -> None
    + write_product(data: Dict) -> None
    + write_foundry(data: Dict) -> None
    + write_material(data: Dict) -> None
    + write_relation(source, target,
            rel_type, source_type, target_type,
            properties=None) -> None
    + import_companies_batch(stocks) -> ImportStats
    + import_sectors_batch(sectors) -> ImportStats
    + get_statistics() -> KGStats
}

class KGQueryEngine {
    # ===== 属性 =====
    - driver: Neo4jDriver

    # ===== 公开方法 =====
    + query_supply_chain(company: str) -> Dict
    + query_impact_chain(start: str, end: str,
            max_hops: int = 3) -> List[Dict]
    + query_risk_assessment(entity: str) -> Dict
    + query_entity_relations(name: str) -> List[Dict]
    + fuzzy_match(pattern: str) -> List[str]
}
```

**本体论实体类型（9种）**:
`Company | Person | Sector | Product | Foundry | Material | Location | Event | Indicator`

**关键关系类型（18种，核心8种）**:
`DESIGNS | OUTSOURCES_TO | DEPENDS_ON | COMPETES_WITH | SUPPLIES | BELONGS_TO | AFFECTED_BY | LOCATED_IN`

---

### 2.5 TriggerManager（触发系统）

```
class TriggerManager {
    # ===== 属性 =====
    - triggers: Dict[TriggerType, BaseTrigger]  # 触发器字典
    - llm_trigger: LLMTrigger | None            # LLM 触发(可选)
    - use_llm: bool                             # 是否启用 LLM 增强

    # ===== 公开方法 =====
    + analyze(question: str) -> List[TriggerResult]

    # ===== 内置触发器类型 =====
    # TriggerType.FINANCE_AKSHARE  → KeywordTrigger
    # TriggerType.FINANCE_TUSHARE  → KeywordTrigger
    # TriggerType.KG               → KeywordTrigger
    # TriggerType.RESSET           → KeywordTrigger
    # TriggerType.QUANT            → KeywordTrigger
    # TriggerType.MEMORY           → KeywordTrigger
    # TriggerType.COMPLIANCE       → KeywordTrigger
    # TriggerType.HYDE             → KeywordTrigger
    # TriggerType.RERANKER         → KeywordTrigger
}

class BaseTrigger (ABC):              # 抽象基类
    + check(question: str) -> Optional[TriggerResult]

class KeywordTrigger(BaseTrigger):     # 关键词触发 (<1ms)
    - keywords: List[str]
    - trigger_type: TriggerType
    + check(question) -> Optional[TriggerResult]
    # 置信度 = min(0.5 + 0.1 * match_count, 1.0)

class LLMTrigger(BaseTrigger):         # LLM 语义触发 (~500ms)
    - llm: ChatOllama
    + check(question) -> Optional[TriggerResult]
```

---

### 2.6 KGScheduler & NewsCrawler（数据采集调度）

```
class KGScheduler {
    # ===== 属性 =====
    - tasks: Dict[str, TaskInfo]
    - running: bool
    - _thread: Thread
    - task_history: List[Dict]

    # ===== 公开方法 =====
    + register_task(name, func, schedule_type,
                    **kwargs) -> None
    + start(blocking=False) -> None
    + stop() -> None
    + run_task_now(name: str) -> None
    + get_status() -> Dict

    # ===== 预定义任务 =====
    # "crawl_news"      → task_crawl_news() [hourly]
    # "update_stocks"   → task_update_stocks() [daily @09:30]
    # "update_sectors"  → task_update_sectors() [daily @09:00]
}

class NewsCrawler {
    - cache_dir: Path                     # 新闻缓存目录
    - cache_expiry_hours: int = 6         # 缓存有效期

    + crawl_all(max_count_per_source=20,
                filter_relevant=True) -> List[NewsArticle]
    + crawl_financial_news(max_count=20) -> List[NewsArticle]
    + _filter_chip_relevant(news_list) -> List[NewsArticle]
    + _is_cached(url) -> bool
    + _load_cache(url) -> NewsArticle | None
    + _save_cache(url, article) -> None
}
```

---

## 三、核心时序图调用链

### 3.1 用户提问完整处理流程（主时序图）

```
User → Streamlit UI → FinRAGAgent.run(question)
  │
  ├──① smart_retrieve(question) ──→ MemoryManagerV2
  │     ├── should_retrieve(question)? → bool (三道防线)
  │     │     NO → return [] (跳过, ~0ms)
  │     │     YES ↓
  │     ├── hybrid_search(question)
  │     │     ├── BM25Scorer.batch_score() ──→ 词频评分
  │     │     ├── EmbeddingCache.get()/put() ──→ 向量相似度
  │     │     └── 加权融合 (0.3×BM25 + 0.7×Vec)
  │     ├── time_decay(days_ago) ──→ 衰减权重
  │     └── mmr_rerank(items) ──→ 去重排序
  │
  ├──② TriggerManager.analyze(question)
  │     ├── KeywordTrigger.check() × 9 (并行, <5ms)
  │     └── LLMTrigger.check()? (可选增强)
  │
  ├──③ _build_system_prompt(history, memories)
  │
  ├──④ LLM 第1轮推理 (temperature=0.3)
  │     └── 返回 tool_calls?
  │           YES ↓          NO ↓ (直接返回答案)
  │
  ├──⑤ _execute_tool(tool_name, args)  [对每个tool_call]
  │     │
  │     ├── "rag_search" → RAGPipeline.retrieve()
  │     │     ├── HyDE? → _hyde_generate(question) ──→ LLM
  │     │     ├── _vector_search(embedding) ──→ Elasticsearch
  │     │     ├── _rerank_and_filter() ──→ CrossEncoder
  │     │     └── generate(state) ──→ LLM (意图分支A/B/C)
  │     │
  │     ├── "akshare_quote" → AKShareSource.get_realtime_data()
  │     │
  │     ├── "resset_fetch" → RessetConnection.get_content_data()
  │     │
  │     ├── "kg_query" → KGQueryEngine.query_*()
  │     │     └── Cypher SQL ──→ Neo4j
  │     │
  │     └── "quant_strategy" → QuantEngine.backtest()
  │
  ├──⑥ LLM 第2轮生成答案 (temperature=0.7)
  │     输入: messages + tool_results
  │
  ├──⑦ ComplianceChecker.check(answer) → 合规审查
  │
  └──⑧ return {answer, sources, used_context, ...}
```

### 3.2 知识图谱构建流水线（子时序图）

```
Scheduler (每小时/手动触发)
  │
  ├── task_crawl_news()
  │     ├── NewsCrawler.crawl_all(filter_relevant=True)
  │     │     └── 过滤芯片/半导体相关新闻
  │     │
  │     └── FOR EACH news IN news_list:
  │           ├── EntityExtractor.extract(text)
  │           │     └── LLM.invoke(extraction_prompt) → JSON
  │           │           → {entities: [...], relations: [...]}
  │           │
  │           ├── FOR EACH entity:
  │           │     └── KGWriter.write_{type}(data)
  │           │           └── MERGE (e:{type} {name:$name}) SET e+=$props
  │           │
  │           └── FOR EACH relation:
  │                 └── KGWriter.write_relation(src, tgt, type)
  │                       MATCH (s{name:$src}) MATCH (t{name:$tgt})
  │                       MERGE (s)-[r:{type}]->(t) SET r+=$props
  │
  ├── task_update_stocks()  [每天 09:30]
  │     ├── AKShareSource.get_all_stocks() → ~5000只A股
  │     └── KGWriter.import_companies_batch(stocks[:100])
  │
  └── task_update_sectors()  [每天 09:00]
        ├── AKShareSource.get_industry_classification() → ~200行业
        └── KGWriter.import_sectors_batch(sectors)
```

### 3.3 混合检索详细子流程

```
query → MemoryManagerV2.smart_retrieve(query, top_k=5)
  │
  ├── should_retrieve(query)?
  │     防线1: 关键词匹配 (~30个, <1ms)
  │     防线2: 有实质内容 + 是疑问句
  │     防线3: 自我指涉检测 ("我的"/"我想"/...)
  │     └── 全部未命中 → return []  ✓ 快速短路
  │
  ├── hybrid_search(query):
  │     │
  │     ├── [分支A] BM25 评分
  │     │     tokenize_chinese_bigram(query)  // unigram+bigram
  │     │     FOR EACH doc:
  │     │       idf = log((N-df+0.5)/(df+0.5)+1)
  │     │       tf_norm = tf×(k1+1)/(tf+k1×(1-b+b×dl/avgdl))
  │     │       score += idf × tf_norm
  │     │
  │     ├── [分支B] 向量相似度
  │     │     query_emb = EmbeddingCache.get_or_compute(query)
  │     │     FOR EACH doc:
  │     │       chunk = best_chunk(doc, size=400)
  │     │       chunk_emb = EmbeddingCache.get_or_compute(chunk)
  │     │       vec_score = cosine(query_emb, chunk_emb)
  │     │
  │     └── fusion: hybrid = 0.3×norm(BM25) + 0.7×norm(Vec)
  │
  ├── time_decay():
  │     IF NOT evergreen:
  │       decay = exp(-ln(2)/30 × days_ago)  // T½=30天
  │       score *= decay
  │
  └── mmr_rerank(items, λ=0.7, k=top_k):
        WHILE len(selected) < k:
          FOR EACH item remaining:
            rel = item.decayed_score
            div = max(Jaccard(item, s) for s in selected)
            mmr = λ×rel - (1-λ)×div
          selected.add(best_mmr_item)
        return selected
```

---

## 四、TikZ 画图速查表

### 4.1 类图模板

```latex
% TikZ UML 类图基础配置
\usetikzlibrary{shapes.multipart, positioning, arrows.meta, fit, backgrounds}

% 类节点样式
\tikzset{
    class/.style={
        draw, rectangle split, rectangle split parts=3,
        font=\ttfamily\small, align=center,
        inner sep=2pt, minimum width=6cm
    },
    dependency/.style={-{Stealth}, thick},
    composition/.style={-{Diamond[open]}, thick},
    inheritance/.style->{Triangle[open], thick}
}

% 使用示例：
% \node[class] (agent) {
%   \textbf{FinRAGAgent}
%   \nodepart{second}
%   \begin{tabular}{l}- client: ChatOllama\\- tools: List[Tool]\end{tabular}
%   \nodepart{third}
%   \begin{tabular}{l}+ run(): Dict\\- \_execute\_tool()\end{tabular}
% };
```

### 4.2 时序图模板

```latex
\usetikzlibrary{positioning, arrows.meta, calc}

% 参与者样式
\tikzstyle{actor}=[rectangle, draw, minimum height=2em, minimum width=3em]
\tikzstyle{arrow}=[-{Stealth}, thick]
\tikzstyle{dashed_arrow}={[arrow, dashed]}
\tikzstyle{activation}=[rectangle, fill=gray!20]

% 时序图绘制模式:
% 1. 先画所有参与者 (竖线)
% 2. 从左到右画消息箭头
% 3. activation 矩形表示方法执行期间
% 4. return 用虚线
% 5. alt/opt 片段用方框+标签
```

### 4.3 架构分层图模板

```latex
\usetikzlibrary{fit, backgrounds, positioning}

% 分层框样式
\tikzstyle{layer}=[
    rectangle, rounded corners, draw=black!60,
    fill=#1!10, minimum width=12cm, minimum height=2cm,
    font=\bfseries, align=center
]

% 使用:
% \node[layer=blue] (pres) at (0,4) {Presentation Layer};
% \node[layer=green] (agent) at (0,2) {Agent Core};
% \node[layer=orange] (retrieval) at (0,0) {Retrieval Layer};
```

---

## 五、论文中建议画的图及对应素材位置

| 论文图号 | 推荐图表类型 | 对应本文档章节 |
|----------|-------------|---------------|
| Fig.X1 | **系统架构分层图** | 第一节（整体架构） |
| Fig.X2 | **Agent 类图** | §2.1 FinRAGAgent |
| Fig.X3 | **主时序图（问答全流程）** | §3.1 主流程 |
| Fig.X4 | **RAG Pipeline 时序图** | §3.1 中 rag_search 子调用 |
| Fig.X5 | **记忆系统类图** | §2.3 MemoryManagerV2 |
| Fig.X6 | **混合检索流程图** | §3.3 详细子流程 |
| Fig.X7 | **知识图谱本体论图** | §2.4 实体+关系类型 |
| Fig.X8 | **KG构建流水线时序图** | §3.2 构建流程 |
| Fig.X9 | **触发系统类图** | §2.5 TriggerManager |
