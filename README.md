# FinRAG-Advisor: 智能投顾与合规双模 RAG 系统

基于多模态知识图谱增强的智能投顾与合规审查系统，一个面向金融机构的 RAG 知识库系统 built with [LangChain](https://www.langchain.com/)、[Ollama](https://ollama.com) 和 [Elasticsearch](https://github.com/elastic/elasticsearch)。

该系统不仅支持客户与员工的自然语言问答，而且深入融合了投资建议生成与合规风险自动校验，实现智能服务 + 自动合规审查一体化。

![RAG architecture](./img/RAG_Elasticsearch.png)

## 核心特性

### 🔄 双 RAG 子系统
- **投资建议生成**：基于检索增强的智能问答
- **合规验证**：对投资建议进行实时审计，降低幻觉风险 多模态知识图

### 📊谱增强
- 年报、报表、图片通过 OCR/多模态大模型结构化解析
- 转化为知识三元组，构建动态金融知识图谱

### 📰 实时监管政策
- 接入央行、证监会等 RSS 源
- 自动抓取最新政策并更新知识库

### 🔍 混合检索
- **语义检索**：BGE-M3-Financial 向量模型
- **关键词检索**：Elasticsearch
- **知识图谱检索**：Neo4j 图数据库
- **RRF 融合**：三种检索结果融合排序

### 📄 智能文档处理
- PDF 布局识别 (pdfplumber)
- 表格结构识别与还原
- 层级感知动态分块算法

### 📈 系统化评估
- RAGAS 框架评估
- 五大核心指标：faithfulness、context precision、answer relevance、response time、compliance coverage
- 可视化仪表盘展示

---

## 安装说明

### 1. 安装 Ollama

本地运行需要安装 [Ollama](https://ollama.com/download)：

```bash
# 拉取 embedding 模型
ollama pull my-bge-m3

# 拉取对话模型
ollama pull my-qwen25
```

### 2. 安装 Elasticsearch

```bash
curl -fsSL https://elastic.co/start-local | sh
```

Elasticsearch 将安装在 `elastic-start-local` 目录，服务运行在 `localhost:9200`。

### 3. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

---

## 快速开始

### 1. 导入文档到知识库

```bash
python src/store_data.py
```

该脚本会：
- 读取 `data/` 目录下的 PDF 文件
- 使用 Docling 进行文档解析
- 分块处理后存入 Elasticsearch

### 2. 启动 Web 界面

```bash
streamlit run src/streamlit_app.py
```

访问 `http://localhost:8501` 即可使用。

### 3. 使用评估功能

在 Streamlit 侧边栏选择「评估」页面，可对 RAG 系统进行批量评估。

---

## 项目结构

```
langchain-ollama-elasticsearch/
├── data/                    # PDF 文档目录
├── src/
│   ├── rag.py              # RAG 核心逻辑
│   ├── store_data.py       # 文档导入
│   ├── streamlit_app.py    # Web 界面
│   ├── evaluator.py        # RAGAS 评估器
│   └── reporter.py         # HTML 报告生成
├── retrieval_logs/         # 检索日志
├── memory/                 # 对话历史存储
└── requirements.txt        # Python 依赖
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | Ollama (Qwen, Llama) |
| Embedding | BGE-M3 |
| 向量数据库 | Elasticsearch |
| 框架 | LangChain + LangGraph |
| 文档解析 | Docling |
| 评估 | RAGAS |
| Web UI | Streamlit |

---

## Copyright

Copyright (C) 2026 by [Redem-cat](https://github.com/Redem-cat).

This project is derived from the original work by Enrico Zimuel (Apache License).
