# FinRAG-Advisor: 智能投顾与合规双模 RAG 系统

基于多模态知识图谱增强的智能投顾与合规审查系统，一个面向金融机构的 RAG 知识库系统 built with [LangChain](https://www.langchain.com/)、[Ollama](https://ollama.com) 和 [Elasticsearch](https://github.com/elastic/elasticsearch)。

该系统不仅支持客户与员工的自然语言问答，而且深入融合了投资建议生成与合规风险自动校验，实现智能服务 + 自动合规审查一体化。

![RAG architecture](./img/RAG_Elasticsearch.png)

## 核心特性

### 双 RAG 子系统
- **投资建议生成**：基于检索增强的智能问答
- **合规验证**：对投资建议进行实时审计，降低幻觉风险

### 多模态知识图谱增强
- 年报、报表、图片通过 OCR/多模态大模型结构化解析
- 转化为知识三元组，构建动态金融知识图谱

### 实时监管政策
- 接入央行、证监会等 RSS 源
- 自动抓取最新政策并更新知识库

### 混合检索
- **语义检索**：BGE-M3-Financial 向量模型
- **关键词检索**：Elasticsearch
- **知识图谱检索**：Neo4j 图数据库
- **RRF 融合**：三种检索结果融合排序

### 智能文档处理
- PDF 布局识别 (pdfplumber)
- 表格结构识别与还原
- 层级感知动态分块算法

### 系统化评估
- RAGAS 框架评估
- 五大核心指标：faithfulness、context precision、answer relevance、response time、compliance coverage
- 可视化仪表盘展示

---

## 环境要求

- Python 3.10+
- Docker Desktop (用于运行 Elasticsearch 和 Neo4j)
- Ollama (用于本地大模型)

---

## 安装说明

### 1. 安装 Ollama

本地运行需要安装 [Ollama](https://ollama.com/download)。

#### 拉取所需模型

```bash
# 拉取 Embedding 模型（用于文档向量化）
ollama pull my-bge-m3

# 拉取对话模型（用于生成回答）
ollama pull my-qwen25
```

> **注意**：首次拉取模型需要下载较大文件，请确保网络稳定。

#### 验证模型安装

```bash
# 查看已安装的模型
ollama list
```

---

### 2. 安装并启动 Elasticsearch

本项目使用 Docker 运行 Elasticsearch。

#### 方式一：使用 Docker Compose（推荐）

```bash
# 创建 docker-compose.yml 文件
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: es-langchain
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
      - "9300:9300"
    volumes:
      - es-data:/usr/share/elasticsearch/data
    restart: unless-stopped

volumes:
  es-data:
    driver: local
EOF

# 启动 Elasticsearch
docker-compose up -d
```

#### 方式二：直接使用 Docker

```bash
# 拉取并启动 Elasticsearch
docker run -d \
  --name es-langchain \
  -p 9200:9200 \
  -p 9300:9300 \
  -e discovery.type=single-node \
  -e xpack.security.enabled=false \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  -v es-data:/usr/share/elasticsearch/data \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

#### 验证 Elasticsearch 启动

```bash
# 等待约 30 秒后检查
curl http://localhost:9200
```

正常启动后会返回类似以下 JSON：

```json
{
  "name" : "...",
  "cluster_name" : "docker-cluster",
  "cluster_uuid" : "...",
  "version" : {
    "number" : "8.11.0"
  },
  "tagline" : "You Know, for Search"
}
```

#### 启动/停止 Elasticsearch

```bash
# 停止
docker stop es-langchain

# 启动
docker start es-langchain

# 查看日志
docker logs -f es-langchain
```

---

### 3. 安装并启动 Neo4j（知识图谱，可选）

知识图谱功能需要 Neo4j 数据库。

#### 使用 Docker 启动

```bash
# 拉取并启动 Neo4j
docker run -d \
  --name neo4j-langchain \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5.15-community
```

#### 验证 Neo4j 启动

```bash
# 访问 http://localhost:7474
# 使用用户名: neo4j，密码: password 登录
```

> **注意**：如果不需要知识图谱功能，可以跳过此步骤。

---

### 4. 配置环境变量

创建 `.env` 文件配置必要的环境变量：

```bash
# 创建 .env 文件
cat > .env << 'EOF'
# ===========================================
# LLM 配置（必填）
# ===========================================
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=my-bge-m3
CHAT_MODEL=my-qwen25

# ===========================================
# Elasticsearch 配置
# ===========================================
ES_LOCAL_URL=http://localhost:9200

# ===========================================
# Neo4j 配置（可选）
# ===========================================
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# ===========================================
# 其他配置
# ===========================================
RAG_SYSTEM_NAME=FinRAG-Advisor
ES_INDEX_NAME=financial_docs
EOF
```

---

### 5. 构建并安装 AKQuant（必选）

AKQuant 是项目的核心量化框架，需要从源码构建：

```bash
# 进入 AKQuant 目录
cd akquant-main

# 使用 uv 构建并安装（推荐，比 pip 快）
uv pip install -e .

# 或者使用 pip
pip install -e .
```

> **注意**：构建 AKQuant 需要 Rust 工具链。如果没有安装，请先安装 [rustup](https://rustup.rs/)。

#### 验证安装

```python
import akquant
print(akquant.__version__)  # 应该输出 0.2.2
```

### 6. 构建并安装 AKQuant（必选）

AKQuant 是项目的核心量化框架，需要从源码构建：

```bash
# 进入 AKQuant 目录
cd akquant-main

# 使用 uv 构建并安装（推荐，比 pip 快）
uv pip install -e .

# 或者使用 pip
pip install -e .
```

> **注意**：构建 AKQuant 需要 Rust 工具链。如果没有安装，请先安装 [rustup](https://rustup.rs/)。

#### 验证安装

```python
import akquant
print(akquant.__version__)  # 应该输出 0.2.2
```

---

#### 主要依赖说明

| 依赖包 | 用途 |
|--------|------|
| `langchain`, `langchain-ollama` | LLM 和 Embedding 集成 |
| `langchain-elasticsearch` | 向量数据库连接 |
| `elasticsearch` | Elasticsearch 客户端 |
| `sentence-transformers` | 文本向量化 |
| `akshare` | 金融数据获取 |
| `neo4j` | 知识图谱数据库 |
| `streamlit` | Web 界面框架 |
| `plotly` | 图表可视化 |
| `ragas` | RAG 评估框架 |
| `pandas` | 数据处理 |

---

## 快速开始

### 启动流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        启动流程                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 启动 Ollama 服务                                             │
│     └─> ollama serve                                             │
│                                                                 │
│  2. 启动 Elasticsearch (Docker)                                  │
│     └─> docker start es-langchain                               │
│                                                                 │
│  3. 启动 Neo4j (Docker，可选)                                      │
│     └─> docker start neo4j-langchain                            │
│                                                                 │
│  4. 启动 Web 界面                                                 │
│     └─> streamlit run src/streamlit_app.py                      │
│                                                                 │
│  5. 访问 http://localhost:8501                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 详细步骤

```bash
# 1. 启动 Ollama 后台服务
ollama serve

# 2. 启动 Elasticsearch
docker start es-langchain

# 3. (可选) 启动 Neo4j
docker start neo4j-langchain

# 4. 启动 Web 界面
streamlit run src/streamlit_app.py
```

访问 `http://localhost:8501` 即可使用。

---

## 项目结构

```
langchain-ollama-elasticsearch/
│
├── data/                    # 文档目录
│   ├── 01_金融法规/          # 金融法规文档
│   └── ...
│
├── src/                     # 源代码目录
│   ├── streamlit_app.py     # Streamlit 主应用（多页面）
│   ├── rag.py               # RAG 检索与生成逻辑
│   ├── knowledge_graph.py   # 知识图谱 (Neo4j)
│   ├── intent_classifier.py # 意图分类
│   ├── quantitative.py      # 量化回测引擎
│   ├── finance_data.py      # 金融数据 (AKShare)
│   ├── trigger_system.py    # 触发系统
│   ├── evaluator.py         # RAG 评估器 (RAGAS)
│   ├── reporter.py          # 评估报告
│   ├── auth.py              # 用户认证
│   └── config.py            # 配置管理
│
├── scripts/                 # 脚本目录
│   └── index_documents.py   # 文档索引脚本
│
├── elastic-start-local/     # ES 本地配置
├── akquant-main/           # AKQuant 量化框架
│
├── requirements.txt         # Python 依赖
├── .env                    # 环境变量配置
└── docker-compose.yml       # Docker 服务编排
```

---

## 系统架构图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           FinRAG-Advisor 系统架构                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                         用户界面层 (Streamlit)                   │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │    │
│  │  │   CHAT  │ │  MARKET │ │    KG   │ │  QUANT  │ │   EVAL  │    │    │
│  │  │  对话页 │ │  行情页  │ │ 知识图谱 │ │ 量化回测 │ │  评估页 │   │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                    │
│  ┌─────────────────────────────────┴─────────────────────────────────┐  │
│  │                      业务逻辑层 (LangChain/LangGraph)              │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │  │
│  │  │意图分类器 │ │ RAG 检索 │ │合规审查器 │ │回测引擎  │               │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│          │                                    │                         │
│  ┌───────┴────────────────────────────────────┴───────────────┐         │
│  │                         数据层                             │       │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │       │
│  │  │Elasticsearch│  │    Neo4j    │  │   AKShare   │         │       │
│  │  │  向量存储    │  │  知识图谱   │  │  金融数据    │         │       │
│  │  │   :9200     │  │   :7687     │  │            │           │       │
│  │  └─────────────┘  └─────────────┘  └─────────────┘           │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                    │                                     │
│  ┌─────────────────────────────────┴─────────────────────────────┐       │
│  │                      模型层 (Ollama 本地)                         │       │
│  │  ┌─────────────────────┐    ┌─────────────────────┐           │       │
│  │  │  my-bge-m3           │    │  my-qwen25          │           │       │
│  │  │  (Embedding)         │    │  (对话生成)          │           │       │
│  │  └─────────────────────┘    └─────────────────────┘           │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 功能模块说明

| 模块 | 说明 |
|------|------|
| **CHAT** | 基于 RAG 的智能问答，支持检索增强和对话历史 |
| **MARKET** | K线图、均线、MACD、RSI 等技术指标可视化 |
| **KG** | Neo4j 知识图谱可视化与查询 |
| **QUANT** | AKQuant 量化回测，支持基准对比和流式进度 |
| **EVAL** | RAGAS 框架评估：Faithfulness、Context Precision 等 |

---

## 技术栈

| 组件 | 技术 | 说明 |
|:-----|:-----|:-----|
| LLM | Ollama (Qwen, Llama) | 本地大模型推理 |
| Embedding | BGE-M3 | 文本向量化 |
| 向量数据库 | Elasticsearch 8.11 | 文档存储与检索 |
| 知识图谱 | Neo4j 5.15 | 图数据库 |
| 金融数据 | AKShare | 免费金融数据接口 |
| 量化框架 | AKQuant | 策略回测 |
| RAG 框架 | LangChain + LangGraph | 应用框架 |
| 评估框架 | RAGAS | RAG 系统评估 |
| Web UI | Streamlit | 前端界面 |
| 可视化 | Plotly | 图表展示 |

---

## Copyright

Copyright (C) 2026 by [Redem-cat](https://github.com/Redem-cat).

This project is derived from the original work by Enrico Zimuel (Apache License).
