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
- Docker Desktop (用于运行 Elasticsearch)
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

### 3. 配置环境变量（可选）

如需使用外部 API（如 OpenAI），可创建 `.env` 文件：

```bash
# 创建 .env 文件
cat > .env << 'EOF'
# OpenAI API（可选，如使用 Ollama 则无需配置）
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1

# Elasticsearch（使用本地 Docker 时无需修改）
ES_LOCAL_URL=http://localhost:9200
EOF
```

---

### 4. 安装 Python 依赖

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

---

## 快速开始

### 1. 启动 Ollama 服务（如未运行）

```bash
# 启动 Ollama 后台服务
ollama serve
```

### 2. 启动 Elasticsearch

```bash
# 确保 Docker 已启动
docker start es-langchain
```

### 3. 导入文档到知识库（如需更新数据）

```bash
python scripts/index_documents.py --dir "data/01_金融法规"
```

该脚本会：
- 读取 `data/` 目录下的 Markdown 文件
- 使用 Embedding 模型进行向量化
- 存入 Elasticsearch

### 4. 启动 Web 界面

```bash
streamlit run src/streamlit_app.py
```

访问 `http://localhost:8501` 即可使用。

### 5. 使用评估功能

在 Streamlit 侧边栏选择「评估」页面，可对 RAG 系统进行批量评估。

---

## 项目结构

```
langchain-ollama-elasticsearch/
├── data/                    # 文档目录
│   ├── 01_金融法规/         # 金融法规
│   ├── 案例/                # 投资案例
│   └── ...
├── src/
│   ├── rag.py              # RAG 核心逻辑
│   ├── streamlit_app.py     # Web 界面
│   ├── finance_data.py     # 金融数据 (AKShare)
│   ├── finance_trigger.py  # 金融数据自动触发
│   ├── intent_classifier.py # 意图分类器
│   └── compliance_checker.py # 合规审查
├── scripts/
│   ├── index_documents.py  # 文档索引
│   └── ...
├── retrieval_logs/         # 检索日志
├── memory/                  # 对话历史
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
| 金融数据 | AKShare |
| 评估 | RAGAS |
| Web UI | Streamlit |

---

## Copyright

Copyright (C) 2026 by [Redem-cat](https://github.com/Redem-cat).

This project is derived from the original work by Enrico Zimuel (Apache License).
