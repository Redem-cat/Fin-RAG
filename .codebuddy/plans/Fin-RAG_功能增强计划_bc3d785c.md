---
name: Fin-RAG 功能增强计划
overview: 将 Fin-RAG-new 的 Streamlit 页面、停用词处理和缓存机制整合到当前基于 Elasticsearch + Ollama 的 Fin-RAG 项目中
todos:
  - id: copy-stopwords
    content: 复制 Fin-RAG-new/中文停用词库.txt 到项目根目录
    status: completed
  - id: modify-rag-py
    content: 修改 src/rag.py：添加停用词处理类和缓存管理类，改造 retrieve 函数
    status: completed
    dependencies:
      - copy-stopwords
  - id: modify-store-data
    content: 修改 src/store_data.py：添加缓存检测逻辑
    status: completed
  - id: create-streamlit
    content: 新建 streamlit_app.py：创建 Web 界面，包含对话、侧边栏配置、聊天历史
    status: completed
  - id: test-connection
    content: 测试运行 Streamlit 应用
    status: completed
    dependencies:
      - modify-rag-py
      - create-streamlit
---

## 用户需求

保留当前 Elasticsearch + Ollama 本地部署架构，添加 Streamlit Web 界面，同时集成中文停用词处理机制和文档分块/向量索引缓存机制。

## 核心功能

1. Streamlit Web 界面：仿照 Fin-RAG-new 设计，包含对话界面、侧边栏配置、聊天历史、参考文档展示
2. 中文停用词处理：在查询前对用户输入进行分词和停用词过滤，提升检索精度
3. 缓存机制：缓存文档分块和向量索引，避免重复计算，加快启动速度

## 技术栈选择

- 保留 Elasticsearch + Ollama 本地部署架构
- 使用 Streamlit 构建 Web 界面
- 使用 jieba 进行中文分词
- 使用 pickle/joblib 实现缓存机制
- 使用 langchain 现有组件

## 实现方案

### 架构设计

```
┌─────────────────────────────────────────────────┐
│                  Streamlit Web                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │  对话界面   │  │  侧边栏配置 │  │聊天历史 │ │
│  └─────────────┘  └─────────────┘  └─────────┘ │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│              RAG System (rag.py)                │
│  ┌──────────────┐  ┌──────────────────────────┐ │
│  │  停用词处理  │──│  Elasticsearch 检索      │ │
│  │ (jieba分词) │  │  + LangGraph 生成答案     │ │
│  └──────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│              缓存层 (Cache)                      │
│  ┌────────────┐  ┌────────────┐  ┌───────────┐  │
│  │分块缓存    │  │向量索引缓存│  │对话历史缓存│  │
│  │(pickle)   │  │(joblib)    │  │(Session)   │  │
│  └────────────┘  └────────────┘  └───────────┘  │
└─────────────────────────────────────────────────┘
```

### 关键改动

1. 修改 rag.py：添加 StopWords 类、CacheManager 类，改造 retrieve 函数
2. 修改 store_data.py：添加缓存检测逻辑
3. 新建 streamlit_app.py：参考 Fin-RAG-new 界面设计

### 性能优化

- 向量索引构建后缓存，避免重复计算
- Session 级别的对话历史缓存
- 分块结果缓存加速冷启动

## Agent 扩展

无额外的 Agent 扩展需要使用，当前任务使用 Python 原生技术栈即可完成。