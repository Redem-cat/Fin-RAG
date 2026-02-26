---
name: 修复RAG系统未完成的问题
overview: 修复TopK默认值不统一和硬编码问题
todos:
  - id: fix-default-top-k
    content: 修改 streamlit_app.py 第108行，将 search_top_k 默认值从 8 改为 3
    status: completed
  - id: fix-rag-top-k-param
    content: 修改 rag.py，在 State 添加 top_k 字段，ask_question 传递 top_k 到 graph，retrieve 使用传入的 top_k 值
    status: completed
  - id: fix-similarity-calculation
    content: 修复 rag.py 相似度计算逻辑，将 Elasticsearch 距离转换为相似度，确保 0.7 阈值过滤正常工作
    status: completed
---

## 用户需求

修复以下问题：

1. streamlit_app.py 第108行 search_top_k 初始值应为 3（当前为 8）
2. rag.py 第289行 retrieve 函数中 k=3 硬编码，未使用 ask_question 传入的 top_k 参数
3. 确保相似度阈值 0.7 过滤正常工作（当前 Elasticsearch 返回的是距离而非相似度，需要正确转换）

## 核心功能

- 修正搜索参数默认值
- 将 top_k 参数正确传递到检索函数
- 修复相似度计算逻辑（距离转相似度）

## 技术方案

### 问题分析

1. **streamlit_app.py 第108行**: 默认值硬编码为 8，应改为 3
2. **rag.py 第289行**: `k=3` 硬编码在 retrieve 函数中，ask_question 接收 top_k 参数但未传递给检索函数
3. **相似度过滤**: Elasticsearch 的 `similarity_search_with_score` 返回的是距离（距离越小越相似），当前代码直接当作相似度处理，导致过滤逻辑错误

### 修复方案

1. 修改 streamlit_app.py 第108行：`search_top_k = 3`
2. 修改 rag.py：

- 在 State 中添加 top_k 字段
- ask_question 调用 graph 时传入 top_k
- retrieve 函数从 state 获取 top_k 而非硬编码
- 修复相似度计算：相似度 = 1 / (1 + 距离)
- 保持 0.7 阈值过滤逻辑正确工作