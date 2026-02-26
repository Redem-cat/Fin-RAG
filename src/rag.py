# RAG architecture using LangChain, Ollama and Elasticsearch
# Modified by Redem-cat

import os
from datetime import datetime, timedelta
from pathlib import Path
import json

import numpy as np
from dotenv import load_dotenv

from langchain_elasticsearch import ElasticsearchStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# =========================
# 🔹 检索日志管理器
# =========================
class RetrievalLogger:
    """检索日志管理器：记录检索详情并定期清理"""

    def __init__(self, log_dir: str = None, max_log_files: int = 10):
        if log_dir is None:
            log_dir = base_path / "retrieval_logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.max_log_files = max_log_files
        self.session_count = 0

    def log(self, question: str, retrieved_docs: list, answer: str, used_context: bool):
        """记录一次检索的详细信息"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"retrieval_{timestamp}.json"

        # 准备日志数据
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "retrieved_docs": [],
            "answer": answer,
            "used_context": used_context
        }

        # 处理检索到的文档
        for doc in retrieved_docs:
            if isinstance(doc, tuple):
                document, score = doc
                log_data["retrieved_docs"].append({
                    "content": document.page_content[:500],  # 只保存前500字符
                    "metadata": document.metadata,
                    "raw_score": score
                })
            else:
                log_data["retrieved_docs"].append({
                    "content": doc.page_content[:500],
                    "metadata": doc.metadata,
                    "raw_score": None
                })

        # 写入日志文件
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        # 每次记录后清理，保持最多 max_log_files 个
        self.clean_old_logs()

    def clean_old_logs(self):
        """清理旧的日志文件，保留最近的 max_log_files 个"""
        log_files = list(self.log_dir.glob("retrieval_*.json"))
        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        if len(log_files) > self.max_log_files:
            for old_file in log_files[self.max_log_files:]:
                old_file.unlink()




# =========================
# 🔹 配置和初始化
# =========================
base_path = Path(__file__).parent.parent.resolve()
retrieval_logger = RetrievalLogger(max_log_files=10)

# 加载环境变量
dotenv_path = Path(base_path / "elastic-start-local/.env")
if not dotenv_path.is_file():
    print("Error: it seems Elasticsearch has not been installed")
    print("using start-local, please execute the following command:")
    print("curl -fsSL https://elastic.co/start-local | sh")
    exit(1)
    
load_dotenv(dotenv_path=dotenv_path)
index_name = "rag-langchain"

# Embeddings
embeddings = OllamaEmbeddings(
    model="my-bge-m3",
)

# LLM
llm = ChatOllama(model="my-qwen25", temperature=0.0000000001)


# =========================
# 🔹 对话历史管理器（混合检索 + 分层存储）
# =========================
class MemoryManager:
    """对话历史管理器：混合检索 + 分层存储"""
    
    def __init__(self, memory_dir: str = None, compaction_interval: int = 10):
        if memory_dir is None:
            memory_dir = base_path / "memory"
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        # 文件路径
        self.soul_file = self.memory_dir / "SOUL.md"
        self.agents_file = self.memory_dir / "AGENTS.md"
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.daily_dir = self.memory_dir / "daily"
        self.daily_dir.mkdir(exist_ok=True)
        
        # compaction 设置
        self.compaction_interval = compaction_interval
        self.conversation_count = 0
        
        # 初始化必要文件
        self._ensure_files()
    
    def _ensure_files(self):
        """确保必要文件存在"""
        if not self.soul_file.exists():
            self.soul_file.write_text("# AI 灵魂配置\n", encoding="utf-8")
        if not self.agents_file.exists():
            self.agents_file.write_text("# Agent 规范\n", encoding="utf-8")
        if not self.memory_file.exists():
            self.memory_file.write_text("# 长期记忆\n\n## 用户偏好\n\n## 核心事实\n\n## 关键决策\n\n", encoding="utf-8")
    
    def _get_today_file(self) -> Path:
        """获取今日日志文件"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.daily_dir / f"{today}.md"
    
    def _extract_keywords(self, text: str) -> set:
        """简单关键词提取（基于字符分割）"""
        # 移除标点，分割成词
        import re
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text)
        # 过滤短词
        keywords = {w.lower() for w in words if len(w) >= 2}
        return keywords
    
    def _chunk_text(self, text: str, chunk_size: int = 400) -> list:
        """将文本分割成 chunks"""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            current_chunk.append(line)
            current_size += len(line)
            if current_size >= chunk_size:
                chunks.append('\n'.join(current_chunk))
                # 保留最后一行作为 overlap
                current_chunk = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                current_size = 0
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _keyword_filter(self, query: str, files: list) -> list:
        """阶段1: 关键词快速过滤"""
        query_keywords = self._extract_keywords(query)
        if not query_keywords:
            return files
        
        candidates = []
        for file_path in files:
            if not file_path.exists():
                continue
            content = file_path.read_text(encoding="utf-8").lower()
            file_keywords = self._extract_keywords(content)
            
            # 检查是否有交集
            if query_keywords & file_keywords:
                candidates.append(file_path)
        
        return candidates
    
    def _vector_rerank(self, query: str, files: list, threshold: float = 0.3, top_k: int = 3) -> list:
        """阶段2: 向量重排 + 阈值过滤"""
        if not files:
            return []
        
        query_embedding = embeddings.embed_query(query)
        
        scored_files = []
        for file_path in files:
            content = file_path.read_text(encoding="utf-8")
            if not content.strip():
                continue
            
            # 分 chunk
            chunks = self._chunk_text(content)
            chunk_scores = []
            
            for chunk in chunks:
                chunk_embedding = embeddings.embed_query(chunk)
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding) + 1e-8
                )
                chunk_scores.append((similarity, chunk))
            
            if chunk_scores:
                # 取最高相似度
                best_score = max(chunk_scores, key=lambda x: x[0])
                scored_files.append((best_score[0], file_path.name, best_score[1]))
        
        # 排序并过滤
        scored_files.sort(key=lambda x: x[0], reverse=True)
        results = [(score, name, chunk) for score, name, chunk in scored_files if score >= threshold]
        
        return results[:top_k]
    
    def add_message(self, role: str, content: str):
        """添加对话消息到当日日志"""
        today_file = self._get_today_file()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 初始化文件
        if not today_file.exists():
            today_file.write_text(f"# {datetime.now().strftime('%Y-%m-%d')} 对话日志\n\n", encoding="utf-8")
        
        content_md = today_file.read_text(encoding="utf-8")
        content_md += f"- **{timestamp} {role}**: {content}\n\n"
        
        today_file.write_text(content_md, encoding="utf-8")
        
        # 计数
        self.conversation_count += 1
        
        # 检查是否需要 compaction
        if self.conversation_count >= self.compaction_interval:
            self.compact()
            self.conversation_count = 0
    
    def retrieve_relevant_history(self, query: str, top_k: int = 3, threshold: float = 0.3) -> str:
        """混合检索: 关键词过滤 + 向量重排"""
        # 收集要搜索的文件
        search_files = [self.memory_file, self.soul_file, self.agents_file]
        
        # 添加最近 N 天的日志（最多7天）
        days_to_search = 7
        for i in range(days_to_search):
            day = datetime.now() - timedelta(days=i)
            day_file = self.daily_dir / f"{day.strftime('%Y-%m-%d')}.md"
            search_files.append(day_file)
        
        # 阶段1: 关键词过滤
        candidates = self._keyword_filter(query, search_files)
        
        # 阶段2: 向量重排
        results = self._vector_rerank(query, candidates, threshold, top_k)
        
        if not results:
            return ""
        
        # 格式化输出
        formatted = []
        for score, name, chunk in results:
            formatted.append(f"<memory-snippet file=\"{name}\" score=\"{score:.3f}\">\n{chunk}\n</memory-snippet>")
        
        return "\n\n".join(formatted)
    
    def compact(self):
        """定期将重要信息压缩到长期记忆"""
        # 读取最近几天的日志
        recent_content = []
        for i in range(3):  # 最近3天
            day = datetime.now() - timedelta(days=i)
            day_file = self.daily_dir / f"{day.strftime('%Y-%m-%d')}.md"
            if day_file.exists():
                content = day_file.read_text(encoding="utf-8")
                if content.strip():
                    recent_content.append(content)
        
        if not recent_content:
            return
        
        # 读取现有记忆
        memory_content = self.memory_file.read_text(encoding="utf-8")
        
        # 简单追加策略：保留最近对话的摘要
        memory_content += f"\n### {datetime.now().strftime('%Y-%m-%d')} 摘要\n"
        memory_content += "（近期对话已整合）\n"
        
        self.memory_file.write_text(memory_content, encoding="utf-8")
        print("🔄 Memory compaction 完成")
    
    def get_soul(self) -> str:
        """获取灵魂配置"""
        return self.soul_file.read_text(encoding="utf-8") if self.soul_file.exists() else ""
    
    def get_agents(self) -> str:
        """获取 Agent 规范"""
        return self.agents_file.read_text(encoding="utf-8") if self.agents_file.exists() else ""
    
    def clear_history(self):
        """清空对话历史"""
        # 清空每日日志
        for f in self.daily_dir.glob("*.md"):
            f.unlink()
        
        # 重置长期记忆（保留结构）
        self.memory_file.write_text("# 长期记忆\n\n## 用户偏好\n\n## 核心事实\n\n## 关键决策\n\n", encoding="utf-8")
        self.conversation_count = 0
        print("🗑️ 对话历史已清空")

# =========================
# 🔹 初始化组件
# =========================
memory_manager = MemoryManager()

vector_db = ElasticsearchStore(
    es_url=os.getenv('ES_LOCAL_URL'),
    embedding=embeddings,
    index_name=index_name
)

# 定义 Prompt（包含对话历史）
prompt_template = PromptTemplate.from_template(
    template="""Previous conversation:
{history}

[DOCUMENT FRAGMENTS START]
{context}
[DOCUMENT FRAGMENTS END]

[USER QUESTION START]
{question}
[USER QUESTION END]

Instructions:
1. The text above in [DOCUMENT FRAGMENTS START]...[DOCUMENT FRAGMENTS END] contains retrieved document fragments for reference only.
2. The text above in [USER QUESTION START]...[USER QUESTION END] is the user's question.
3. Answer the user's question based on the document fragments when relevant, otherwise use your own knowledge.
4. CRITICAL: Answer in the SAME LANGUAGE as the user's question, NOT the language of the document fragments.
5. Write only three sentences."""
)

# 定义状态
class State(TypedDict):
    question: str
    top_k: int
    context: List[Document]
    history: str
    answer: str

# 定义应用步骤
def retrieve(state: State):
    """检索相关文档和对话历史"""
    # 检索文档（带相似度分数），使用传入的 top_k
    top_k = state.get("top_k", 3)
    retrieved_docs_with_scores = vector_db.similarity_search_with_score(state["question"], k=top_k)
    
    # 检索相关对话历史
    relevant_history = memory_manager.retrieve_relevant_history(state["question"], top_k=3)
    
    return {"context": retrieved_docs_with_scores, "history": relevant_history}


def generate(state: State):
    """生成答案"""
    # 阈值设置：文档相似度阈值和整体意图判断阈值
    DOC_SIMILARITY_THRESHOLD = 0.75
    INTENT_SIMILARITY_THRESHOLD = 0.7

    # 处理带分数的文档（(doc, score) 元组列表），过滤低相似度
    context_docs = []
    all_scores = []

    # 先归一化分数
    context_items = state.get("context", [])
    if context_items:
        # 提取分数并归一化
        scored_docs = []
        for item in context_items:
            if isinstance(item, tuple):
                doc, score = item
                scored_docs.append((doc, score))

        if scored_docs:
            raw_scores = [s for _, s in scored_docs]
            all_scores = raw_scores
            max_s, min_s = max(raw_scores), min(raw_scores)

            # 判断是距离还是相似度：距离通常 > 1，相似度通常 <= 1
            is_distance = max_s > 1.0

            for doc, score in scored_docs:
                if is_distance:
                    # 距离转换为相似度: similarity = 1 / (1 + distance)
                    normalized = 1.0 / (1.0 + score)
                else:
                    # 已经是相似度，直接使用，不进行归一化
                    normalized = score

                # 记录归一化后的相似度
                doc.metadata["similarity"] = normalized

                if normalized >= DOC_SIMILARITY_THRESHOLD:
                    context_docs.append(doc)

    # 意图判断：计算最高相似度
    max_similarity = 0
    if all_scores:
        max_raw = max(all_scores)
        min_raw = min(all_scores)
        is_distance = max_raw > 1.0
        if is_distance:
            max_similarity = 1.0 / (1.0 + min_raw)  # 最小距离对应最高相似度
        else:
            max_similarity = max_raw  # 直接使用原始相似度

    # 判断是否使用检索结果
    use_retrieved_context = max_similarity >= INTENT_SIMILARITY_THRESHOLD

    if use_retrieved_context and context_docs:
        docs_content = "\n\n".join(doc.page_content for doc in context_docs)
        context_info = f"（使用了 {len(context_docs)} 个相关文档片段，最高相似度: {max_similarity:.3f}）"
    else:
        docs_content = ""
        if max_similarity < INTENT_SIMILARITY_THRESHOLD:
            context_info = f"（检索到的文档相关性不足（最高相似度: {max_similarity:.3f}），不使用检索结果）"
        else:
            context_info = "（未找到足够相关的文档片段）"

    history = state.get("history", "") or "No previous conversation."

    # 根据是否使用上下文调整提示词
    if use_retrieved_context and docs_content:
        prompt = prompt_template.format(
            question=state["question"],
            context=docs_content,
            history=history
        )
    else:
        # 不使用检索结果，直接基于模型知识回答
        no_context_prompt = PromptTemplate.from_template(
            template="""Previous conversation:
{history}

[USER QUESTION START]
{question}
[USER QUESTION END]

Instructions:
1. The retrieved documents are not relevant to this question.
2. Answer based on your own knowledge.
3. CRITICAL: Answer in the SAME LANGUAGE as the user's question.
4. Write only three sentences."""
        )
        prompt = no_context_prompt.format(
            question=state["question"],
            history=history
        )

    response = llm.invoke(prompt)

    # 记录到检索日志
    retrieval_logger.log(
        question=state["question"],
        retrieved_docs=context_items,
        answer=response.content,
        used_context=use_retrieved_context
    )

    # 在答案中添加上下文信息说明（仅用于调试，可移除）
    final_answer = response.content
    # final_answer = f"{response.content}\n\n{context_info}"  # 取消注释可显示调试信息

    return {"answer": final_answer}


# 编译应用
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# =========================
# 🔹 对话函数（供 Streamlit 调用）
# =========================
def ask_question(question: str, top_k: int = 3):
    """
    问答函数，供 Web 界面调用

    Args:
        question: 用户问题
        top_k: 返回的文档数量

    Returns:
        dict: 包含 answer, source, question, used_context
    """
    # 调用图，传递 top_k 参数
    response = graph.invoke({"question": question, "top_k": top_k})

    # 保存对话历史到 Markdown
    memory_manager.add_message("用户", question)
    memory_manager.add_message("AI", response["answer"])

    # 整理结果（处理带分数的文档）
    sources = []
    context_items = response.get("context", [])

    # 提取所有分数
    all_scores = []
    for item in context_items:
        if isinstance(item, tuple):
            _, score = item
            all_scores.append(score)

    # 判断是距离还是相似度
    has_scores = bool(all_scores)
    is_distance = False
    if has_scores:
        max_score_val = max(all_scores)
        is_distance = max_score_val > 1.0

    # 使用与 generate 函数相同的阈值
    DOC_SIMILARITY_THRESHOLD = 0.75

    # 过滤并处理文档片段
    for item in context_items:
        if isinstance(item, tuple):
            doc, score = item
            # 判断是距离还是相似度
            if has_scores:
                if is_distance:
                    # 距离转换为相似度: similarity = 1 / (1 + distance)
                    normalized_score = 1.0 / (1.0 + score)
                else:
                    # 已经是相似度，直接使用
                    normalized_score = score
            else:
                normalized_score = 0.5

            # 只添加达到文档相似度阈值的文档
            if normalized_score >= DOC_SIMILARITY_THRESHOLD:
                sources.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page_label", "unknown"),
                    "similarity": normalized_score
                })

    # 根据 sources 是否为空判断是否使用了检索结果
    used_context = len(sources) > 0

    return {
        "question": question,
        "answer": response["answer"],
        "source": sources,
        "used_context": used_context
    }


def clear_conversation_history():
    """清空对话历史"""
    memory_manager.clear_history()


def create_rag_chain():
    """创建并返回 RAG 链，供评估器使用

    Returns:
        compiled graph: 编译好的 LangGraph
    """
    return graph

# =========================
# 🔹 主函数（命令行测试）
# =========================
if __name__ == "__main__":
    # 测试用，请修改问题后运行
    pass
