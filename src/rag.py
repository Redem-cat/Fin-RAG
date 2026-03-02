# RAG architecture using LangChain, Ollama and Elasticsearch
# Modified by Redem-cat

import os
from datetime import datetime, timedelta
from pathlib import Path
import json
import re
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv

from langchain_elasticsearch import ElasticsearchStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# 尝试导入 reranker（可选依赖）
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    print("警告: sentence-transformers 未安装，精排功能不可用")

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

# =========================
# 🔹 延迟加载组件
# =========================
# 这些组件在首次使用时才初始化，避免启动慢
embeddings = None
llm = None
vector_db = None
compliance_checker = None
COMPLIANCE_ENABLED = False
memory_manager = None
retrieval_logger = None

_loaded = {
    "embeddings": False,
    "llm": False,
    "vector_db": False,
    "memory": False,
}


def warm_up():
    """空函数，保持接口兼容"""
    pass


def _get_embeddings():
    """延迟加载 Embeddings"""
    global embeddings, _loaded
    if _loaded["embeddings"]:
        return embeddings
    print("正在加载 Embeddings 模型...")
    embeddings = OllamaEmbeddings(model="my-bge-m3")
    _loaded["embeddings"] = True
    print("[OK] Embeddings 加载完成")
    return embeddings


def _get_llm():
    """延迟加载 LLM"""
    global llm, _loaded
    if _loaded["llm"]:
        return llm
    print("正在加载 LLM 模型...")
    llm = ChatOllama(model="my-qwen25", temperature=0.0000000001)
    _loaded["llm"] = True
    print("[OK] LLM 加载完成")
    return llm


def _get_vector_db():
    """延迟加载 Vector DB"""
    global vector_db, _loaded
    if _loaded["vector_db"]:
        return vector_db
    print("正在连接 Elasticsearch...")
    emb = _get_embeddings()
    vector_db = ElasticsearchStore(
        es_url=os.getenv('ES_LOCAL_URL'),
        embedding=emb,
        index_name=index_name
    )
    _loaded["vector_db"] = True
    print("[OK] Elasticsearch 连接完成")
    return vector_db


def _get_memory_manager():
    """延迟加载 Memory Manager"""
    global memory_manager, _loaded
    if _loaded["memory"]:
        return memory_manager
    memory_manager = MemoryManager()
    _loaded["memory"] = True
    return memory_manager


# 合规审查器单例
_get_compliance = type('_GetCompliance', (), {'instance': None})()


# =========================
# 🔹 精排模型（延迟加载）
# =========================
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
reranker = None
reranker_loaded = False
RERANKER_ENABLED = True  # 默认启用精排
RERANK_CANDIDATE_COUNT = 10  # 召回候选数量（精排前）


def _get_reranker():
    """延迟加载精排模型"""
    global reranker, reranker_loaded
    if reranker_loaded:
        return reranker
    
    if not RERANKER_AVAILABLE:
        reranker_loaded = True
        return None
    
    try:
        print(f"正在加载精排模型: {RERANKER_MODEL} ...")
        reranker = CrossEncoder(RERANKER_MODEL, max_length=1024)
        print("[OK] 精排模型加载成功")
    except Exception as e:
        print(f"[WARN] 精排模型加载失败: {e}")
        reranker = None
    
    reranker_loaded = True
    return reranker


def rerank_documents(query: str, documents: List[Document], top_k: int = 3) -> List[tuple]:
    """
    使用 BGE-reranker 对文档进行重排（延迟加载）
    
    Args:
        query: 用户查询
        documents: 待重排的文档列表
        top_k: 返回重排后的 top_k 个文档
        
    Returns:
        List[tuple]: 重排后的文档列表 (doc, score)
    """
    # 延迟加载 reranker
    current_reranker = _get_reranker()
    
    if not current_reranker or not documents:
        # 如果没有 reranker，直接返回原始文档
        return [(doc, 1.0) for doc in documents[:top_k]]
    
    # 准备 query-doc 对
    pairs = [[query, doc.page_content] for doc in documents]
    
    # 获取重排分数
    try:
        scores = current_reranker.predict(pairs)
        
        # 将文档和分数配对
        doc_scores = list(zip(documents, scores))
        
        # 按分数降序排序
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回 top_k
        return doc_scores[:top_k]
    except Exception as e:
        print(f"精排预测失败: {e}")
        return [(doc, 1.0) for doc in documents[:top_k]]


# LLM
llm = ChatOllama(model="my-qwen25", temperature=0.0000000001)

# 合规审查器（使用 DeepSeek API）
try:
    compliance_checker = ComplianceChecker()
    COMPLIANCE_ENABLED = True
except Exception as e:
    print(f"警告: 合规审查器初始化失败: {e}")
    COMPLIANCE_ENABLED = False
    compliance_checker = None


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
    """检索相关文档和对话历史（包含精排）"""
    top_k = state.get("top_k", 3)
    question = state["question"]
    
    # 延迟加载组件
    vdb = _get_vector_db()
    memory = _get_memory_manager()
    
    # 步骤1: 向量检索召回候选文档
    # 召回更多候选，用于精排
    candidate_count = max(RERANK_CANDIDATE_COUNT, top_k * 3)
    retrieved_docs_with_scores = vdb.similarity_search_with_score(question, k=candidate_count)
    
    # 提取文档（不区分原始分数）
    retrieved_docs = []
    for item in retrieved_docs_with_scores:
        if isinstance(item, tuple):
            doc, _ = item
            retrieved_docs.append(doc)
        else:
            retrieved_docs.append(item)
    
    # 步骤2: 精排（如启用且可用）
    if RERANKER_ENABLED and reranker and retrieved_docs:
        reranked_docs = rerank_documents(question, retrieved_docs, top_k=top_k)
    else:
        # 如果没有精排，保留原始向量检索分数
        reranked_docs = []
        for item in retrieved_docs_with_scores[:top_k]:
            if isinstance(item, tuple):
                reranked_docs.append(item)
            else:
                reranked_docs.append((item, 1.0))
    
    # 检索相关对话历史
    relevant_history = memory.retrieve_relevant_history(question, top_k=3)
    
    return {"context": reranked_docs, "history": relevant_history}


def generate(state: State):
    """生成答案"""
    # 阈值设置：文档相似度阈值和整体意图判断阈值
    DOC_SIMILARITY_THRESHOLD = 0.75
    INTENT_SIMILARITY_THRESHOLD = 0.7
    
    # =========================
    # 🔹 金融数据自动补充（触发式）
    # =========================
    finance_context = ""
    question = state["question"]
    
    # 延迟加载金融触发器
    try:
        from src.finance_trigger import get_finance_trigger
        finance_trigger = get_finance_trigger()
        
        # 检查是否需要补充金融数据
        if finance_trigger.is_finance_related(question):
            finance_context, finance_sources = finance_trigger.get_finance_context(question)
            if finance_context:
                print(f"[金融数据] 检测到金融相关问题，补充数据: {finance_sources}")
    except Exception as e:
        print(f"金融数据模块加载失败: {e}")

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

    # 合并上下文：文档检索 + 金融数据
    if finance_context:
        if docs_content:
            combined_context = docs_content + "\n\n" + finance_context
        else:
            combined_context = finance_context
    else:
        combined_context = docs_content

    # 根据是否使用上下文调整提示词
    if use_retrieved_context and docs_content:
        prompt = prompt_template.format(
            question=state["question"],
            context=combined_context,
            history=history
        )
    elif finance_context:
        # 没有文档检索结果，但有金融数据
        finance_prompt_template = PromptTemplate.from_template(
            template="""Previous conversation:
{history}

[FINANCE DATA START]
{finance_context}
[FINANCE DATA END]

[USER QUESTION START]
{question}
[USER QUESTION END]

Instructions:
1. The text above in [FINANCE DATA START]...[FINANCE DATA END] contains real-time financial data.
2. Use the financial data to answer the question when relevant.
3. Answer based on your own knowledge if the financial data is not sufficient.
4. CRITICAL: Answer in the SAME LANGUAGE as the user's question.
5. Write only three sentences."""
        )
        prompt = finance_prompt_template.format(
            question=state["question"],
            finance_context=finance_context,
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

    # 延迟加载 LLM
    llm_model = _get_llm()
    response = llm_model.invoke(prompt)

    # 记录到检索日志（延迟加载）
    logger = RetrievalLogger(max_log_files=10)
    logger.log(
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
def ask_question(question: str, top_k: int = 3, user_name: str = None) -> dict:
    """
    问答函数，供 Web 界面调用

    意图分支逻辑：
    - INVESTMENT（投资顾问）→ 主要使用金融数据 + RAG
    - POLICY（政策咨询）→ 主要使用RAG文档检索
    - MIXED → 两者都使用
    - GENERAL → 两者都尝试

    Args:
        question: 用户问题
        top_k: 返回的文档数量
        user_name: 用户称呼（可选，用于个性化回复）

    Returns:
        dict: 包含 answer, source, question, used_context, intent
    """
    # =========================
    # 🔹 步骤1: 意图分类
    # =========================
    intent = "general"
    intent_reason = ""
    
    try:
        from src.intent_classifier import get_intent_classifier, Intent
        classifier = get_intent_classifier()
        intent_obj, intent_reason = classifier.classify(question)
        intent = intent_obj.value
        print(f"[意图分类] {intent} - {intent_reason}")
    except Exception as e:
        print(f"[意图分类] 分类失败，使用默认: {e}")
    
    # =========================
    # 🔹 步骤2: 根据意图分支处理
    # =========================
    use_finance = intent in ["investment", "mixed", "general"]
    use_rag = intent in ["investment", "policy", "mixed", "general"]  # 投资意图也需要RAG检索
    
    # 只有纯投资咨询（需要实时行情）才跳过RAG
    # 判断标准：问题包含具体的股票代码、基金代码或实时行情关键词
    pure_investment_patterns = [
        r'sh\d{6}', r'sz\d{6}',  # 股票代码
        r'\d{6}',  # 基金代码
        r'今天.*(涨|跌|收盘)', r'实时行情', r'当前价格',
    ]
    is_pure_investment = any(re.search(p, question) for p in pure_investment_patterns)
    
    if intent == "investment" and is_pure_investment and not use_rag:
        return _handle_investment_question(question, top_k)
    
    # 正常RAG流程
    response = graph.invoke({"question": question, "top_k": top_k})

    # 保存对话历史到 Markdown（延迟加载）
    memory = _get_memory_manager()
    memory.add_message("用户", question)
    memory.add_message("AI", response["answer"])

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

    # =========================
    # 🔹 合规审查（延迟加载）
    # =========================
    compliance_result = None
    answer_with_compliance = response["answer"]

    # 延迟加载合规审查器
    try:
        from src.compliance_checker import ComplianceChecker

        try:
            if not hasattr(_get_compliance, 'instance'):
                _get_compliance.instance = ComplianceChecker()
            checker = _get_compliance.instance

            if checker:
                try:
                    # 提取产品信息（从sources中获取）
                    product_info = "未知基金产品"
                    if sources:
                        source_names = set(s.get("source", "") for s in sources)
                        if source_names:
                            product_names = [Path(s).stem for s in source_names if s != "unknown"]
                            if product_names:
                                product_info = ", ".join(product_names)

                    # 调用合规审查
                    compliance_result = checker.check(
                        question=question,
                        answer=response["answer"],
                        product_info=product_info
                    )

                    # 在答案末尾添加合规标识
                    compliance_tag = _build_compliance_tag(compliance_result)
                    answer_with_compliance = response["answer"] + compliance_tag

                except Exception as e:
                    print(f"合规审查出错: {e}")
                    compliance_result = {
                        "is_compliant": None,
                        "risk_level": "unknown",
                        "violations": [],
                        "summary": f"合规审查失败: {str(e)}"
                    }

        except ImportError as e:
            print(f"[WARN] ComplianceChecker 模块导入失败: {e}")
            compliance_result = {
                "is_compliant": None,
                "risk_level": "unknown",
                "violations": [],
                "summary": "合规审查模块不可用"
            }

    except ValueError as e:
        # API Key 未配置
        print(f"[WARN] {e}")
        compliance_result = {
            "is_compliant": None,
            "risk_level": "unknown",
            "violations": [],
            "summary": "合规审查未配置"
        }
    except Exception as e:
        print(f"合规审查异常: {e}")
        compliance_result = {
            "is_compliant": None,
            "risk_level": "unknown",
            "violations": [],
            "summary": f"合规审查初始化失败: {str(e)}"
        }

    # 添加用户称呼（如果有）
    final_answer = answer_with_compliance
    if user_name:
        final_answer = f"{user_name}，{answer_with_compliance}"

    return {
        "question": question,
        "answer": final_answer,
        "source": sources,
        "used_context": used_context,
        "compliance": compliance_result,
        "intent": intent,
        "intent_reason": intent_reason
    }


def _handle_investment_question(question: str, top_k: int):
    """
    处理纯投资顾问类问题（不需要RAG文档检索）
    只使用金融数据 + LLM回答
    """
    from src.finance_trigger import get_finance_trigger
    from langchain_ollama import ChatOllama
    
    # 1. 获取金融数据
    finance_trigger = get_finance_trigger()
    finance_context, finance_sources = finance_trigger.get_finance_context(question)
    
    # 2. 使用LLM生成回答
    llm = ChatOllama(model="my-qwen25:latest", temperature=0.7)
    
    if finance_context:
        prompt = f"""你是一位专业的投资顾问。请根据以下实时金融数据回答用户问题。

金融数据:
{finance_context}

用户问题: {question}

要求:
1. 基于提供的金融数据进行分析
2. 如果数据不足，说明情况
3. 给出专业的投资建议
4. 用中文回答
"""
    else:
        # 没有获取到金融数据，使用通用回答
        prompt = f"""你是一位专业的投资顾问。请回答以下问题。

用户问题: {question}

要求:
1. 基于你的金融知识回答
2. 给出专业的投资建议
3. 用中文回答
4. 注意合规提示
"""
    
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, 'content') else str(response)
    
    # 3. 保存对话历史
    memory = _get_memory_manager()
    memory.add_message("用户", question)
    memory.add_message("AI", answer)
    
    return {
        "question": question,
        "answer": answer,
        "source": [],
        "used_context": bool(finance_context),
        "compliance": None,
        "intent": "investment",
        "intent_reason": "纯投资顾问类问题，使用金融数据"
    }


def _build_compliance_tag(compliance_result: dict) -> str:
    """构建合规标识（紧凑标签样式）"""
    if not compliance_result:
        return ""

    is_compliant = compliance_result.get("is_compliant")

    if is_compliant is True:
        tag = '<span class="status-compliance">[合规]</span>'
    elif is_compliant is False:
        tag = '<span class="status-compliance-risk">[风险]</span>'
    else:
        tag = ""
    return tag


def ask_question_stream(question: str, top_k: int, user_name: str = None):
    """
    流式版本的 ask_question
    返回生成器，逐步输出回答
    """
    # 1. 意图识别和检索（阻塞部分，快速完成）
    graph = _get_rag_graph()

    # 执行检索
    response = graph.invoke({"question": question, "top_k": top_k})

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

    DOC_SIMILARITY_THRESHOLD = 0.75

    for item in context_items:
        if isinstance(item, tuple):
            doc, score = item
            if has_scores:
                if is_distance:
                    normalized_score = 1.0 / (1.0 + score)
                else:
                    normalized_score = score
            else:
                normalized_score = 0.5

            if normalized_score >= DOC_SIMILARITY_THRESHOLD:
                sources.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page_label", "unknown"),
                    "similarity": normalized_score
                })

    used_context = len(sources) > 0

    # 获取意图信息
    intent = response.get("intent", "unknown")
    intent_reason = response.get("intent_reason", "")

    # 2. 合规审查（阻塞部分）
    compliance_result = None
    try:
        from src.compliance_checker import ComplianceChecker

        try:
            if not hasattr(_get_compliance, 'instance'):
                _get_compliance.instance = ComplianceChecker()
            checker = _get_compliance.instance

            if checker:
                product_info = "未知基金产品"
                if sources:
                    source_names = set(s.get("source", "") for s in sources)
                    if source_names:
                        product_names = [Path(s).stem for s in source_names if s != "unknown"]
                        if product_names:
                            product_info = ", ".join(product_names)

                compliance_result = checker.check(
                    question=question,
                    answer=response["answer"],
                    product_info=product_info
                )

        except ImportError as e:
            print(f"[WARN] ComplianceChecker 模块导入失败: {e}")
            compliance_result = {
                "is_compliant": None,
                "risk_level": "unknown",
                "violations": [],
                "summary": "合规审查模块不可用"
            }

    except ValueError as e:
        print(f"[WARN] {e}")
        compliance_result = {
            "is_compliant": None,
            "risk_level": "unknown",
            "violations": [],
            "summary": "合规审查未配置"
        }
    except Exception as e:
        print(f"合规审查异常: {e}")
        compliance_result = {
            "is_compliant": None,
            "risk_level": "unknown",
            "violations": [],
            "summary": f"合规审查初始化失败: {str(e)}"
        }

    # 构建完整答案
    answer_with_compliance = response["answer"]
    if compliance_result:
        compliance_tag = _build_compliance_tag(compliance_result)
        answer_with_compliance = response["answer"] + compliance_tag

    # 添加用户称呼（如果有）
    final_answer = answer_with_compliance
    if user_name:
        final_answer = f"{user_name}，{answer_with_compliance}"

    # 3. 流式输出答案
    # 先输出检索信息（如果有）
    if used_context:
        yield {"type": "sources", "data": sources}

    # 分字符流式输出答案
    for i in range(len(final_answer)):
        yield {"type": "content", "data": final_answer[:i+1]}

    # 完成后返回完整结果
    yield {
        "type": "complete",
        "data": {
            "question": question,
            "answer": final_answer,
            "source": sources,
            "used_context": used_context,
            "compliance": compliance_result,
            "intent": intent,
            "intent_reason": intent_reason
        }
    }


def clear_conversation_history():
    """清空对话历史"""
    memory = _get_memory_manager()
    memory.clear_history()


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
