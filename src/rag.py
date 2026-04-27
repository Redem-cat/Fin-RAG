# RAG architecture using LangChain, DeepSeek API and Elasticsearch
# Modified by Redem-cat

import os
import sys
import math
from typing import Dict, List, Any, Optional, Tuple

# 修复 Windows 控制台编码问题：强制使用 UTF-8 输出
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, IOError):
        pass

from datetime import datetime, timedelta
from pathlib import Path
import json
import re
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv

from langchain_elasticsearch import ElasticsearchStore
from langchain_ollama import OllamaEmbeddings
from src.llm_client import get_llm, ChatDeepSeek
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from typing import Dict, Any

# 尝试导入 reranker（可选依赖）
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    print("警告: sentence-transformers 未安装，精排功能不可用")

# 尝试导入触发系统
try:
    from src.trigger_system import get_trigger_manager, TriggerResult
    TRIGGER_SYSTEM_AVAILABLE = True
except ImportError as e:
    TRIGGER_SYSTEM_AVAILABLE = False
    print(f"警告: 触发系统模块导入失败: {e}")

# 尝试导入知识图谱模块
try:
    from src.knowledge_graph import (
        get_kg_retriever,
        hybrid_retrieval,
        KG_ENABLED,
        check_kg_status
    )
    KG_SYSTEM_AVAILABLE = True
    # 检查 KG 是否实际可用
    kg_status, _ = check_kg_status()
    if not kg_status:
        KG_SYSTEM_AVAILABLE = False
        print("警告: Neo4j 未连接，知识图谱功能不可用")
except ImportError as e:
    KG_SYSTEM_AVAILABLE = False
    print(f"警告: 知识图谱模块导入失败: {e}")
except Exception as e:
    KG_SYSTEM_AVAILABLE = False
    print(f"警告: 知识图谱初始化失败: {e}")

# 尝试导入引用追踪模块
try:
    from src.citation import (
        CitationSource,
        SourceType,
        get_citation_tracker
    )
    CITATION_SYSTEM_AVAILABLE = True
except ImportError as e:
    CITATION_SYSTEM_AVAILABLE = False
    print(f"警告: 引用追踪模块导入失败: {e}")


# =========================
# 🔹 HyDE 配置
# =========================
HYDE_ENABLED = os.getenv("HYDE_ENABLED", "false").lower() == "true"
HYDE_GENERATE_PROMPT = PromptTemplate.from_template(
    template="""你是一个专业的金融文档撰写助手。请根据用户的问题，生成一段假设性的文档内容。

用户问题: {question}

要求:
1. 生成的文档应该像是从权威金融资料或基金招募说明书中提取的内容
2. 包含与问题相关的专业术语和概念解释
3. 使用正式、专业的语言风格
4. 内容长度适中（100-200字）
5. 回答应该准确、专业，避免模糊表述

假设性文档:"""
)


def generate_hypothetical_document(question: str, llm_model: ChatDeepSeek = None) -> str:
    """
    使用 LLM 生成假设性文档（HyDE 核心）
    
    Args:
        question: 用户问题
        llm_model: LLM 模型实例
        
    Returns:
        str: 假设性文档内容
    """
    if llm_model is None:
        llm_model = _get_llm()
    
    prompt = HYDE_GENERATE_PROMPT.format(question=question)
    
    try:
        response = llm_model.invoke(prompt)
        hypothetical_doc = response.content if hasattr(response, 'content') else str(response)
        print(f"[HyDE] 生成了假设性文档，长度: {len(hypothetical_doc)} 字符")
        return hypothetical_doc
    except Exception as e:
        print(f"[HyDE] 生成假设性文档失败: {e}")
        return question  # 降级：使用原始问题

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
    """延迟加载 LLM（使用 DeepSeek API）"""
    global llm, _loaded
    if _loaded["llm"]:
        return llm
    print("正在连接 DeepSeek API...")
    llm = get_llm(temperature=0.0)
    _loaded["llm"] = True
    print("[OK] DeepSeek LLM 加载完成")
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

        # BGE-reranker 返回的是 logits，通过 sigmoid 归一化到 0-1
        import numpy as np
        scores = 1.0 / (1.0 + np.exp(-np.array(scores)))

        # 将文档和分数配对
        doc_scores = list(zip(documents, scores))

        # 按分数降序排序
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回 top_k
        return doc_scores[:top_k]
    except Exception as e:
        print(f"精排预测失败: {e}")
        return [(doc, 1.0) for doc in documents[:top_k]]




# =========================
# 🔹 OpenClaw 风格记忆系统核心组件
# =========================

# 中文停用词表
CHINESE_STOPWORDS = {
    "的", "了", "着", "是", "在", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "那", "个", "她", "他", "它", "们",
    "什么", "这个", "那个", "这样", "那样", "哪", "哪些", "哪个",
    "之", "与", "及", "或", "而", "但", "如", "如果", "因为", "所以",
    "可以", "可能", "应该", "需要", "能够", "进行", "通过", "关于",
    "对", "把", "被", "让", "给", "向", "从", "比", "为", "以",
    "中", "来", "将", "又", "还", "再", "更", "已", "已经", "正在",
    "啊", "吧", "呢", "吗", "哦", "嗯", "哈", "呀", "嘛", "罢了",
    "即", "便", "则", "乃", "且", "虽", "其", "此", "彼", "某",
}


class EmbeddingCache:
    """
    Embedding LRU 缓存层（参考 OpenClaw 设计）
    - 同一段文本只计算一次 embedding
    - 使用 LRU 策略淘汰最旧记录
    - 大幅减少 Ollama API 调用次数
    """

    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.cache: Dict[str, np.ndarray] = {}
        self.access_order: List[str] = []  # 用于追踪 LRU 顺序

    def get(self, text: str) -> Optional[np.ndarray]:
        """获取缓存的 embedding"""
        key = self._hash(text)
        if key in self.cache:
            # 更新访问顺序（移到末尾）
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, text: str, embedding: np.ndarray):
        """存入缓存"""
        key = self._hash(text)
        if key in self.cache:
            # 已存在则更新值并移到末尾
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # 达到上限，淘汰最旧的
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]

        self.cache[key] = embedding
        self.access_order.append(key)

    @staticmethod
    def _hash(text: str) -> str:
        """生成文本哈希（用于缓存键）"""
        import hashlib
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @property
    def size(self) -> int:
        return len(self.cache)

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_order.clear()


class BM25Scorer:
    """
    BM25 评分器（Okapi BM25 算法实现）
    参考 OpenClaw 的 SQLite FTS5 BM25 方案，纯 Python 实现
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.df: Dict[str, int] = {}
        self.avgdl: float = 0.0
        self.n_docs: int = 0
        self.doc_lengths: Dict[str, int] = {}
        self.initialized = False

    def build_index(self, documents: Dict[str, str]):
        """构建 BM25 索引"""
        self.df.clear()
        self.doc_lengths.clear()
        total_length = 0

        for doc_id, text in documents.items():
            tokens = tokenize_chinese_bigram(text.lower())
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] = self.df.get(token, 0) + 1

        self.n_docs = len(documents)
        self.avgdl = total_length / max(self.n_docs, 1)
        self.initialized = True
        print(f"[BM25] 索引构建完成: {self.n_docs} 个文档, {len(self.df)} 个独立词汇, 平均文档长度: {self.avgdl:.1f}")

    def batch_score(self, query: str, documents: Dict[str, str]) -> List[Tuple[str, float]]:
        """批量计算所有文档的 BM25 得分"""
        if not self.initialized:
            return []

        query_tokens = set(tokenize_chinese_bigram(query.lower()))
        results = []

        for doc_id, text in documents.items():
            doc_tokens = tokenize_chinese_bigram(text.lower())
            dl = len(doc_tokens)
            doc_score = 0.0

            for token in query_tokens:
                if token not in self.df:
                    continue
                idf = math.log((self.n_docs - self.df[token] + 0.5) / (self.df[token] + 0.5) + 1)
                tf = doc_tokens.count(token)
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1)))
                doc_score += idf * tf_norm

            if doc_score > 0:
                results.append((doc_id, doc_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


def tokenize_chinese_bigram(text: str) -> List[str]:
    """
    中文 bigram 分词器（参考 OpenClaw 设计）
    无词典方案，单字 + bigram，内置停用词过滤
    比简单 regex 分词召回率提升约 30%
    """
    tokens = []
    segments = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9_]+', text.lower())

    for segment in segments:
        if re.match(r'[\u4e00-\u9fff]+', segment):
            for i, char in enumerate(segment):
                if char not in CHINESE_STOPWORDS:
                    tokens.append(char)
                if i < len(segment) - 1:
                    bigram = segment[i:i + 2]
                    if bigram[0] not in CHINESE_STOPWORDS and bigram[1] not in CHINESE_STOPWORDS:
                        tokens.append(bigram)
        else:
            if len(segment) >= 2:
                tokens.append(segment)

    return tokens


def time_decay_score(days_ago: int, half_life: int = 30) -> float:
    """
    时间衰减函数（OpenClaw 指数衰减模型）
    decay = e^(-λ × days_ago), λ = ln(2) / half_life
    半衰期内权重 > 0.5, 60天后 ≈ 0.25
    """
    lam = math.log(2) / half_life
    return math.exp(-lam * days_ago)


def mmr_rerank(
    items: List[Tuple[str, float, Any]],
    lambda_param: float = 0.7,
    top_k: int = 5
) -> List[Tuple[str, float, Any]]:
    """
    MMR 最大边际相关性重排
    在相关性和多样性之间取得平衡，避免结果冗余
    MMR(d) = λ × Rel(d) - (1-λ) × max(Sim(d, d_selected))
    使用轻量级 Jaccard 相似度替代向量相似度
    """
    if not items:
        return []

    selected = []
    remaining = list(items)

    while remaining and len(selected) < top_k:
        best_mmr = -float('inf')
        best_idx = 0

        for i, (item_id, rel_score, content) in enumerate(remaining):
            relevance_part = rel_score

            if selected:
                max_sim = 0.0
                content_tokens = set(tokenize_chinese_bigram(content.lower())[:50])

                for _, _, sel_content in selected:
                    sel_tokens = set(tokenize_chinese_bigram(sel_content.lower())[:50])
                    intersection = content_tokens & sel_tokens
                    union = content_tokens | sel_tokens
                    sim = len(intersection) / max(len(union), 1)
                    max_sim = max(max_sim, sim)

                diversity_penalty = max_sim
            else:
                diversity_penalty = 0.0

            mmr_score = lambda_param * relevance_part - (1 - lambda_param) * diversity_penalty

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


# =========================
# 🔹 OpenClaw 风格 Memory Manager v2.0
# =========================

# 记忆检索触发关键词（轻量级第一道防线）
MEMORY_TRIGGER_KEYWORDS = [
    # 时间/历史类
    "之前", "上次", "之前说的", "之前提到的", "记得", "我说过", "刚才", "那时",
    "以前", "过去", "曾经", "还记得吗", "忘了吗", "你记得",
    # 身份/偏好类
    "我叫", "我的名字", "我是", "我住在", "我喜欢", "我想要", "我不喜欢",
    "我的工作", "我的职业", "我在做", "我的风险", "我的投资风格",
    # 决策/观点类
    "我决定", "我认为", "我觉得", "我计划", "我想买", "我打算",
    "我的目标", "我选择", "我同意", "我不同意",
    # 回溯指代
    "那个", "它", "这个方案", "那件事", "同样的问题",
]


class MemoryManager:
    """
    OpenClaw 风格记忆系统（v2.0）

    核心改进（对标 OpenClaw Memory）:
    1. 四层记忆架构: SOUL/MEMORY/daily/evolution-log
    2. BM25 + 向量混合检索 (7:3 加权)
    3. MMR 最大边际相关性去重
    4. 时间衰减模型（半衰期30天）+ 常青记忆
    5. Embedding LRU 缓存层
    6. 中文 bigram 分词（无依赖）
    7. 智能 Compaction（LLM 驱动的真正摘要）
    8. 进化日志追踪
    9. 【新增】智能读取触发：轻量判断 → 需要才检索
    10. 【新增】智能写入触发：预判价值 → 有值才写
    """
    # 混合检索权重配置（参考 OpenClaw 7:3 配置）
    BM25_WEIGHT = 0.3
    VECTOR_WEIGHT = 0.7
    TIME_DECAY_HALF_LIFE = 30
    TIME_DECAY_ENABLED = True
    MMR_LAMBDA = 0.7
    EMBEDDING_CACHE_SIZE = 500
    EVERGREEN_FILES = {"MEMORY.md", "SOUL.md", "AGENTS.md", "evolution-log.md"}

    def __init__(self, memory_dir: str = None, compaction_interval: int = 10):
        if memory_dir is None:
            memory_dir = base_path / "memory"
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)

        # ===== 文件路径（四层架构）=====
        self.soul_file = self.memory_dir / "SOUL.md"
        self.agents_file = self.memory_dir / "AGENTS.md"
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.daily_dir = self.memory_dir / "daily"
        self.daily_dir.mkdir(exist_ok=True)
        self.evolution_log = self.memory_dir / "evolution-log.md"

        self.compaction_interval = compaction_interval
        self.conversation_count = 0

        # OpenClaw 核心组件
        self.embedding_cache = EmbeddingCache(max_size=self.EMBEDDING_CACHE_SIZE)
        self.bm25 = BM25Scorer(k1=1.5, b=0.75)
        self._bm25_dirty = True
        self._bm25_doc_cache: Dict[str, str] = {}

        self._ensure_files()
        print(f"[Memory v2.0] OpenClaw 风格记忆系统已初始化")

    def _ensure_files(self):
        """确保必要文件存在（四层架构）"""
        if not self.soul_file.exists():
            self.soul_file.write_text(
                "# SOUL.md - AI 灵魂配置文件\n\n"
                "你是谁，定义了你能提供什么样的帮助。\n"
                "（此文件为常青记忆，不受时间衰减影响）\n",
                encoding="utf-8")
        if not self.agents_file.exists():
            self.agents_file.write_text(
                "# AGENTS.md - Agent 行为规范\n\n"
                "# 检索参数\n"
                "- 记忆检索: BM25+向量混合(7:3), MMR去重, 时间衰减(半衰期30天)\n"
                "（此文件为常青记忆，不受时间衰减影响）\n",
                encoding="utf-8")
        if not self.memory_file.exists():
            self.memory_file.write_text(
                "# 长期记忆（常青记忆）\n\n## 用户偏好\n\n## 核心事实\n\n## 关键决策\n\n",
                encoding="utf-8")
        if not self.evolution_log.exists():
            self.evolution_log.write_text(
                "# 进化日志 (evolution-log)\n\n记录 Agent 能力的增长轨迹和重要事件。\n\n---\n\n"
                f"- **{datetime.now().strftime('%Y-%m-%d %H:%M')}**: 记忆系统初始化（OpenClaw 风格 v2.0）\n\n",
                encoding="utf-8")

    def _get_today_file(self) -> Path:
        today = datetime.now().strftime("%Y-%m-%d")
        return self.daily_dir / f"{today}.md"

    def _get_all_daily_files(self, max_days: int = 90) -> List[Path]:
        files = []
        for i in range(min(max_days, 365)):
            day = datetime.now() - timedelta(days=i)
            day_file = self.daily_dir / f"{day.strftime('%Y-%m-%d')}.md"
            if day_file.exists():
                files.append(day_file)
        return files

    def _get_searchable_files(self) -> List[Tuple[Path, bool]]:
        """获取所有可搜索文件及其是否为常青记忆"""
        files = [
            (self.memory_file, True),
            (self.soul_file, True),
            (self.agents_file, True),
            (self.evolution_log, True),
        ]
        daily_files = self._get_all_daily_files()
        for df in daily_files:
            files.append((df, False))
        return files

    def _rebuild_bm25_index_if_needed(self):
        """按需重建 BM25 索引"""
        if not self._bm25_dirty:
            return
        documents = {}
        for file_path, _ in self._get_searchable_files():
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding="utf-8")
                    if content.strip():
                        documents[file_path.name] = content
                except (OSError, IOError):
                    continue
        if documents:
            self.bm25.build_index(documents)
            self._bm25_doc_cache = documents
            self._bm25_dirty = False

    def _invalidate_cache(self):
        """使缓存失效"""
        self._bm25_dirty = True

    def _chunk_text(self, text: str, chunk_size: int = 400) -> list:
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        for line in lines:
            current_chunk.append(line)
            current_size += len(line)
            if current_size >= chunk_size:
                chunks.append('\n'.join(current_chunk))
                current_chunk = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                current_size = 0
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        return chunks

    def _get_cached_embedding(self, text: str) -> np.ndarray:
        """获取 embedding（带 LRU 缓存），相同文本只计算一次"""
        cached = self.embedding_cache.get(text)
        if cached is not None:
            return cached
        emb = embeddings.embed_query(text)
        arr = np.array(emb, dtype=np.float32)
        self.embedding_cache.put(text, arr)
        return arr

    def _hybrid_retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        核心：BM25 + 向量混合检索（OpenClaw 7:3 架构）
        流程: BM25评分 → 向量评分 → 加权融合 → 时间衰减 → MMR去重
        """
        searchable = self._get_searchable_files()
        self._rebuild_bm25_index_if_needed()

        # 准备文档数据
        doc_data = {}
        for file_path, is_evergreen in searchable:
            if not file_path.exists():
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
                if not content.strip():
                    continue
            except (OSError, IOError):
                continue
            if is_evergreen:
                days_ago = 0
            else:
                match = re.search(r'(\d{4}-\d{2}-\d{2})', file_path.name)
                if match:
                    try:
                        file_date = datetime.strptime(match.group(1), "%Y-%m-%d")
                        days_ago = max(0, (datetime.now() - file_date).days)
                    except ValueError:
                        days_ago = 0
                else:
                    days_ago = 0
            doc_data[file_path.name] = (content, is_evergreen, days_ago)

        if not doc_data:
            return []

        # ========== 阶段 A: BM25 评分 ==========
        bm25_results = {}
        if self.bm25.initialized:
            bm25_batch = self.bm25.batch_score(query, {k: v[0] for k, v in doc_data.items()})
            for doc_id, score in bm25_batch:
                bm25_results[doc_id] = score

        # ========== 阶段 B: 向量评分 ==========
        vector_results = {}
        query_embedding = self._get_cached_embedding(query)

        for filename, (content, _, _) in doc_data.items():
            chunks = self._chunk_text(content)
            best_sim = 0.0
            for chunk in chunks:
                chunk_emb = self._get_cached_embedding(chunk)
                similarity = float(np.dot(query_embedding, chunk_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb) + 1e-8))
                best_sim = max(best_sim, similarity)
            vector_results[filename] = best_sim

        # ========== 阶段 C: 归一化 + 加权融合 + 时间衰减 ==========
        fused_results = []
        for filename, (content, is_evergreen, days_ago) in doc_data.items():
            bm25_score = bm25_results.get(filename, 0.0)
            vec_score = vector_results.get(filename, 0.0)
            bm25_norm = min(bm25_score / 20.0, 1.0) if bm25_score > 0 else 0.0
            vec_norm = max(0.0, min(vec_score, 1.0))
            hybrid_score = self.BM25_WEIGHT * bm25_norm + self.VECTOR_WEIGHT * vec_norm

            if self.TIME_DECAY_ENABLED and not is_evergreen and days_ago > 0:
                decay = time_decay_score(days_ago, self.TIME_DECAY_HALF_LIFE)
                decayed_score = hybrid_score * decay
            else:
                decay = 1.0
                decayed_score = hybrid_score

            fused_results.append({
                "file": filename, "content": content,
                "hybrid_score": hybrid_score, "bm25_score": bm25_score,
                "vector_score": vec_score, "decayed_score": decayed_score,
                "decay_factor": decay, "days_ago": days_ago, "is_evergreen": is_evergreen,
            })

        # ========== 阶段 D: MMR 去重重排 ==========
        sorted_by_decay = sorted(fused_results, key=lambda x: x["decayed_score"], reverse=True)
        candidates_for_mmr = [(r["file"], r["decayed_score"], r["content"]) for r in sorted_by_decay[:top_k * 3]]
        reranked = mmr_rerank(candidates_for_mmr, lambda_param=self.MMR_LAMBDA, top_k=top_k)

        # ========== 阶段 E: 最终输出 ==========
        final_results = []
        seen_files = set()
        for item_id, mmr_score, content in reranked:
            if item_id in seen_files:
                continue
            seen_files.add(item_id)
            for r in fused_results:
                if r["file"] == item_id:
                    final_results.append({**r, "mmr_score": mmr_score})
                    break
            if len(final_results) >= top_k:
                break

        if final_results:
            t = final_results[0]
            print(f"[Memory-Hybrid] Top-1: {t['file']} (混合={t['hybrid_score']:.3f}, BM25={t['bm25_score']:.2f}, 向量={t['vector_score']:.3f}, 衰减后={t['decayed_score']:.3f}, 缓存={self.embedding_cache.size}条)")

        return final_results

    def add_message(self, role: str, content: str):
        """添加对话消息到当日日志"""
        today_file = self._get_today_file()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not today_file.exists():
            today_file.write_text(f"# {datetime.now().strftime('%Y-%m-%d')} 对话日志\n\n", encoding="utf-8")
        content_md = today_file.read_text(encoding="utf-8")
        content_md += f"- **{timestamp} {role}**: {content}\n\n"
        today_file.write_text(content_md, encoding="utf-8")
        self._invalidate_cache()
        self.conversation_count += 1
        if self.conversation_count >= self.compaction_interval:
            self.compact()
            self.conversation_count = 0

    def retrieve_relevant_history(self, query: str, top_k: int = 3, threshold: float = 0.15) -> str:
        """混合检索（OpenClaw 风格：BM25+向量+MMR+时间衰减）"""
        results = self._hybrid_retrieve(query, top_k=top_k)
        filtered = [r for r in results if r["decayed_score"] >= threshold]
        if not filtered:
            return ""
        formatted = []
        for r in filtered:
            meta_info = f" ({r['days_ago']}天前, 衰减={r['decay_factor']:.2f})" if r['days_ago'] > 0 else ""
            formatted.append(
                f"<memory-snippet file=\"{r['file']}\" "
                f"hybrid=\"{r['hybrid_score']:.3f}\" decayed=\"{r['decayed_score']:.3f}\"{meta_info}>\n"
                f"{r['content'][:800]}\n</memory-snippet>")
        return "\n\n".join(formatted)

    def get_recent_conversation(self, rounds: int = 5) -> str:
        """获取最近 N 轮完整对话（用于多轮对话指代消解）"""
        all_lines = []
        for i in range(7):
            day = datetime.now() - timedelta(days=i)
            day_file = self.daily_dir / f"{day.strftime('%Y-%m-%d')}.md"
            if day_file.exists():
                c = day_file.read_text(encoding="utf-8")
                for line in c.split("\n"):
                    line = line.strip()
                    if line.startswith("- **") and (" 用户**:" in line or " AI**:" in line):
                        all_lines.append(line)
        all_lines.reverse()
        recent_lines = all_lines[-rounds * 2:]
        return "\n".join(recent_lines) if recent_lines else ""

    def compact(self):
        """智能 Compaction：LLM 驱动的真正摘要（替代简单占位符）"""
        recent_content = []
        dates = []
        for i in range(3):
            day = datetime.now() - timedelta(days=i)
            day_file = self.daily_dir / f"{day.strftime('%Y-%m-%d')}.md"
            if day_file.exists():
                c = day_file.read_text(encoding="utf-8")
                if c.strip():
                    recent_content.append(c)
                    dates.append(day.strftime("%Y-%m-%d"))
        if not recent_content:
            return
        existing_memory = self.memory_file.read_text(encoding="utf-8")

        try:
            from src.llm_client import get_llm as _get_llm
            llm = _get_llm()
            compact_prompt = f"""你是一个智能记忆压缩助手。将近期对话日志提炼为精简的长期记忆摘要。

## 现有长期记忆
{existing_memory}

## 近期对话日志（待压缩）
{chr(10).join([f'=== {d} ==={c}' for d, c in zip(dates, recent_content)])}

## 要求
请分析上述对话，提取值得长期保存的信息：
1. 用户偏好更新：是否有新的投资风格、风险偏好等信息？
2. 事实补充：用户是否透露了新的个人信息？
3. 关键决策：用户是否做出了新决定？

规则：只提取有价值的新信息，不重复已有内容。如果都是闲聊，输出"无重要更新"。
直接输出需要追加的内容（Markdown格式）："""

            response = llm.invoke(compact_prompt)
            summary = response.content if hasattr(response, 'content') else str(response)
            summary = summary.strip()
            if summary.startswith("```"):
                lines = summary.split("\n")
                summary = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            if summary and summary != "无重要更新" and len(summary) > 10:
                memory_content = existing_memory.rstrip()
                memory_content += f"\n\n### {datetime.now().strftime('%Y-%m-%d')} 记忆提炼\n{summary}\n"
                self.memory_file.write_text(memory_content, encoding="utf-8")
                evo_content = self.evolution_log.read_text(encoding="utf-8")
                evo_content += (f"\n- **{datetime.now().strftime('%Y-%m-%d %H:%M')}**: "
                                f"执行 Compaction，提炼 {sum(len(c) for c in recent_content)} 字符为 {len(summary)} 字符\n")
                self.evolution_log.write_text(evo_content, encoding="utf-8")
                print(f"[Memory-Compact] 智能压缩完成: 提炼 {len(summary)} 字符")
            else:
                print("[Memory-Compact] 近期无重要信息需记录")
            self._invalidate_cache()

        except Exception as e:
            print(f"[Memory-Compact] 智能压缩失败，降级为基础模式: {e}")
            memory_content = self.memory_file.read_text(encoding="utf-8")
            memory_content += f"\n### {datetime.now().strftime('%Y-%m-%d')} 摘要\n（近期对话已整合，共涉及 {len(dates)} 天）\n"
            self.memory_file.write_text(memory_content, encoding="utf-8")

    def get_soul(self) -> str:
        return self.soul_file.read_text(encoding="utf-8") if self.soul_file.exists() else ""

    def get_agents(self) -> str:
        return self.agents_file.read_text(encoding="utf-8") if self.agents_file.exists() else ""

    def update_memory_from_conversation(self, question: str, answer: str):
        """用 LLM 分析最近对话，提取关键信息并更新 MEMORY.md"""
        try:
            recent = self.get_recent_conversation(rounds=10)
            current_exchange = f"用户: {question}\nAI: {answer}"
            conversation_context = f"{recent}\n\n## 最新一轮\n{current_exchange}"
            existing_memory = self.memory_file.read_text(encoding="utf-8") if self.memory_file.exists() else ""
            from src.llm_client import get_llm as _get_llm
            llm = _get_llm()

            memory_prompt = f"""你是一个记忆管理助手。从对话中提取关于用户的**重要信息**并更新长期记忆文件。

## 现有长期记忆
{existing_memory}

## 最近对话内容
{conversation_context}

## 要求
请提取：1.用户偏好 2.核心事实 3.关键决策 4.人格设定相关
规则：只提取事实性持久性信息，不重复已有内容，不编造信息。
直接输出更新后的完整记忆：

# 长期记忆

## 用户偏好
（提取的用户偏好）

## 核心事实
（提取的核心事实）

## 关键决策
（提取的关键决策）
"""
            response = llm.invoke(memory_prompt)
            updated_memory = response.content if hasattr(response, 'content') else str(response)
            updated_memory = updated_memory.strip()
            if updated_memory.startswith("```"):
                lines = updated_memory.split("\n")
                updated_memory = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            if "# 长期记忆" in updated_memory or "## 用户偏好" in updated_memory:
                self.memory_file.write_text(updated_memory, encoding="utf-8")
                print(f"[Memory] 长期记忆已更新 ({len(updated_memory)} 字符)")
                self._invalidate_cache()
            else:
                print("[Memory] LLM 返回格式异常，跳过更新")

        except Exception as e:
            print(f"[Memory] 记忆更新失败: {e}")

    def clear_history(self):
        """清空对话历史（含缓存和索引清理）"""
        for f in self.daily_dir.glob("*.md"):
            f.unlink()
        if self.evolution_log.exists():
            self.evolution_log.write_text(
                "# 进化日志 (evolution-log)\n\n---\n\n"
                f"- **{datetime.now().strftime('%Y-%m-%d %H:%M')}:** 对话历史已手动清空\n\n",
                encoding="utf-8")
        self.memory_file.write_text("# 长期记忆\n\n## 用户偏好\n\n## 核心事实\n\n## 关键决策\n\n", encoding="utf-8")
        self.conversation_count = 0
        self.embedding_cache.clear()
        self._invalidate_cache()
        print("[Memory] 对话历史已清空（含缓存和索引）")

    # =========================
    # 🔹 智能触发系统（v2.0 新增）
    # =========================

    def should_retrieve_memory(self, query: str) -> bool:
        """
        轻量级记忆读取触发判断（第一道防线）

        原理：大多数问题（如"你好"、"查询某股票行情"）
             不需要访问长期记忆，跳过检索可节省 ~200-500ms

        触发条件（任一满足即触发）：
          1. 命中记忆关键词列表
          2. MEMORY.md 非空且有实际内容（说明有用户画像可参考）

        Returns:
            True: 需要检索记忆 → 调用 _hybrid_retrieve()
            False: 不需要 → 返回空字符串，跳过检索
        """
        query_lower = query.lower().strip()

        # 条件1：关键词快速匹配（零成本）
        for keyword in MEMORY_TRIGGER_KEYWORDS:
            if keyword in query_lower:
                print(f"[Memory-Trigger] ✓ 命中关键词 '{keyword}' → 执行检索")
                return True

        # 条件2：MEMORY.md 有实质内容时，对疑问句也尝试检索
        if self.memory_file.exists():
            mem_content = self.memory_file.read_text(encoding="utf-8")
            # 检查是否有实际内容（排除纯模板）
            has_real_content = (
                len(mem_content) > 100 and
                ("用户偏好" in mem_content or "核心事实" in mem_content)
            )
            # 疑问词 + 有记忆内容 → 尝试检索
            question_indicators = ["吗", "？", "?", "什么", "怎么", "如何", "哪", "谁", "是否"]
            is_question = any(q in query_lower for q in question_indicators)

            if has_real_content and is_question:
                print(f"[Memory-Trigger] ✓ 疑问句 + 有记忆内容 → 执行检索")
                return True

        # 条件3：涉及"我"的陈述句可能需要更新/参照记忆
        if "我" in query and len(query) > 5:
            self_referential = [
                "我的", "我想", "我要", "我计划", "我喜欢", "我决定",
                "我是", "我在", "我做"
            ]
            if any(r in query for r in self_referential):
                print(f"[Memory-Trigger] ✓ 自我指涉语句 → 执行检索")
                return True

        # 未命中任何条件
        print(f"[Memory-Trigger] ✗ 未命中（问题: {query[:20]}...）→ 跳过记忆检索")
        return False

    def should_update_memory(self, question: str, answer: str) -> tuple:
        """
        轻量级记忆写入预判（避免每轮都调用 LLM 更新记忆）

        原理：大部分对话是闲聊或简单问答（如"你好"、查股价），
             不值得消耗 LLM token 做记忆提取

        判断逻辑（分层过滤）：
          Level 1: 问题长度太短 (< 4字) → 直接跳过
          Level 2: 答案太短 (< 20字) → 大概率无信息量 → 跳过
          Level 3: 关键词预判（不含个人信息类关键词）→ 跳过
          Level 4: 通过所有检查 → 执行 LLM 记忆提取

        Returns:
            (should_update: bool, reason: str)
        """
        q = question.strip()
        a = answer.strip()

        # === Level 1: 问题长度过滤 ===
        if len(q) < 4:
            return False, f"问题过短({len(q)}字)，跳过记忆更新"

        # === Level 2: 回答信息量过滤 ===
        if len(a) < 30:
            return False, f"回答过短({len(a)}字)，大概率无持久性信息"

        # === Level 3: 个人信息关键词检测（子串匹配，更宽容）===
        personal_indicators = [
            # 身份类
            "我叫", "名字叫", "我是", "我的职业", "我的工作", "我住在",
            "我来自", "我在", "我做",
            # 偏好类 - 扩展变体
            "喜欢", "不喜欢", "偏好", "希望", "想要", "关注", "感兴趣",
            "风险", "保守型", "激进型", "稳健型",
            # 决策/计划类
            "决定", "选择", "计划", "打算",
            # 家庭/背景
            "我家", "我有孩子", "我结婚了",
        ]
        has_personal_info = any(indicator in (q + a) for indicator in personal_indicators)

        if not has_personal_info:
            # 额外检查：是否在讨论具体决策
            decision_words = ["决定", "选择", "买", "卖", "买入", "卖出", "持有", "定投", "配置"]
            has_decision = any(dw in q for dw in decision_words)

            if not has_decision:
                return False, "未检测到个人信息或关键决策，跳过记忆更新"

        # === Level 4: 通过所有检查 → 执行 LLM 提取 ===
        return True, "检测到潜在有价值的信息 → 执行 LLM 记忆提取"

    def smart_retrieve(self, query: str, top_k: int = 3, threshold: float = 0.15) -> str:
        """
        智能记忆检索入口（带触发判断的 retrieve_relevant_history 替代品）

        与 retrieve_relevant_history 的区别：
        - 此方法会先做 should_retrieve_memory() 判断
        - 未触发则直接返回空字符串（不调用 BM25/向量等重计算）

        推荐在 Agent 和 RAG 流程中使用此方法替代直接的 retrieve_relevant_history()
        """
        if not self.should_retrieve_memory(query):
            return ""

        # 触发了 → 执行完整混合检索
        return self.retrieve_relevant_history(query, top_k=top_k, threshold=threshold)

    def smart_update(self, question: str, answer: str):
        """
        智能记忆更新入口（带预判的 update_memory_from_conversation 替代品）

        与 update_memory_from_conversation 的区别：
        - 此方法会先做 should_update_memory() 预判
        - 未通过预判则跳过 LLM 调用（节省 token 和时间）
        """
        should, reason = self.should_update_memory(question, answer)

        if not should:
            print(f"[Memory-SmartUpdate] ⏭️ 跳过: {reason}")
            return

        print(f"[Memory-SmartUpdate] ✓ {reason} → 执行更新")
        self.update_memory_from_conversation(question, answer)

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆系统运行统计（调试/监控用）"""
        daily_files = []
        for i in range(90):
            day = datetime.now() - timedelta(days=i)
            df = self.daily_dir / f"{day.strftime('%Y-%m-%d')}.md"
            if df.exists():
                daily_files.append(df)
        mem_sz = self.memory_file.stat().st_size if self.memory_file.exists() else 0
        evo_sz = self.evolution_log.stat().st_size if self.evolution_log.exists() else 0
        return {
            "version": "2.0 (OpenClaw-style)",
            "daily_logs_count": len(daily_files),
            "memory_file_size_kb": round(mem_sz / 1024, 1),
            "evolution_log_size_kb": round(evo_sz / 1024, 1),
            "embedding_cache_size": self.embedding_cache.size,
            "embedding_cache_max": self.embedding_cache.max_size,
            "bm25_initialized": self.bm25.initialized,
            "bm25_vocab_size": len(self.bm25.df),
            "bm25_docs": self.bm25.n_docs,
            "time_decay_half_life": self.TIME_DECAY_HALF_LIFE,
            "search_range_days": 90,
        }

# =========================
# 🔹 初始化组件
# 注意：ElasticsearchStore 和 MemoryManager 使用延迟初始化
# 延迟加载可以避免在 Elasticsearch 未启动时应用无法启动
# =========================
memory_manager = None  # 延迟初始化

# 注意：不要在这里直接初始化 ElasticsearchStore！
# 它会在首次使用时通过 _get_vector_db() 延迟初始化

# =========================
# 🔹 系统身份定义
# =========================
SYSTEM_IDENTITY = """你是一个专业的金融智能助手，名叫 FinRAG-Advisor。

你的职责是：
1. 回答金融法规、投资理财相关问题
2. 提供市场数据分析和投资建议
3. 提示投资风险，确保合规性

请基于提供的知识库信息，给出准确、专业的回答。
如果遇到不确定的问题，请明确告知用户你无法回答。"""

# 定义 Prompt（包含对话历史）
prompt_template = PromptTemplate.from_template(
    template="""{system_identity}

Recent conversation history:
{recent_conversation}

Relevant past memories:
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
3. Use the Recent conversation history to understand context and resolve pronouns (e.g., "it", "this", "that") in the current question.
4. Answer the user's question based on the document fragments when relevant, otherwise use your own knowledge.
5. CRITICAL: Answer in the SAME LANGUAGE as the user's question, NOT the language of the document fragments.
6. Write only three sentences."""
)

# =========================
# 🔹 锐思文本分析工具提示（LLM 自主决策调用）
# =========================
RESSET_TOOL_DESCRIPTION = """
【可用工具 - 锐思文本分析 API】
你拥有调用锐思(RESSET)文本分析API的能力。当你认为需要获取以下类型的文本数据来回答用户问题时，请在回答末尾添加调用标记。

支持的数据类型和调用格式:
1. 中国上市公司财经文本: [RESSET_CALL:cn_report:股票代码:报告类型:年份]
   - 报告类型: 年度报告, 第一季度报告, 第二季度报告, 第三季度报告, 问询函及回复说明, IPO招股说明书, 内部控制评价报告, 社会责任报告, 审计报告, 业绩说明会全文
   - 示例: [RESSET_CALL:cn_report:000002:年度报告:2023] 获取万科A的2023年年度报告

2. 政府工作报告: [RESSET_CALL:gov_report:区域代码::年份]
   - 区域代码: 100100(国务院), 100101(北京), 100109(上海), 100111(浙江), 100119(广东)等
   - 示例: [RESSET_CALL:gov_report:100100::2023] 获取国务院2023年政府工作报告

3. 美国上市公司报告: [RESSET_CALL:us_report:股票代码:报告类型:年份]
   - 报告类型: 10K, 10Q, 424B
   - 示例: [RESSET_CALL:us_report:AMZN:10K:2023] 获取亚马逊2023年10-K报告

4. 财经新闻资讯: [RESSET_CALL:financial_news:::年份]
   - 示例: [RESSET_CALL:financial_news:::2023] 获取2023年财经新闻

5. 研究报告: [RESSET_CALL:research:报告类型::年份]
   - 报告类型: 宏观分析, 行业分析, 证券市场研究, 公司研究, 期货研究, 晨会汇编
   - 示例: [RESSET_CALL:research:行业分析::2023] 获取2023年行业分析研究报告

6. 股吧评论: [RESSET_CALL:forum:论坛类型::年份]
   - 论坛类型: 东方财富, 雪球
   - 示例: [RESSET_CALL:forum:东方财富::2023] 获取2023年东方财富股吧评论

7. 房产拍卖信息: [RESSET_CALL:real_estate:拍卖类型::年份]
   - 拍卖类型: 京东拍卖_拍卖公告, 京东拍卖_竞买须知, 人民法院诉讼资产网_拍卖公告等
   - 示例: [RESSET_CALL:real_estate:京东拍卖_拍卖公告::2023] 获取2023年京东拍卖公告

调用规则:
- 只有当用户问题确实需要上述文本数据时才调用，不要在闲聊或无关问题时调用
- 每次最多调用一个类型
- 调用标记放在回答内容的最后，格式严格为 [RESSET_CALL:类型:参数1:参数2:年份]
- 调用后系统会自动获取数据并补充到你的回答中
"""

RESSET_CALL_PATTERN = r'\[RESSET_CALL:([a-z_]+):([^:]*):([^:]*):(\d{4})\]'

# 定义状态
class State(TypedDict):
    question: str
    top_k: int
    context: List[Document]
    history: str
    recent_conversation: str
    answer: str
    # 触发系统注入的上下文
    resset_context: str
    ml_context: str

# 定义应用步骤
def retrieve(state: State):
    """检索相关文档和对话历史（包含精排 + HyDE）"""
    top_k = state.get("top_k", 3)
    question = state["question"]
    
    # 延迟加载组件
    vdb = _get_vector_db()
    memory = _get_memory_manager()
    
    # HyDE: 如果启用，先生成假设性文档
    hypothetical_doc = ""
    retrieval_query = question
    if HYDE_ENABLED:
        print("[HyDE] 正在生成假设性文档...")
        llm_model = _get_llm()
        hypothetical_doc = generate_hypothetical_document(question, llm_model)
        retrieval_query = hypothetical_doc
        print(f"[HyDE] 检索词从原始问题切换为假设性文档")
    
    # 步骤1: 向量检索召回候选文档
    # 召回更多候选，用于精排
    candidate_count = max(RERANK_CANDIDATE_COUNT, top_k * 3)
    retrieved_docs_with_scores = vdb.similarity_search_with_score(retrieval_query, k=candidate_count)
    
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
    
    # 智能记忆检索：带触发判断（需要才检索，不需要则返回空）
    relevant_history = memory.smart_retrieve(question, top_k=3)

    # 获取最近 N 轮完整对话（用于多轮对话指代消解）
    recent_conversation = memory.get_recent_conversation(rounds=5)

    # ===== 诊断日志：追踪对话历史检索 =====
    print(f"[诊断-retrieve] 检索到的对话历史长度: {len(relevant_history)} 字符")
    print(f"[诊断-retrieve] 最近对话轮次: {len(recent_conversation.split(chr(10))) if recent_conversation else 0} 行")
    if relevant_history:
        print(f"[诊断-retrieve] 对话历史内容:\n{relevant_history[:500]}")
        if question in relevant_history:
            print(f"[诊断-retrieve⚠️] 当前用户问题出现在检索到的对话历史中!")

    # 如果启用了 HyDE，将假设性文档也传给后续步骤
    extra_data = {"hyde_doc": hypothetical_doc} if HYDE_ENABLED else {}

    return {"context": reranked_docs, "history": relevant_history, "recent_conversation": recent_conversation, **extra_data}


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

    # =========================
    # 🔹 锐思文本数据自动补充（LLM 自主决策 + 关键词触发）
    # =========================
    resset_context = ""
    try:
        from src.resset_data import should_trigger_resset, get_resset_context
        if should_trigger_resset(question):
            resset_context = get_resset_context(question)
            if resset_context:
                print(f"[锐思数据] 检测到文本分析相关问题，补充数据")
    except Exception as e:
        print(f"锐思数据模块加载失败: {e}")

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

    history = state.get("history", "") or "No relevant past memories."
    recent_conversation = state.get("recent_conversation", "") or "No previous conversation."

    # =========================
    # 🔹 ML 策略上下文补充（优先使用触发系统注入，回退到关键词检测）
    # =========================
    ml_context = state.get("ml_context", "") or ""
    if not ml_context:
        try:
            ml_keywords = ["机器学习", "训练模型", "滚动训练", "walk_forward", "walkforward",
                           "xgboost", "lightgbm", "randomforest", "lstm", "深度学习",
                           "热启动", "快照恢复", "checkpoint", "warm_start", "warmstart", "快照",
                           "多策略", "模拟盘", "组合策略", "slot",
                           "特征工程", "pipeline", "交叉验证", "过拟合",
                           "量化", "回测", "策略"]
            if any(kw in question for kw in ml_keywords):
                ml_context = get_ml_strategy_context(question)
                if ml_context:
                    print(f"[ML策略] 检测到量化/ML相关问题，补充上下文")
        except Exception as e:
            print(f"ML策略上下文获取失败: {e}")

    # 锐思上下文（优先使用触发系统注入，回退到关键词检测）
    resset_context_from_trigger = state.get("resset_context", "") or ""
    if resset_context_from_trigger:
        resset_context = resset_context_from_trigger

    # 合并上下文：文档检索 + ML 上下文 + 锐思文本 + 金融数据
    combined_context = docs_content
    if ml_context:
        combined_context = (combined_context + "\n\n" + ml_context) if combined_context else ml_context
    if resset_context:
        combined_context = (combined_context + "\n\n" + resset_context) if combined_context else resset_context
    if finance_context:
        combined_context = (combined_context + "\n\n" + finance_context) if combined_context else finance_context

    # 根据是否使用上下文调整提示词
    if use_retrieved_context and docs_content:
        prompt = prompt_template.format(
            system_identity=SYSTEM_IDENTITY,
            question=state["question"],
            context=combined_context,
            history=history,
            recent_conversation=recent_conversation
        )
    elif finance_context:
        # 没有文档检索结果，但有金融数据
        finance_prompt_template = PromptTemplate.from_template(
            template="""{system_identity}

Recent conversation history:
{recent_conversation}

Relevant past memories:
{history}

[FINANCE DATA START]
{finance_context}
[FINANCE DATA END]

[USER QUESTION START]
{question}
[USER QUESTION END]

Instructions:
1. The text above in [FINANCE DATA START]...[FINANCE DATA END] contains real-time financial data.
2. Use the Recent conversation history to understand context and resolve pronouns.
3. Use the financial data to answer the question when relevant.
4. Answer based on your own knowledge if the financial data is not sufficient.
5. CRITICAL: Answer in the SAME LANGUAGE as the user's question.
6. Write only three sentences."""
        )
        prompt = finance_prompt_template.format(
            system_identity=SYSTEM_IDENTITY,
            question=state["question"],
            finance_context=finance_context,
            history=history,
            recent_conversation=recent_conversation
        )
    else:
        # 不使用检索结果，直接基于模型知识回答
        no_context_prompt = PromptTemplate.from_template(
            template="""{system_identity}

Recent conversation history:
{recent_conversation}

Relevant past memories:
{history}

[USER QUESTION START]
{question}
[USER QUESTION END]

Instructions:
1. Use the Recent conversation history to understand context and resolve pronouns.
2. The retrieved documents are not relevant to this question.
3. Answer based on your own knowledge.
4. CRITICAL: Answer in the SAME LANGUAGE as the user's question.
5. Write only three sentences."""
        )
        prompt = no_context_prompt.format(
            system_identity=SYSTEM_IDENTITY,
            question=state["question"],
            history=history,
            recent_conversation=recent_conversation
        )

    # 🔹 注入锐思工具描述到 prompt 末尾
    try:
        from src.resset_data import check_resset_available
        available, _ = check_resset_available()
        if available:
            prompt = prompt + "\n\n" + RESSET_TOOL_DESCRIPTION
    except Exception:
        pass  # RESSET 不可用则不注入

    # 延迟加载 LLM
    llm_model = _get_llm()

    # ===== 诊断日志：追踪用户问题是否被 LLM 重复返回 =====
    print(f"\n{'='*60}")
    print(f"[诊断] 用户原始问题: {state['question']}")
    print(f"[诊断] 发送给 LLM 的 prompt 长度: {len(prompt)} 字符")
    print(f"[诊断] prompt 前500字符:\n{prompt[:500]}")
    print(f"[诊断] prompt 后300字符:\n{prompt[-300:]}")
    print(f"{'='*60}")

    response = llm_model.invoke(prompt)

    # 🔹 诊断：检查 LLM 原始回复是否包含用户问题
    raw_answer = response.content
    user_q = state["question"]
    print(f"\n[诊断] LLM 原始回复长度: {len(raw_answer)} 字符")
    print(f"[诊断] LLM 原始回复前300字符:\n{raw_answer[:300]}")
    if user_q in raw_answer:
        # 找到用户问题在回复中出现的位置
        idx = raw_answer.find(user_q)
        context_start = max(0, idx - 50)
        context_end = min(len(raw_answer), idx + len(user_q) + 50)
        print(f"[诊断⚠️] 用户问题出现在 LLM 回复中! 位置: {idx}")
        print(f"[诊断⚠️] 上下文: ...{raw_answer[context_start:context_end]}...")
    else:
        print(f"[诊断✅] 用户问题未出现在 LLM 回复中")
    print(f"{'='*60}\n")

    # 🔹 后处理：检测 LLM 输出中的 RESSET 调用标记并执行
    answer_text = response.content
    resset_enrichment = ""

    import re as _re
    resset_matches = _re.findall(RESSET_CALL_PATTERN, answer_text)
    if resset_matches:
        for match in resset_matches:
            call_type, param1, param2, year = match
            try:
                from src.resset_data import (
                    get_cn_company_report, get_government_report,
                    get_us_company_report, get_financial_news,
                    get_research_report, get_forum_posts,
                    get_real_estate_info, format_resset_content,
                )

                data = []
                data_type_label = ""

                if call_type == "cn_report":
                    data = get_cn_company_report(param1, param2 or "年度报告", year)
                    data_type_label = f"中国上市公司{param2 or '年度报告'} ({param1})"
                elif call_type == "gov_report":
                    data = get_government_report(param1 or "100100", year)
                    data_type_label = f"政府工作报告 ({param1 or '国务院'})"
                elif call_type == "us_report":
                    data = get_us_company_report(param1, param2 or "10K", year)
                    data_type_label = f"美国上市公司{param2 or '10K'} ({param1})"
                elif call_type == "financial_news":
                    data = get_financial_news(year)
                    data_type_label = f"财经新闻资讯 ({year})"
                elif call_type == "research":
                    data = get_research_report(param1 or "公司研究", year)
                    data_type_label = f"研究报告-{param1 or '公司研究'} ({year})"
                elif call_type == "forum":
                    data = get_forum_posts(param1 or "东方财富", year)
                    data_type_label = f"股吧评论-{param1 or '东方财富'} ({year})"
                elif call_type == "real_estate":
                    data = get_real_estate_info(param1 or "京东拍卖_拍卖公告", year)
                    data_type_label = f"房产拍卖-{param1 or '京东拍卖_拍卖公告'} ({year})"

                if data:
                    formatted = format_resset_content(data, data_type_label)
                    resset_enrichment += f"\n\n---\n**[锐思数据补充 - {data_type_label}]**\n\n{formatted}"
                    print(f"[锐思数据] LLM 调用成功: {data_type_label}, 获取 {len(data)} 条数据")
                else:
                    resset_enrichment += f"\n\n*[锐思数据: 未找到 {data_type_label} 的相关数据]*"

            except Exception as e:
                resset_enrichment += f"\n\n*[锐思数据调用失败: {e}]*"
                print(f"[锐思数据] LLM 调用失败: {e}")

        # 移除调用标记，附加获取到的数据
        answer_text = _re.sub(RESSET_CALL_PATTERN, '', answer_text).strip()
        answer_text += resset_enrichment

    # 更新 response 的内容
    response.content = answer_text

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
    # 🔹 步骤0: 触发系统分析
    # =========================
    trigger_results = []
    kg_context = ""
    kg_sources = []
    resset_context_for_answer = ""
    ml_context_for_answer = ""
    if TRIGGER_SYSTEM_AVAILABLE:
        try:
            trigger_manager = get_trigger_manager()
            trigger_results = trigger_manager.analyze(question)

            # 打印触发结果日志
            for result in trigger_results:
                if result:
                    print(f"[触发] {result.trigger_type}: {result.reason} (置信度: {result.confidence:.2f})")

                    # 如果触发了 KG，执行知识图谱检索
                    if result.trigger_type == "kg" and KG_SYSTEM_AVAILABLE:
                        try:
                            retriever = get_kg_retriever()
                            kg_result = retriever.query(question)
                            if kg_result.entities:
                                kg_context = f"\n\n【知识图谱分析】\n{kg_result.explanation}\n"
                                for entity in kg_result.entities[:5]:
                                    kg_context += f"- {entity.get('name', '')} ({entity.get('type', '')})"
                                    if 'relation' in entity:
                                        kg_context += f" - {entity['relation']}"
                                    kg_context += "\n"
                                kg_sources.append({"type": "knowledge_graph", "confidence": kg_result.confidence})
                                print(f"[KG] 检索到 {len(kg_result.entities)} 个实体")
                        except Exception as e:
                            print(f"[KG] 检索失败: {e}")

                    # 如果触发了锐思文本分析，获取文本数据
                    if result.trigger_type == "resset":
                        try:
                            from src.resset_data import get_resset_context
                            resset_context_for_answer = get_resset_context(question)
                            if resset_context_for_answer:
                                print(f"[锐思] 获取到文本分析数据")
                        except Exception as e:
                            print(f"[锐思] 数据获取失败: {e}")

                    # 如果触发了量化策略，获取 ML/快照/多策略上下文
                    if result.trigger_type == "quant":
                        try:
                            ml_context_for_answer = get_ml_strategy_context(question)
                            if ml_context_for_answer:
                                print(f"[量化] 获取到 ML 策略上下文数据")
                        except Exception as e:
                            print(f"[量化] ML 策略上下文获取失败: {e}")
        except Exception as e:
            print(f"[触发系统] 分析失败: {e}")
    
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
    # 🔹 步骤2: Agent 模式（按需调用工具）
    # =========================
    memory = _get_memory_manager()
    recent_conversation = memory.get_recent_conversation(rounds=5)

    _agent_mode = True
    try:
        from src.agent import agent_ask as _agent_ask
        agent_result = _agent_ask(question, conversation_history=recent_conversation, top_k=top_k)
        agent_answer = agent_result["answer"]
        agent_used_context = agent_result["used_context"]
        agent_tool_calls = agent_result.get("tool_calls", [])

        # 从工具调用结果构建 sources（用于前端展示）
        sources = []
        for tc in agent_tool_calls:
            if tc["name"] == "rag_search" and tc["result"]:
                # 解析检索结果，构建 source 列表
                sources.append({
                    "content": tc["result"][:1000],
                    "source": f"Agent检索: {tc['args'].get('query', '')}",
                    "similarity": 0.85
                })

        used_context = agent_used_context
        print(f"[Agent] 工具调用: {len(agent_tool_calls)} 个, 使用上下文: {used_context}")

    except Exception as e:
        print(f"[Agent] 执行失败，fallback 到传统 RAG: {e}")
        _agent_mode = False
        # Fallback: 传统 RAG 流程
        response = graph.invoke({
            "question": question,
            "top_k": top_k,
            "resset_context": resset_context_for_answer,
            "ml_context": ml_context_for_answer,
        })
        agent_answer = response["answer"]
        used_context = False
        sources = []

    # 保存对话历史到 Markdown（延迟加载）
    memory = _get_memory_manager()
    memory.add_message("用户", question)
    memory.add_message("AI", agent_answer)

    # 智能记忆更新：先预判是否有价值 → 有价值才调 LLM 提取（节省 token）
    # 替代原来的无条件调用 update_memory_from_conversation()
    memory.smart_update(question, agent_answer)

    # Fallback 模式下需要重新处理 sources
    if not _agent_mode:
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
    answer_with_compliance = agent_answer

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
                        answer=agent_answer,
                        product_info=product_info
                    )

                    # 在答案末尾添加合规标识
                    compliance_tag = _build_compliance_tag(compliance_result)
                    answer_with_compliance = agent_answer + compliance_tag

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

    # 添加用户称呼（如果有，且看起来像合理的人名）
    final_answer = answer_with_compliance
    if user_name and len(user_name) <= 12 and not any(c in user_name for c in "。，！？；"):
        final_answer = f"{user_name}，{answer_with_compliance}"

    # ===== 诊断日志：追踪最终返回给前端的答案 =====
    print(f"\n[诊断-ask_question] 用户问题: {question}")
    print(f"[诊断-ask_question] 最终答案前300字符: {final_answer[:300]}")
    if question in final_answer:
        idx = final_answer.find(question)
        ctx_start = max(0, idx - 30)
        ctx_end = min(len(final_answer), idx + len(question) + 30)
        print(f"[诊断-ask_question⚠️] 用户问题出现在最终答案中! 位置: {idx}")
        print(f"[诊断-ask_question⚠️] 上下文: ...{final_answer[ctx_start:ctx_end]}...")
    else:
        print(f"[诊断-ask_question✅] 用户问题未出现在最终答案中")

    # =========================
    # 🔹 引用追踪
    # =========================
    citation_summary = ""
    if CITATION_SYSTEM_AVAILABLE:
        try:
            citation_sources = []
            
            # 向量检索来源
            for src in sources:
                citation_sources.append(CitationSource(
                    source_type=SourceType.VECTOR_SEARCH,
                    source_name=src.get("source", "未知"),
                    title=Path(src.get("source", "")).stem if src.get("source") else "文档片段",
                    content=src.get("content", ""),
                    confidence=src.get("similarity", 0.0),
                    metadata={"page": src.get("page", "")}
                ))
            
            # 知识图谱来源
            if kg_sources:
                for kg_src in kg_sources:
                    citation_sources.append(CitationSource(
                        source_type=SourceType.KNOWLEDGE_GRAPH,
                        source_name="Neo4j 知识图谱",
                        title=kg_src.get("type", "实体"),
                        content=kg_context[:200] if kg_context else "",
                        confidence=kg_src.get("confidence", 0.8)
                    ))
            
            # 记录引用
            tracker = get_citation_tracker()
            tracker.record(question, final_answer, citation_sources)
            
            # 生成摘要
            if citation_sources:
                citation_summary = tracker.format_summary(citation_sources)
            
        except Exception as e:
            print(f"[引用追踪] 错误: {e}")

    return {
        "question": question,
        "answer": final_answer,
        "source": sources,
        "used_context": used_context,
        "compliance": compliance_result,
        "intent": intent,
        "intent_reason": intent_reason,
        "triggers": [r.to_dict() if hasattr(r, 'to_dict') else r for r in trigger_results],
        "kg_context": kg_context,
        "kg_sources": kg_sources,
        "has_kg_results": bool(kg_context),
        "resset_context": resset_context_for_answer,
        "has_resset_results": bool(resset_context_for_answer),
        "ml_context": ml_context_for_answer,
        "has_ml_results": bool(ml_context_for_answer),
        "citation_summary": citation_summary
    }


def _handle_investment_question(question: str, top_k: int):
    """
    处理纯投资顾问类问题（不需要RAG文档检索）
    只使用金融数据 + LLM回答
    """
    from src.finance_trigger import get_finance_trigger
    
    # 1. 获取金融数据
    finance_trigger = get_finance_trigger()
    finance_context, finance_sources = finance_trigger.get_finance_context(question)
    
    # 2. 使用 DeepSeek LLM 生成回答
    llm = get_llm(temperature=0.7)
    
    if finance_context:
        prompt = f"""{SYSTEM_IDENTITY}

根据以下实时金融数据回答用户问题：

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
        prompt = f"""{SYSTEM_IDENTITY}

请回答以下问题：

用户问题: {question}

要求:
1. 基于你的金融知识回答
2. 给出专业的投资建议
3. 用中文回答
"""
    
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, 'content') else str(response)
    
    # 3. 保存对话历史
    memory = _get_memory_manager()
    memory.add_message("用户", question)
    memory.add_message("AI", answer)

    # 智能记忆更新（预判价值 → 有值才写）
    memory.smart_update(question, answer)
    
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

    # 添加用户称呼（如果有，且看起来像合理的人名）
    final_answer = answer_with_compliance
    if user_name and len(user_name) <= 12 and not any(c in user_name for c in "。，！？；"):
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


def get_ml_strategy_context(question: str) -> str:
    """
    根据 ML/热启动/多策略相关问题生成上下文

    Args:
        question: 用户问题

    Returns:
        上下文字符串
    """
    question_lower = question.lower()
    context_parts = []

    # 检测是否是 ML 相关问题
    ml_keywords = ["机器学习", "训练模型", "滚动训练", "walk_forward", "walkforward",
                   "xgboost", "lightgbm", "randomforest", "lstm", "深度学习",
                   "特征工程", "pipeline", "交叉验证", "过拟合"]
    snapshot_keywords = ["热启动", "快照恢复", "checkpoint", "warm_start", "warmstart", "快照"]
    multi_keywords = ["多策略", "模拟盘", "组合策略", "slot"]

    if any(kw in question_lower for kw in ml_keywords):
        try:
            from src.ml_strategy import get_available_ml_strategies, ML_STRATEGY_TEMPLATES
            strategies = get_available_ml_strategies()
            strategy_list = "\n".join([f"- {s['name']}: {s['description']}" for s in strategies])
            context_parts.append(f"""【ML 策略系统信息】
系统支持以下 ML 策略模板:
{strategy_list}

Walk-forward Validation 参数:
- train_window: 训练窗口（默认50）
- test_window: 测试窗口（默认20）
- rolling_step: 滚动步长（默认10）

特征集: basic（收益率）, technical（+RSI/MACD/布林带）, extended（全部特征）
""")
        except ImportError:
            context_parts.append("【ML 策略模块未安装】")

    if any(kw in question_lower for kw in snapshot_keywords):
        try:
            from src.snapshot_manager import list_snapshots
            snapshots = list_snapshots()
            if snapshots:
                snap_list = "\n".join([
                    f"- {s.get('name', 'unknown')}: {s.get('strategy_type', 'N/A')}, "
                    f"收益率 {s.get('total_return_pct', 0):.2f}%, "
                    f"创建于 {s.get('created_at', 'N/A')[:10]}"
                    for s in snapshots[:5]
                ])
                context_parts.append(f"""【热启动快照信息】
可用快照:
{snap_list}

热启动流程:
1. Phase 1: 正常回测 → save_snapshot() 保存状态
2. Phase 2: run_warm_start() 从快照恢复继续运行
3. 策略需在 on_start() 中通过 self.is_restored 避免覆盖已恢复状态
""")
            else:
                context_parts.append("【热启动】暂无已保存的快照。运行回测时可指定快照名称保存。")
        except ImportError:
            context_parts.append("【快照管理模块未安装】")

    if any(kw in question_lower for kw in multi_keywords):
        context_parts.append("""【多策略模拟盘信息】
系统支持多 slot 策略配置:
- 每个 slot 独立策略，共享同一引擎的 Portfolio
- 支持跨策略风控：限额、日损、仅平仓激活态
- 可计算跨策略指标：收益相关性、组合夏普比率

配置方式: MultiStrategySimulator.add_slot() 添加策略槽位
""")

    return "\n\n".join(context_parts)


def get_resset_explanation(question: str) -> str:
    """
    根据锐思文本分析相关问题生成上下文说明

    Args:
        question: 用户问题

    Returns:
        上下文字符串
    """
    question_lower = question.lower()
    context_parts = []

    # 锐思 API 能力说明
    resset_capabilities = """【锐思文本分析平台】
系统已接入锐思文本分析 API，支持以下数据获取:

1. 中国上市公司财经文本:
   - 年度报告、季度报告、问询函及回复、IPO招股说明书
   - 内部控制评价报告、社会责任报告、审计报告等
   - 需要提供股票代码（6位数字）和年份

2. 政府工作报告:
   - 国务院及各省市政府工作报告（1954年至今）
   - 需要提供行政区域代码和年份

3. 美国上市公司报告:
   - 10K年报、10Q季报、424B招股说明书
   - 需要提供美股代码（如 AMZN）和年份

4. 财经资讯 / 研究报告 / 股吧评论 / 房产拍卖信息

使用方式: 在 CHAT 页面直接输入包含"年报"、"政府工作报告"、"研究报告"等关键词的问题，
系统会自动触发锐思数据获取并融入回答。"""

    if any(kw in question_lower for kw in ["锐思", "resset"]):
        context_parts.append(resset_capabilities)

    return "\n\n".join(context_parts)


def create_rag_chain():
    """创建并返回 RAG 链，供评估器使用

    Returns:
        compiled graph: 编译好的 LangGraph
    """
    return graph


# =========================
# 🔹 锐思数据填充 RAG 知识库
# =========================
def ingest_resset_data_to_rag(
    data_type: str = "cn_report",
    stock_code: str = "000002",
    report_type: str = "年度报告",
    year: str = "2023",
    region_code: str = "100100",
    max_items: int = 10,
) -> Dict[str, Any]:
    """
    从锐思 API 获取文本数据并注入到 RAG 向量数据库

    支持的数据类型:
    - cn_report: 中国上市公司财经文本（需提供 stock_code, report_type, year）
    - gov_report: 政府工作报告（需提供 region_code, year）
    - us_report: 美国上市公司报告（需提供 stock_code, report_type, year）
    - financial_news: 财经新闻资讯（需提供 year）
    - research: 研究报告（需提供 report_type, year）
    - forum: 股吧评论（需提供 report_type=论坛名, year）
    - real_estate: 房产拍卖（需提供 report_type=拍卖类型, year）

    Args:
        data_type: 数据类型
        stock_code: 股票代码
        report_type: 报告/数据子类型
        year: 年份
        region_code: 行政区域代码
        max_items: 最大获取条数

    Returns:
        注入结果字典
    """
    try:
        from src.resset_data import (
            get_cn_company_report, get_government_report,
            get_us_company_report, get_financial_news,
            get_research_report, get_forum_posts,
            get_real_estate_info, check_resset_available,
        )
    except ImportError:
        return {"success": False, "error": "锐思模块未安装"}

    # 检查 API 可用性
    available, msg = check_resset_available()
    if not available:
        return {"success": False, "error": f"锐思 API 不可用: {msg}"}

    # 获取数据
    data = []
    label = ""

    if data_type == "cn_report":
        data = get_cn_company_report(stock_code, report_type, year)
        label = f"中国上市公司{report_type}_{stock_code}_{year}"
    elif data_type == "gov_report":
        data = get_government_report(region_code, year)
        label = f"政府工作报告_{region_code}_{year}"
    elif data_type == "us_report":
        data = get_us_company_report(stock_code, report_type, year)
        label = f"美国上市公司{report_type}_{stock_code}_{year}"
    elif data_type == "financial_news":
        data = get_financial_news(year)
        label = f"财经新闻资讯_{year}"
    elif data_type == "research":
        data = get_research_report(report_type, year)
        label = f"研究报告_{report_type}_{year}"
    elif data_type == "forum":
        data = get_forum_posts(report_type, year, max_items=max_items)
        label = f"股吧评论_{report_type}_{year}"
    elif data_type == "real_estate":
        data = get_real_estate_info(report_type, year)
        label = f"房产拍卖_{report_type}_{year}"
    else:
        return {"success": False, "error": f"未知数据类型: {data_type}"}

    if not data:
        return {"success": False, "error": f"未获取到数据 ({label})"}

    # 注入到向量数据库
    vdb = _get_vector_db()
    documents = []

    for i, item in enumerate(data[:max_items]):
        # 提取内容
        content = (
            item.get("part_content") or item.get("all_content") or
            item.get("Content") or item.get("content") or item.get("announcement") or ""
        )
        if isinstance(content, list):
            content = content[0] if content else ""
        if not content or len(content.strip()) < 50:
            continue

        # 截断过长内容
        if len(content) > 8000:
            content = content[:8000] + "...(内容已截断)"

        # 提取元数据
        title = item.get("title", "")
        if isinstance(title, list):
            title = title[0] if title else ""
        name = item.get("name", "")
        if isinstance(name, list):
            name = name[0] if name else ""
        code = item.get("code", "")
        if isinstance(code, list):
            code = code[0] if code else ""

        doc = Document(
            page_content=content,
            metadata={
                "source": f"resset_{data_type}",
                "title": title,
                "name": name,
                "code": code,
                "year": year,
                "data_type": data_type,
                "report_type": report_type,
                "ingested_at": datetime.now().isoformat(),
            }
        )
        documents.append(doc)

    if not documents:
        return {"success": False, "error": "获取的数据内容为空或过短"}

    # 批量添加到向量数据库
    try:
        ids = vdb.add_documents(documents)
        print(f"[锐思数据] 成功注入 {len(ids)} 条文档到 RAG 知识库 ({label})")
        return {
            "success": True,
            "label": label,
            "documents_ingested": len(ids),
            "total_data_fetched": len(data),
        }
    except Exception as e:
        return {"success": False, "error": f"注入向量数据库失败: {e}"}

# =========================
# 🔹 主函数（命令行测试）
# =========================
if __name__ == "__main__":
    # 测试用，请修改问题后运行
    pass
