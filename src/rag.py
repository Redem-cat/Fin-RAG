# RAG architecture using LangChain, Ollama and Elasticsearch
# Modified by Redem-cat

import os
import pickle
import re
import string
from pathlib import Path

import jieba
from dotenv import load_dotenv

from langchain_elasticsearch import ElasticsearchStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# =========================
# 🔹 配置和初始化
# =========================
base_path = Path(__file__).parent.parent.resolve()

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
llm = ChatOllama(model="llama3.2:3b", temperature=0.0000000001)

# =========================
# 🔹 停用词处理类
# =========================
class ChineseTextProcessor:
    """中文文本处理器：分词 + 停用词过滤"""
    
    def __init__(self, stopwords_file: str = None):
        self.stopwords = self.load_stopwords(stopwords_file)
    
    def load_stopwords(self, stopwords_file: str = None):
        """加载停用词库"""
        stopwords = set()
        if stopwords_file is None:
            stopwords_file = base_path / "中文停用词库.txt"
        
        stopwords_path = Path(stopwords_file)
        if stopwords_path.exists():
            with open(stopwords_path, "r", encoding="utf-8") as f:
                stopwords = {line.strip() for line in f if line.strip()}
            print(f"[OK] Loaded {len(stopwords)} stopwords")
        else:
            print("⚠️ 未找到停用词库文件，使用默认停用词")
            stopwords.update({
                "的", "了", "和", "是", "在", "我", "有", "就", "不", "人",
                "都", "一个", "上", "也", "很", "到", "说", "要", "去",
                "你", "会", "着", "没有", "看", "自己", "这", "那", "还", "什么"
            })
        return stopwords
    
    def process(self, text: str) -> str:
        """对文本进行分词并过滤停用词"""
        if not text:
            return text
        
        # 判断是否为中文文本
        if re.search(r'[\u4e00-\u9fff]', text):
            words = jieba.cut(text)
            cleaned_words = []
            for word in words:
                word = word.strip()
                if not word or word in self.stopwords:
                    continue
                if word in string.punctuation or re.match(r"^[\W_]+$", word):
                    continue
                if len(word) == 1:
                    continue
                cleaned_words.append(word)
            return " ".join(cleaned_words)
        
        # 英文文本直接返回
        return text

# =========================
# 🔹 缓存管理类
# =========================
class CacheManager:
    """缓存管理器：管理文档分块和向量索引缓存"""
    
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = base_path / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.chunks_cache = self.cache_dir / "doc_chunks.pkl"
        self.vectorizer_cache = self.cache_dir / "vectorizer_cache.pkl"
        self.vector_matrix_cache = self.cache_dir / "vector_matrix_cache.pkl"
    
    def save_chunks(self, chunks: List[Document]):
        """保存文档分块到缓存"""
        with open(self.chunks_cache, "wb") as f:
            pickle.dump(chunks, f)
        print(f"💾 已缓存 {len(chunks)} 个文档分块")
    
    def load_chunks(self) -> List[Document]:
        """从缓存加载文档分块"""
        if self.chunks_cache.exists():
            with open(self.chunks_cache, "rb") as f:
                chunks = pickle.load(f)
            print(f"✅ 从缓存加载了 {len(chunks)} 个文档分块")
            return chunks
        return None
    
    def clear_cache(self):
        """清除所有缓存"""
        for cache_file in [self.chunks_cache, self.vectorizer_cache, self.vector_matrix_cache]:
            if cache_file.exists():
                cache_file.unlink()
        print("🗑️ 缓存已清除")

# =========================
# 🔹 初始化组件
# =========================
text_processor = ChineseTextProcessor()
cache_manager = CacheManager()

vector_db = ElasticsearchStore(
    es_url=os.getenv('ES_LOCAL_URL'),
    embedding=embeddings,
    index_name=index_name
)

# 定义 Prompt
prompt_template = PromptTemplate.from_template(
    template="Given the following context: {context}, answer to the following question: {question}. Write only three sentences."
)

# 定义状态
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# 定义应用步骤
def retrieve(state: State):
    """检索相关文档"""
    # 对查询进行停用词处理
    processed_query = text_processor.process(state["question"])
    
    # 如果处理后的查询与原查询不同，打印日志
    if processed_query != state["question"]:
        print(f"🔍 查询处理: '{state['question']}' -> '{processed_query}'")
    
    retrieved_docs = vector_db.similarity_search(processed_query, k=8)
    return {"context": retrieved_docs}


def generate(state: State):
    """生成答案"""
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = prompt_template.format(question=state["question"], context=docs_content) 
    response = llm.invoke(prompt)
    return {"answer": response.content}


# 编译应用
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# =========================
# 🔹 对话函数（供 Streamlit 调用）
# =========================
def ask_question(question: str, top_k: int = 8):
    """
    问答函数，供 Web 界面调用
    
    Args:
        question: 用户问题
        top_k: 返回的文档数量
    
    Returns:
        dict: 包含 answer, source, question
    """
    # 临时修改 k 值
    original_k = 8
    
    # 调用图
    response = graph.invoke({"question": question})
    
    # 整理结果
    sources = []
    for doc in response.get("context", []):
        sources.append({
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "similarity": 0.9  # ES 不返回相似度，使用默认值
        })
    
    return {
        "question": question,
        "answer": response["answer"],
        "source": sources
    }

# =========================
# 🔹 主函数（命令行测试）
# =========================
if __name__ == "__main__":
    question = "Who won the Nobel Prize in Physics 2024?"
    print(question)
    response = graph.invoke({"question": question})
    print(response["answer"])
