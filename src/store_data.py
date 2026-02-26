# Store PDF documents in Elasticsearch using Ollama embeddings
# Modified by Redem-cat

import glob, os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_elasticsearch import ElasticsearchStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load and chunk contents of the PDF
base_path = Path(__file__).parent.parent.resolve()
  
# Index the chunks in Elasticsearch
dotenv_path = Path(base_path / "elastic-start-local/.env")
if not dotenv_path.is_file():
    print("Error: it seems Elasticsearch has not been installed")
    print("using start-local, please execute the following command:")
    print("curl -fsSL https://elastic.co/start-local | sh")
    exit(1)
    
load_dotenv(dotenv_path=dotenv_path)
index_name="rag-langchain"

# Embeddings
embeddings = OllamaEmbeddings(
    model="my-bge-m3",
)

vector_db  = ElasticsearchStore(
    es_url=os.getenv('ES_LOCAL_URL'),

    embedding=embeddings,
    index_name=index_name
)

# =========================
# 🔹 缓存管理
# =========================
cache_dir = base_path / "cache"
cache_dir.mkdir(exist_ok=True)
chunks_cache = cache_dir / "doc_chunks.pkl"

# 尝试从缓存加载已处理的文档分块
cached_chunks = None
if chunks_cache.exists():
    print("💾 发现文档分块缓存，正在加载...")
    try:
        with open(chunks_cache, "rb") as f:
            cached_chunks = pickle.load(f)
        print(f"✅ 从缓存加载了 {len(cached_chunks)} 个分块")
    except Exception as e:
        print(f"⚠️ 缓存加载失败: {e}")
        cached_chunks = None

# Check if the index already exists
res = vector_db.client.indices.exists(index=index_name)
if res.body:
    if cached_chunks:
        print(f"索引 {index_name} 已存在，且有缓存，跳过处理")
    else:
        print(f"索引 {index_name} 已存在于 Elasticsearch")
        exit(1)
    
# =========================
# 🔹 处理文档
# =========================
if cached_chunks is None:
    print(f"Reading the PDFs in {base_path}/data")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    # Read the PDF files and split into chunks
    all_splits = []
    for file in glob.glob(f"{base_path}/data/*.pdf"):
        loader = PyPDFLoader(file)
        docs = loader.load()
        pages=len(docs)
        print(f"Read {file} with {pages} pages")
        chunks = text_splitter.split_documents(docs)
        num_chunks=len(chunks)
        print(f"Splitted in {num_chunks} chunks")
        all_splits.append(chunks)
    
    # 合并所有分块
    all_chunks = []
    for chunks in all_splits:
        all_chunks.extend(chunks)
    
    # 保存到缓存
    try:
        with open(chunks_cache, "wb") as f:
            pickle.dump(all_chunks, f)
        print(f"💾 已缓存 {len(all_chunks)} 个文档分块")
    except Exception as e:
        print(f"⚠️ 缓存保存失败: {e}")
else:
    all_chunks = cached_chunks
    print(f"📂 使用缓存中的 {len(all_chunks)} 个文档分块")
            
print(f"Storing chunks in Elasticsearch")
# Index the chunks to Elasticsearch
vector_db.add_documents(all_chunks)
print(f"Stored {len(all_chunks)} chunks in {index_name} index")

