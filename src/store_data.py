# Store PDF documents in Elasticsearch using Ollama embeddings
# Modified by Redem-cat

import glob, os
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat


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
index_name = "rag-langchain"

# Embeddings
embeddings = OllamaEmbeddings(
    model="my-bge-m3",
)

vector_db = ElasticsearchStore(
    es_url=os.getenv('ES_LOCAL_URL'),
    embedding=embeddings,
    index_name=index_name
)

# =========================
# 🔹 检查索引状态
# =========================
res = vector_db.client.indices.exists(index=index_name)
if res.body:
    print(f"索引 {index_name} 已存在，正在删除旧索引...")
    vector_db.client.indices.delete(index=index_name)
    print(f"旧索引已删除")

# =========================
# 🔹 处理文档
# =========================
print(f"Reading the PDFs in {base_path}/data")
# 法规文档专用分割策略：
# - chunk_size 1500 保证一个条款通常不会被截断
# - 去掉 "\n" 分隔符，避免在条款中间分割
# - 优先按段落(\n\n)和句子(。！？)分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    separators=["\n\n", "。", "！", "？", "；", "\n", "：", "，", "、", " ", ""]
)

# Read the PDF and Markdown files and split into chunks
converter = DocumentConverter()
all_splits = []

# 递归收集所有文件
pdf_files = list(Path(f"{base_path}/data").rglob("*.pdf"))
md_files = list(Path(f"{base_path}/data").rglob("*.md"))

print(f"Found {len(pdf_files)} PDF files, {len(md_files)} Markdown files")

# 处理 PDF 文件
for file in pdf_files:
    file_str = str(file)
    print(f"Reading {file_str}")
    try:
        docling_doc = converter.convert(file_str)
        pages = len(docling_doc.pages) if hasattr(docling_doc, 'pages') else 1
        print(f"Read {file_str} with {pages} pages")
        markdown_text = docling_doc.export_to_markdown()
        doc = Document(
            page_content=markdown_text,
            metadata={"source": file_str, "file_type": "pdf"}
        )
        chunks = text_splitter.split_documents([doc])
        print(f"Splitted in {len(chunks)} chunks")
        all_splits.append(chunks)
    except Exception as e:
        print(f"[WARN] Failed to process {file_str}: {e}")

# 处理 Markdown 文件
for file in md_files:
    file_str = str(file)
    print(f"Reading {file_str}")
    try:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
        doc = Document(
            page_content=content,
            metadata={"source": file_str, "file_type": "markdown"}
        )
        chunks = text_splitter.split_documents([doc])
        print(f"Splitted in {len(chunks)} chunks")
        all_splits.append(chunks)
    except Exception as e:
        print(f"[WARN] Failed to process {file_str}: {e}")

# 合并所有分块
all_chunks = []
for chunks in all_splits:
    all_chunks.extend(chunks)

print(f"Storing chunks in Elasticsearch")
# Index the chunks to Elasticsearch
vector_db.add_documents(all_chunks)
print(f"Stored {len(all_chunks)} chunks in {index_name} index")
