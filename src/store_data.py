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
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, 
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", "；", "：", "，", "、", " ", ""]
)

# Read the PDF and Markdown files and split into chunks
converter = DocumentConverter()
all_splits = []

# 处理 PDF 文件
for file in glob.glob(f"{base_path}/data/*.pdf"):
    print(f"Reading {file}")
    docling_doc = converter.convert(file)

    pages = len(docling_doc.pages) if hasattr(docling_doc, 'pages') else 1
    print(f"Read {file} with {pages} pages")

    markdown_text = docling_doc.export_to_markdown()
    doc = Document(
        page_content=markdown_text,
        metadata={"source": file, "file_type": "pdf"}
    )

    chunks = text_splitter.split_documents([doc])
    num_chunks = len(chunks)
    print(f"Splitted in {num_chunks} chunks")
    all_splits.append(chunks)

# 处理 Markdown 文件
for file in glob.glob(f"{base_path}/data/*.md"):
    print(f"Reading {file}")
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()

    doc = Document(
        page_content=content,
        metadata={"source": file, "file_type": "markdown"}
    )

    chunks = text_splitter.split_documents([doc])
    num_chunks = len(chunks)
    print(f"Splitted in {num_chunks} chunks")
    all_splits.append(chunks)

# 合并所有分块
all_chunks = []
for chunks in all_splits:
    all_chunks.extend(chunks)

print(f"Storing chunks in Elasticsearch")
# Index the chunks to Elasticsearch
vector_db.add_documents(all_chunks)
print(f"Stored {len(all_chunks)} chunks in {index_name} index")
