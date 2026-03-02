"""
数据索引脚本 - 将文档索引到Elasticsearch
支持指定目录，灵活配置
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_elasticsearch import ElasticsearchStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载环境变量
env_path = Path(__file__).parent.parent / "elastic-start-local" / ".env"
load_dotenv(env_path)


def get_text_splitter():
    """获取文本分割器"""
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )


def index_directory(directory: str, index_name: str = "rag-langchain"):
    """索引指定目录的所有文档"""
    
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"目录不存在: {directory}")
        return
    
    # 收集所有md文件
    md_files = list(dir_path.rglob("*.md"))
    md_files = [f for f in md_files if not f.name.startswith("_")]
    
    print(f"找到 {len(md_files)} 个Markdown文件")
    
    if not md_files:
        print("没有找到需要索引的文件")
        return
    
    # 初始化
    print("初始化 Embeddings 和 Elasticsearch...")
    embeddings = OllamaEmbeddings(model="my-bge-m3")
    vector_db = ElasticsearchStore(
        es_url=os.getenv('ES_LOCAL_URL'),
        embedding=embeddings,
        index_name=index_name
    )
    
    splitter = get_text_splitter()
    
    # 索引计数
    indexed_count = 0
    error_count = 0
    
    for md_file in md_files:
        try:
            # 读取文件
            content = md_file.read_text(encoding="utf-8")
            if not content.strip():
                print(f"跳过空文件: {md_file.name}")
                continue
            
            # 分割文本
            chunks = splitter.split_text(content)
            
            # 获取相对路径作为source
            source = str(md_file.relative_to(dir_path.parent))
            
            # 添加到向量库
            from langchain_core.documents import Document
            docs = [
                Document(
                    page_content=chunk,
                    metadata={"source": source, "file": md_file.name}
                )
                for chunk in chunks
            ]
            
            vector_db.add_documents(docs)
            indexed_count += 1
            print(f"已索引: {md_file.name} ({len(chunks)} 个chunks)")
            
        except Exception as e:
            error_count += 1
            print(f"错误 [{md_file.name}]: {e}")
    
    print(f"\n索引完成!")
    print(f"成功: {indexed_count} 个文件")
    print(f"失败: {error_count} 个文件")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="索引文档到Elasticsearch")
    parser.add_argument("--dir", "-d", type=str, 
                       default="data/01_金融法规",
                       help="要索引的目录")
    parser.add_argument("--index", "-i", type=str,
                       default="rag-langchain",
                       help="Elasticsearch索引名")
    
    args = parser.parse_args()
    
    # 转换相对路径为绝对路径
    base_dir = Path(__file__).parent.parent
    dir_path = base_dir / args.dir
    
    print(f"索引目录: {dir_path}")
    print(f"索引名称: {args.index}")
    print("-" * 50)
    
    index_directory(str(dir_path), args.index)


if __name__ == "__main__":
    main()
