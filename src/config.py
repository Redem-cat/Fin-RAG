"""
配置文件 - 管理所有敏感配置和参数
从 .env 文件加载环境变量
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

# 加载 .env 文件
load_dotenv(ENV_FILE)


# ==================== API Keys ====================

# Tushare Pro Token
# 注册地址: https://tushare.pro/register?reg=538
# 免费注册后获取 Token
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")


# DeepSeek API Key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")


# ==================== Elasticsearch ====================

ES_LOCAL_URL = os.getenv("ES_LOCAL_URL", "http://localhost:9200")
ES_LOCAL_API_KEY = os.getenv("ES_LOCAL_API_KEY", "")


# ==================== DeepSeek LLM ====================

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")


# ==================== Ollama（已弃用，使用 DeepSeek 替代）===================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")  # 仅作参考，不再使用


# ==================== 其他配置 ====================

# RAG 相关
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


def is_tushare_configured() -> bool:
    """检查 Tushare 是否已配置"""
    return bool(TUSHARE_TOKEN.strip())
