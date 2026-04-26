"""
统一 LLM 客户端 - 使用 DeepSeek API
替换所有 Ollama 本地模型调用
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import Field, ConfigDict
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel

# 加载环境变量
BASE_DIR = Path(__file__).parent.parent
ENV_FILES = [
    BASE_DIR / ".env",
    BASE_DIR / "elastic-start-local/.env",
]
for env_file in ENV_FILES:
    if env_file.exists():
        load_dotenv(env_file)
        break

# DeepSeek 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


class ChatDeepSeek(BaseChatModel):
    """
    DeepSeek Chat 模型封装，兼容 LangChain BaseChatModel 接口。
    可直接替换 ChatOllama 使用。
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    # Pydantic 字段声明
    model_: str = Field(default=DEEPSEEK_MODEL, alias="model")
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 120
    api_key: str = Field(default=DEEPSEEK_API_KEY)
    base_url: str = Field(default=DEEPSEEK_BASE_URL)

    # 非 Pydantic 字段
    _client: Optional[OpenAI] = None

    def model_post_init(self, __context: Any) -> None:
        """初始化后设置非 Pydantic 字段"""
        # 解析 api_key（支持环境变量回退）
        if not self.api_key:
            self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "未找到 DeepSeek API Key，请设置 DEEPSEEK_API_KEY 环境变量\n"
                "1. 访问 https://platform.deepseek.com/\n"
                "2. 注册并获取 API Key\n"
                "3. 在 .env 文件中添加 DEEPSEEK_API_KEY=your_key"
            )
        # 解析 model
        if not self.model_ or self.model_ == DEEPSEEK_MODEL:
            env_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            if env_model:
                self.model_ = env_model
        # 创建客户端
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        """生成回复"""
        # 转换 LangChain messages 为 OpenAI 格式
        openai_messages = []
        for msg in messages:
            role = getattr(msg, 'type', 'user')
            if role == 'ai':
                role = 'assistant'
            elif role == 'system':
                role = 'system'
            else:
                role = 'user'
            content = getattr(msg, 'content', str(msg))
            openai_messages.append({"role": role, "content": content})

        # 调用 DeepSeek API
        request_kwargs: Dict[str, Any] = {
            "model": self.model_,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if stop:
            request_kwargs["stop"] = stop

        response = self._client.chat.completions.create(**request_kwargs)

        # 解析响应
        content = response.choices[0].message.content or ""

        generation = ChatGeneration(message=BaseMessage(content=content, type="ai"))
        return ChatResult(generations=[generation])

    def invoke(self, prompt: str, **kwargs) -> BaseMessage:
        """兼容旧接口：直接接收字符串 prompt"""
        if isinstance(prompt, str):
            messages = [BaseMessage(content=prompt, type="user")]
        else:
            messages = prompt

        result = self._generate(messages)
        return result.generations[0].message

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model_,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


# =========================
# 🔹 全局 LLM 实例（延迟加载）
# =========================
_llm_instance: Optional[ChatDeepSeek] = None


def get_llm(
    model: str = None,
    temperature: float = 0.0,
    api_key: str = None,
) -> ChatDeepSeek:
    """
    获取全局 LLM 实例（单例模式）

    Args:
        model: 模型名称，默认使用 DEEPSEEK_MODEL 环境变量
        temperature: 温度参数
        api_key: API Key，默认从环境变量获取

    Returns:
        ChatDeepSeek 实例
    """
    global _llm_instance
    if _llm_instance is None:
        kwargs = {"temperature": temperature}
        if model:
            kwargs["model"] = model
        if api_key:
            kwargs["api_key"] = api_key
        _llm_instance = ChatDeepSeek(**kwargs)
    return _llm_instance


def reset_llm():
    """重置 LLM 实例（用于切换模型等场景）"""
    global _llm_instance
    _llm_instance = None


def check_deepseek_available() -> tuple[bool, str]:
    """
    检查 DeepSeek API 是否可用

    Returns:
        (是否可用, 错误信息或成功消息)
    """
    try:
        llm = get_llm(temperature=0.0)
        response = llm._client.chat.completions.create(
            model=llm.model_,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        return True, "DeepSeek API 连接正常"
    except ValueError as e:
        return False, str(e)
    except Exception as e:
        return False, f"DeepSeek API 调用失败: {e}"
