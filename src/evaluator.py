"""RAG 系统评估器 - 使用 RAGAS 框架进行批量评估"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings

from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import (  # 使用旧路径以支持自定义 embeddings
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.run_config import RunConfig
import pandas as pd
from openai import OpenAI


class RAGEvaluator:
    """RAG 系统批量评估器"""

    def __init__(
        self,
        rag_chain,
        llm_provider: str = "deepseek",  # "deepseek" 或 "ollama"
        model_name: str = "deepseek-v4-pro",
        embed_model: str = "bge-m3:latest",
        embed_provider: str = "ollama",  # "openai" 或 "ollama"
        base_url: str = "http://localhost:11434",
        api_key: str = None
    ):
        """初始化评估器

        Args:
            rag_chain: RAG 检索链
            llm_provider: LLM 提供商，"deepseek" 或 "ollama"
            model_name: 使用的 LLM 模型名称
            embed_model: 使用的 Embedding 模型名称
            embed_provider: Embedding 提供商，"openai" 或 "ollama"
            base_url: Ollama 服务地址
            api_key: API Key（用于 DeepSeek/OpenAI）
        """
        # 加载环境变量
        base_path = Path(__file__).parent.parent.resolve()
        dotenv_path = base_path / "elastic-start-local/.env"
        load_dotenv(dotenv_path=dotenv_path)

        self.rag_chain = rag_chain

        # 初始化 LLM (使用 llm_factory 创建 InstructorLLM)
        if llm_provider == "deepseek":
            if api_key is None:
                api_key = os.getenv("DEEPSEEK_API_KEY")
            # 创建 OpenAI 客户端（指向 DeepSeek API）
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            # 使用 llm_factory 创建 InstructorLLM
            self.llm = llm_factory(model_name, client=client)
            # DeepSeek 不支持 n>1，需要在 run_config 中设置
            self._is_deepseek = True
        elif llm_provider == "ollama":
            # Ollama 也需要通过 OpenAI 兼容接口
            client = OpenAI(
                api_key="ollama",  # Ollama 不需要真实的 API key
                base_url=f"{base_url}/v1"
            )
            self.llm = llm_factory(model_name, client=client)
            self._is_deepseek = False
        else:
            raise ValueError(f"不支持的 LLM 提供商: {llm_provider}")

        # 初始化 Embeddings
        if embed_provider == "openai":
            # 使用 OpenAI 的 embedding API（如果有的话）
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY")
            self.embeddings = OpenAIEmbeddings(api_key=api_key)
        elif embed_provider == "ollama":
            self.embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url)
        else:
            raise ValueError(f"不支持的 Embedding 提供商: {embed_provider}")

        self.eval_results = None
        self._is_deepseek = getattr(self, '_is_deepseek', False)

    def load_testset(self, testset_path: str) -> List[Dict[str, Any]]:
        """加载测试集

        Args:
            testset_path: 测试集 JSON 文件路径

        Returns:
            测试数据列表
        """
        with open(testset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def _run_rag_for_testset(
        self,
        testset: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """对测试集中的每个问题运行 RAG 系统

        Args:
            testset: 测试集数据

        Returns:
            包含问题和检索上下文的列表
        """
        results = []

        for idx, item in enumerate(testset):
            question = item["question"]

            # 运行 RAG 链并获取检索到的上下文
            # 注意：这里需要根据你的 rag.py 实现调整
            try:
                # 调用 LangGraph
                response = self.rag_chain.invoke({"question": question, "top_k": 3})

                # 提取上下文
                contexts = []
                context_items = response.get("context", [])

                # 提取文档内容
                for item_ctx in context_items:
                    if isinstance(item_ctx, tuple):
                        doc, score = item_ctx
                        contexts.append(doc.page_content)
                    elif hasattr(item_ctx, 'page_content'):
                        contexts.append(item_ctx.page_content)

                results.append({
                    "question": question,
                    "answer": response.get("answer", ""),
                    "contexts": contexts,
                    "ground_truth": item.get("ground_truth", ""),
                    "reference": item.get("reference", ""),
                })
            except Exception as e:
                print(f"处理问题 {idx + 1} 时出错: {question[:50]}... - {e}")
                results.append({
                    "question": question,
                    "answer": "",
                    "contexts": [],
                    "ground_truth": item.get("ground_truth", ""),
                    "reference": item.get("reference", ""),
                })

        return results

    def evaluate(
        self,
        testset_path: str,
        metrics: Optional[List[str]] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """执行批量评估

        Args:
            testset_path: 测试集文件路径
            metrics: 要计算的指标列表，默认包含所有指标
            save_dir: 结果保存目录

        Returns:
            评估结果字典
        """
        # 默认评估指标
        if metrics is None:
            metrics = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]

        # 加载测试集
        print(f"加载测试集: {testset_path}")
        testset = self.load_testset(testset_path)
        print(f"共 {len(testset)} 个测试问题")

        # 运行 RAG 系统
        print("运行 RAG 系统...")
        rag_results = self._run_rag_for_testset(testset)

        # 构建 RAGAS 评估数据集
        print("准备 RAGAS 评估数据...")
        evaluation_data = []
        for result in rag_results:
            evaluation_data.append({
                "user_input": result["question"],
                "response": result["answer"],
                "retrieved_contexts": result["contexts"],
                "reference": result["ground_truth"] or result.get("reference", ""),
            })

        dataset = EvaluationDataset.from_list(evaluation_data)

        # 选择指标（旧版 metrics，支持自定义 embeddings）
        metric_map = {
            "faithfulness": Faithfulness(),
            "answer_relevance": AnswerRelevancy(),
            "context_precision": ContextPrecision(),
            "context_recall": ContextRecall(),
        }

        selected_metrics = [metric_map[m] for m in metrics if m in metric_map]

        # 执行评估
        print(f"执行评估，指标: {metrics}")
        # 创建 RunConfig
        # DeepSeek 不支持 n>1，会导致警告，所以设置较高的超时时间
        run_config = RunConfig(timeout=60, max_retries=2)

        # 禁用 RAGAS 的多生成请求（DeepSeek 不支持）
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 忽略警告
            result = evaluate(
                dataset=dataset,
                metrics=selected_metrics,
                llm=self.llm,
                embeddings=self.embeddings,  # 使用本地 Ollama 嵌入模型
                run_config=run_config,
            )

        # 转换为 DataFrame
        df = result.to_pandas()

        # 处理 NaN 值：将 NaN 替换为 None（JSON 中会变为 null）
        df = df.where(pd.notnull(df), None)

        # 保存结果
        self.eval_results = {
            "testset_path": testset_path,
            "metrics": metrics,
            "scores": df.to_dict(orient="records"),
            "summary": {
                metric: float(df[metric].mean()) if pd.notnull(df[metric].mean()) else None
                for metric in metrics
                if metric in df.columns
            },
            "timestamp": datetime.now().isoformat(),
        }

        # 保存到文件
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            # 保存详细结果
            detail_file = save_path / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(detail_file, 'w', encoding='utf-8') as f:
                json.dump(self.eval_results, f, ensure_ascii=False, indent=2)

            # 保存 CSV
            csv_file = save_path / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')

            print(f"结果已保存到: {detail_file}")

        return self.eval_results

    def get_summary(self) -> Dict[str, float]:
        """获取评估摘要

        Returns:
            各指标的平均分
        """
        if self.eval_results is None:
            raise ValueError("尚未执行评估，请先调用 evaluate() 方法")
        return self.eval_results["summary"]

    def get_dataframe(self) -> pd.DataFrame:
        """获取评估结果 DataFrame

        Returns:
            评估结果表格
        """
        if self.eval_results is None:
            raise ValueError("尚未执行评估，请先调用 evaluate() 方法")
        return pd.DataFrame(self.eval_results["scores"])
