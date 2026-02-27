"""RAG 系统评估器 - 使用 RAGAS 框架进行批量评估"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from ragas import EvaluationDataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
import pandas as pd


class RAGEvaluator:
    """RAG 系统批量评估器"""

    def __init__(
        self,
        rag_chain,
        model_name: str = "qwen2.5:7b",
        embed_model: str = "bge-m3:latest",
        base_url: str = "http://localhost:11434"
    ):
        """初始化评估器

        Args:
            rag_chain: RAG 检索链
            model_name: 使用的 Ollama LLM 模型名称
            embed_model: 使用的 Ollama Embedding 模型名称
            base_url: Ollama 服务地址
        """
        self.rag_chain = rag_chain
        self.llm = ChatOllama(model=model_name, base_url=base_url)
        self.embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url)
        self.eval_results = None

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

        # 选择指标
        metric_map = {
            "faithfulness": Faithfulness(),
            "answer_relevance": AnswerRelevancy(),
            "context_precision": ContextPrecision(),
            "context_recall": ContextRecall(),
        }

        selected_metrics = [metric_map[m] for m in metrics if m in metric_map]

        # 执行评估
        print(f"执行评估，指标: {metrics}")
        result = evaluate(
            dataset=dataset,
            metrics=selected_metrics,
            llm=self.llm,
            embeddings=self.embeddings,  # 使用本地 Ollama 嵌入模型
        )

        # 转换为 DataFrame
        df = result.to_pandas()

        # 保存结果
        self.eval_results = {
            "testset_path": testset_path,
            "metrics": metrics,
            "scores": df.to_dict(orient="records"),
            "summary": {
                metric: df[metric].mean()
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
