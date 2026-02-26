"""生成 RAG 评估 HTML 报告"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from jinja2 import Template
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class EvaluationReporter:
    """RAG 评估报告生成器"""

    def __init__(self, eval_results: Dict[str, Any]):
        """初始化报告生成器

        Args:
            eval_results: RAGAS 评估结果
        """
        self.results = eval_results
        self.df = pd.DataFrame(eval_results["scores"])

    def _generate_summary_charts(self) -> Dict[str, str]:
        """生成摘要图表的 HTML

        Returns:
            图表 HTML 字典
        """
        charts = {}

        # 1. 指标柱状图
        if "summary" in self.results:
            summary = self.results["summary"]
            pastel_colors = ['#A8D5E5', '#FAD6A5', '#B5E5CF', '#D4A5D9']
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(summary.keys()),
                y=list(summary.values()),
                marker_color=pastel_colors[:len(summary)],
                marker_line_color='#888888',
                marker_line_width=1,
            ))
            fig.update_layout(
                title="RAG 评估指标概览",
                xaxis_title="指标",
                yaxis_title="分数 (0-1)",
                yaxis=dict(range=[0, 1]),
                height=400,
            )
            charts["summary_bar"] = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # 2. 指标分布箱线图
        metrics_cols = [col for col in self.df.columns if col not in ['user_input', 'response', 'retrieved_contexts', 'reference']]
        if metrics_cols:
            pastel_box_colors = ['#A8D5E5', '#FAD6A5', '#B5E5CF', '#D4A5D9']
            fig = go.Figure()
            for idx, metric in enumerate(metrics_cols):
                fig.add_trace(go.Box(
                    y=self.df[metric],
                    name=metric,
                    boxmean=True,
                    marker_color=pastel_box_colors[idx % len(pastel_box_colors)],
                    line_color='#888888',
                ))
            fig.update_layout(
                title="指标分布情况",
                yaxis_title="分数",
                height=500,
            )
            charts["distribution_box"] = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # 3. 各问题评分热力图
        if metrics_cols:
            # 淡雅渐变色：浅蓝 -> 浅绿 -> 浅黄
            fig = go.Figure(data=go.Heatmap(
                z=self.df[metrics_cols].T.values,
                x=[f"Q{i+1}" for i in range(len(self.df))],
                y=metrics_cols,
                colorscale=[[0, '#E3F2FD'], [0.5, '#C8E6C9'], [1, '#FFF9C4']],
                zmid=0.5,
                zmin=0,
                zmax=1,
            ))
            fig.update_layout(
                title="问题级评分热力图",
                height=400,
            )
            charts["heatmap"] = fig.to_html(full_html=False, include_plotlyjs='cdn')

        return charts

    def generate_html_report(
        self,
        output_path: Optional[str] = None,
        title: str = "RAG 系统评估报告"
    ) -> str:
        """生成 HTML 评估报告

        Args:
            output_path: 输出文件路径，默认为 evaluation_results 目录
            title: 报告标题

        Returns:
            生成的 HTML 文件路径
        """
        # 生成图表
        charts = self._generate_summary_charts()

        # 生成详细结果表格
        detail_table_rows = []
        for idx, row in self.df.iterrows():
            # 提取指标分数
            metrics_cols = [col for col in self.df.columns if col not in ['user_input', 'response', 'retrieved_contexts', 'reference']]
            metrics_html = "<br>".join([f"<b>{m}:</b> {row[m]:.3f}" for m in metrics_cols if m in row])

            detail_table_rows.append(f"""
                <tr>
                    <td>Q{idx + 1}</td>
                    <td>{row.get('user_input', '')[:100]}...</td>
                    <td>{metrics_html}</td>
                    <td><details><summary>查看答案</summary><div style="white-space: pre-wrap;">{row.get('response', '')}</div></details></td>
                </tr>
            """)

        # 汇总信息
        summary_info = f"""
            <div class="summary-card">
                <h3>评估概要</h3>
                <ul>
                    <li><b>测试集:</b> {self.results.get('testset_path', 'N/A')}</li>
                    <li><b>测试问题数:</b> {len(self.df)}</li>
                    <li><b>评估指标:</b> {', '.join(self.results.get('metrics', []))}</li>
                    <li><b>评估时间:</b> {self.results.get('timestamp', 'N/A')}</li>
                </ul>
            </div>
        """

        # 指标评分卡片
        summary_cards = ""
        if "summary" in self.results:
            for metric, score in self.results["summary"].items():
                # 根据分数设置颜色
                if score >= 0.8:
                    color_class = "score-good"
                    emoji = "✅"
                elif score >= 0.6:
                    color_class = "score-medium"
                    emoji = "⚠️"
                else:
                    color_class = "score-poor"
                    emoji = "❌"

                summary_cards += f"""
                    <div class="score-card {color_class}">
                        <div class="score-emoji">{emoji}</div>
                        <div class="score-name">{metric}</div>
                        <div class="score-value">{score:.3f}</div>
                    </div>
                """

        # HTML 模板
        html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 30px;
        }}
        .header {{
            text-align: center;
            padding-bottom: 30px;
            border-bottom: 2px solid #eee;
        }}
        .header h1 {{
            color: #333;
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            color: #666;
            margin-top: 10px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .score-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .score-card.score-good {{ border-left: 5px solid #28a745; }}
        .score-card.score-medium {{ border-left: 5px solid #ffc107; }}
        .score-card.score-poor {{ border-left: 5px solid #dc3545; }}
        .score-emoji {{ font-size: 2em; margin-bottom: 10px; }}
        .score-name {{ font-size: 0.9em; color: #666; margin-bottom: 5px; }}
        .score-value {{ font-size: 2em; font-weight: bold; color: #333; }}
        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }}
        .summary-card {{
            background: #e7f3ff;
            border-left: 4px solid #007bff;
            padding: 20px;
            border-radius: 5px;
            margin: 30px 0;
        }}
        .summary-card h3 {{
            margin-top: 0;
            color: #0056b3;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: bold;
            color: #333;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        details {{
            cursor: pointer;
        }}
        details[open] > summary {{
            margin-bottom: 10px;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #eee;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p>使用 RAGAS 框架生成的 RAG 系统评估报告</p>
        </div>

        <div class="summary">
            {summary_cards}
        </div>

        {summary_info}

        <div class="chart-grid">
            <div class="chart-container">
                {charts.get('summary_bar', '')}
            </div>
            <div class="chart-container">
                {charts.get('distribution_box', '')}
            </div>
        </div>

        <div class="chart-container">
            {charts.get('heatmap', '')}
        </div>

        <h2>详细评估结果</h2>
        <table>
            <thead>
                <tr>
                    <th width="5%">#</th>
                    <th width="30%">问题</th>
                    <th width="25%">指标评分</th>
                    <th width="40%">模型答案</th>
                </tr>
            </thead>
            <tbody>
                {''.join(detail_table_rows)}
            </tbody>
        </table>

        <div class="footer">
            <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>"""

        # 确定输出路径
        if output_path is None:
            output_dir = Path("evaluation_results")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        else:
            output_path = Path(output_path)

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)

        return str(output_path)


def generate_report_from_json(json_path: str, output_path: Optional[str] = None) -> str:
    """从 JSON 文件生成报告

    Args:
        json_path: 评估结果 JSON 文件路径
        output_path: 输出 HTML 文件路径

    Returns:
        生成的 HTML 文件路径
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        eval_results = json.load(f)

    reporter = EvaluationReporter(eval_results)
    return reporter.generate_html_report(output_path=output_path)
