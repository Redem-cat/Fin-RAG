import streamlit as st
from pathlib import Path
import sys
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag import ask_question, clear_conversation_history
from src.evaluator import RAGEvaluator
from src.reporter import EvaluationReporter

# =========================
# 🔹 页面样式定义
# =========================
st.set_page_config(page_title="RAG 知识库问答系统", layout="wide", page_icon="📚")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&display=swap');

.main-header {
    font-size: 2.5rem;
    background: linear-gradient(135deg, #1e3a5f, #2d5a87);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 1.5rem;
    font-weight: 700;
    font-family: 'Noto Serif SC', serif;
}

.user-message {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    border-left: 4px solid #2196f3;
    margin-left: 2rem;
    padding: 0.8rem;
    border-radius: 0.8rem;
}

.assistant-message {
    background: linear-gradient(135deg, #fff8e1, #ffecb3);
    border-left: 4px solid #f57c00;
    margin-right: 2rem;
    padding: 0.8rem;
    border-radius: 0.8rem;
}

.source-info {
    background: linear-gradient(135deg, #f3e5f5, #e1bee7);
    padding: 0.8rem;
    border-radius: 0.8rem;
    margin-top: 0.8rem;
    font-size: 0.9rem;
    border: 1px solid #ce93d8;
}

.status-success { color: #2e7d32; font-weight: bold; }
.status-error { color: #d32f2f; font-weight: bold; }
.status-warning { color: #f57c00; font-weight: bold; }

.metric-card {
    background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
    padding: 1rem;
    border-radius: 0.8rem;
    text-align: center;
    margin: 0.5rem 0;
    border: 1px solid #81c784;
}
</style>
""", unsafe_allow_html=True)


# =========================
# 🔹 工具函数
# =========================
def display_chat_message(role, content, sources=None, msg_index=None, used_context=None):
    """显示用户和助手消息"""
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>🧑 您:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        # 根据是否使用上下文显示不同的提示
        context_hint = ""
        if used_context is not None:
            if used_context:
                context_hint = '<span class="status-success">✅ 使用了检索结果</span>'
            else:
                context_hint = '<span class="status-warning">⚠️ 未使用检索结果（基于模型知识回答）</span>'

        st.markdown(f"""
        <div class="assistant-message">
            <strong>📖 智能助手:</strong> {content}<br><br>
            {context_hint}
        </div>
        """, unsafe_allow_html=True)

    if sources and used_context:
        with st.expander(f"📄 参考文档片段 ({len(sources)}个)", expanded=False):
            for i, source in enumerate(sources, 1):
                similarity_color = "#4caf50" if source.get('similarity', 0) > 0.5 else "#ff9800"
                content_full = source.get('content', source.get('content_preview', ''))
                st.markdown(f"""
                <div class="source-info">
                    <strong>📄 片段 {i}: {source.get('source', 'unknown')}</strong>
                    <span style="background:{similarity_color};color:white;padding:0.2rem 0.5rem;border-radius:0.25rem;">
                        相似度: {source.get('similarity', 0):.3f}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                st.text_area(f"片段 {i} 完整内容", content_full, height=200, key=f"source_{msg_index}_{i}")


# =========================
# 🔹 初始化系统状态
# =========================
def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'search_top_k' not in st.session_state:
        st.session_state.search_top_k = 3
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = True


# =========================
# 🔹 主界面
# =========================
def main():
    st.markdown('<h1 class="main-header">📚 RAG 知识库问答系统</h1>', unsafe_allow_html=True)

    init_session_state()

    # ========== Sidebar ==========
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        st.header("📊 系统状态")
        if st.session_state.system_ready:
            st.markdown('<span class="status-success">✅ 系统已就绪</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warning">⚠️ 系统未初始化</span>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <strong>🔧 检索模型:</strong> my-bge-m3
        </div>
        <div class="metric-card">
            <strong>💬 对话模型:</strong> my-qwen25
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.header("🔧 搜索参数设置")
        st.session_state.search_top_k = st.slider("最大返回片段数", 1, 10, st.session_state.search_top_k)


    # ========== 主体内容 ==========
    st.header("💬 智能对话助手")

    # 清空按钮（放在输入框上方）
    col_clear = st.columns([1])[0]
    with col_clear:
        if st.button("🧹 清空对话", use_container_width=True):
            st.session_state.chat_history = []
            clear_conversation_history()
            st.success("✅ 对话已清空")
            st.rerun()

    # 显示聊天历史
    for idx, msg in enumerate(st.session_state.chat_history):
        if len(msg) == 2:
            display_chat_message(msg[0], msg[1], msg_index=idx)
        elif len(msg) == 3:
            display_chat_message(msg[0], msg[1], msg[2], msg_index=idx)
        else:
            # 新格式: (role, content, sources, used_context)
            display_chat_message(msg[0], msg[1], msg[2], msg_index=idx, used_context=msg[3])

    # 聊天输入框（支持回车发送）
    if user_input := st.chat_input("请输入您的问题..."):
        if user_input.strip():
            if not st.session_state.system_ready:
                st.error("⚠️ 系统尚未初始化，请检查配置。")
            else:
                with st.spinner("🤔 正在检索与生成回答..."):
                    result = ask_question(user_input, top_k=st.session_state.search_top_k)
                    st.session_state.chat_history.append(("user", user_input))
                    st.session_state.chat_history.append(("assistant", result['answer'], result['source'], result['used_context']))
                st.rerun()


# =========================
# 🔹 评估页面
# =========================
def evaluation_page():
    """独立的评估页面"""
    st.markdown('<h1 class="main-header">📊 RAG 系统评估</h1>', unsafe_allow_html=True)

    # 初始化评估状态
    if 'eval_results' not in st.session_state:
        st.session_state.eval_results = None
    if 'eval_df' not in st.session_state:
        st.session_state.eval_df = None

    # ========== 侧边栏配置 ==========
    with st.sidebar:
        st.header("⚙️ 评估配置")

        # 测试集文件选择
        testset_dir = project_root / "src"
        testset_files = list(testset_dir.glob("*.json"))
        testset_files = [f for f in testset_files if f.name not in ["testset_template.json", "retrieval_*.json"]]

        if testset_files:
            selected_file = st.selectbox(
                "选择测试集文件",
                options=[f.name for f in testset_files],
                index=0
            )
        else:
            st.warning("⚠️ 未找到测试集文件，请先创建测试集")
            selected_file = None

        st.divider()

        # 评估指标选择
        st.header("📈 评估指标")
        available_metrics = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
        selected_metrics = st.multiselect(
            "选择要评估的指标",
            options=available_metrics,
            default=available_metrics
        )

        st.divider()

        # 评估按钮
        eval_button = st.button("🚀 开始评估", use_container_width=True, type="primary")

    # ========== 主内容区 ==========
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 测试集预览")

        if selected_file:
            testset_path = testset_dir / selected_file
            with open(testset_path, 'r', encoding='utf-8') as f:
                testset_data = json.load(f)

            st.info(f"共 {len(testset_data)} 个测试问题")

            for i, item in enumerate(testset_data[:3]):  # 只显示前3个
                with st.expander(f"问题 {i+1}: {item['question'][:50]}...", expanded=False):
                    st.markdown(f"**问题:** {item['question']}")
                    st.markdown(f"**标准答案:** {item.get('ground_truth', item.get('reference', 'N/A'))}")

            if len(testset_data) > 3:
                st.caption(f"...还有 {len(testset_data) - 3} 个问题")

    with col2:
        st.subheader("📖 指标说明")

        metric_descriptions = {
            "faithfulness": "忠实度：答案是否基于检索到的上下文",
            "answer_relevance": "答案相关性：答案与问题的相关程度",
            "context_precision": "上下文精确度：检索到的片段与问题的相关程度",
            "context_recall": "上下文召回率：检索内容覆盖标准答案的程度"
        }

        for metric in selected_metrics:
            st.markdown(f"**{metric}**")
            st.caption(metric_descriptions.get(metric, ""))
            st.divider()

    # ========== 执行评估 ==========
    if eval_button and selected_file:
        if not selected_metrics:
            st.error("⚠️ 请至少选择一个评估指标")
        else:
            try:
                with st.spinner("🔄 正在执行评估..."):
                    # 获取 RAG 链（需要从 rag.py 导入）
                    from src.rag import create_rag_chain
                    rag_chain = create_rag_chain()

                    # 创建评估器
                    evaluator = RAGEvaluator(
                        rag_chain=rag_chain,
                        model_name="my-qwen25",
                        base_url="http://localhost:11434"
                    )

                    # 执行评估
                    eval_results = evaluator.evaluate(
                        testset_path=str(testset_path),
                        metrics=selected_metrics,
                        save_dir=str(project_root / "evaluation_results")
                    )

                    # 保存到会话状态
                    st.session_state.eval_results = eval_results
                    st.session_state.eval_df = evaluator.get_dataframe()

                st.success("✅ 评估完成！")

            except Exception as e:
                st.error(f"❌ 评估失败: {str(e)}")
                st.exception(e)

    # ========== 显示评估结果 ==========
    if st.session_state.eval_results is not None:
        st.divider()
        st.header("📊 评估结果")

        # 1. 摘要指标卡片
        summary = st.session_state.eval_results["summary"]
        cols = st.columns(len(summary))

        for i, (metric, score) in enumerate(summary.items()):
            # 根据分数设置颜色 - 淡雅配色
            if score >= 0.8:
                color = "#4CAF50"  # 淡绿
                icon = "✓"
            elif score >= 0.6:
                color = "#FFB74D"  # 淡橙
                icon = "◐"
            else:
                color = "#E57373"  # 淡红
                icon = "✗"

            cols[i].metric(
                label=f"{metric}\n{icon}",
                value=f"{score:.3f}",
                delta_color="normal" if score >= 0.6 else "inverse"
            )

        st.divider()

        # 2. 图表可视化
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader("📊 指标概览")
            # 淡雅配色方案
            pastel_colors = ['#A8D5E5', '#FAD6A5', '#B5E5CF', '#D4A5D9'][:len(summary)]
            fig_summary = go.Figure()
            fig_summary.add_trace(go.Bar(
                x=list(summary.keys()),
                y=list(summary.values()),
                marker_color=pastel_colors,
                marker_line_color='#888888',
                marker_line_width=1,
            ))
            fig_summary.update_layout(
                yaxis=dict(range=[0, 1]),
                height=300
            )
            st.plotly_chart(fig_summary, use_container_width=True)

        with col_chart2:
            st.subheader("📈 指标分布")
            if st.session_state.eval_df is not None:
                metrics_cols = [col for col in st.session_state.eval_df.columns
                               if col not in ['user_input', 'response', 'retrieved_contexts', 'reference']]

                if metrics_cols:
                    # 淡雅配色
                    pastel_box_colors = ['#A8D5E5', '#FAD6A5', '#B5E5CF', '#D4A5D9']
                    fig_box = go.Figure()
                    for idx, metric in enumerate(metrics_cols):
                        fig_box.add_trace(go.Box(
                            y=st.session_state.eval_df[metric],
                            name=metric,
                            boxmean=True,
                            marker_color=pastel_box_colors[idx % len(pastel_box_colors)],
                            line_color='#888888',
                        ))
                    fig_box.update_layout(height=300)
                    st.plotly_chart(fig_box, use_container_width=True)

        # 3. 热力图
        st.subheader("◐ 问题级评分热力图")
        if st.session_state.eval_df is not None:
            metrics_cols = [col for col in st.session_state.eval_df.columns
                           if col not in ['user_input', 'response', 'retrieved_contexts', 'reference']]

            if metrics_cols:
                # 淡雅渐变色：浅蓝 -> 浅绿 -> 浅黄
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=st.session_state.eval_df[metrics_cols].T.values,
                    x=[f"Q{i+1}" for i in range(len(st.session_state.eval_df))],
                    y=metrics_cols,
                    colorscale=[[0, '#E3F2FD'], [0.5, '#C8E6C9'], [1, '#FFF9C4']],
                    zmid=0.5,
                    zmin=0,
                    zmax=1,
                ))
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)

        st.divider()

        # 4. 详细结果表格
        st.subheader("📋 详细评估结果")
        if st.session_state.eval_df is not None:
            display_cols = [col for col in st.session_state.eval_df.columns
                           if col not in ['retrieved_contexts', 'reference']]
            st.dataframe(st.session_state.eval_df[display_cols], use_container_width=True)

        st.divider()

        # 5. 导出报告
        st.subheader("💾 导出报告")
        col_export_html, col_export_json = st.columns(2)

        with col_export_html:
            if st.button("📄 生成 HTML 报告", use_container_width=True):
                try:
                    reporter = EvaluationReporter(st.session_state.eval_results)
                    html_path = reporter.generate_html_report()
                    st.success(f"✅ HTML 报告已生成: {html_path}")
                except Exception as e:
                    st.error(f"❌ 生成报告失败: {str(e)}")

        with col_export_json:
            # 下载 JSON
            json_str = json.dumps(st.session_state.eval_results, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 下载 JSON 结果",
                data=json_str,
                file_name=f"eval_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )


# =========================
# 🔹 主函数
# =========================
def main():
    # 创建多页面导航
    page = st.navigation([
        st.Page(chat_page, title="对话", icon="💬"),
        st.Page(evaluation_page, title="评估", icon="📊"),
    ])

    page.run()


def chat_page():
    """对话页面（原有主界面逻辑）"""
    st.markdown('<h1 class="main-header">📚 RAG 知识库问答系统</h1>', unsafe_allow_html=True)

    init_session_state()

    # ========== Sidebar ==========
    with st.sidebar:
        st.header("⚙️ 系统配置")

        st.header("📊 系统状态")
        if st.session_state.system_ready:
            st.markdown('<span class="status-success">✅ 系统已就绪</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warning">⚠️ 系统未初始化</span>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <strong>🔧 检索模型:</strong> my-bge-m3
        </div>
        <div class="metric-card">
            <strong>💬 对话模型:</strong> my-qwen25
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.header("🔧 搜索参数设置")
        st.session_state.search_top_k = st.slider("最大返回片段数", 1, 10, st.session_state.search_top_k)

        st.divider()
        if st.button("🗑️ 清除对话历史"):
            st.session_state.chat_history = []
            clear_conversation_history()
            st.success("✅ 对话已清空")
            st.rerun()


    # ========== 主体内容 ==========
    st.header("💬 智能对话助手")

    # 聊天输入框（支持回车发送）
    if user_input := st.chat_input("请输入您的问题..."):
        if user_input.strip():
            if not st.session_state.system_ready:
                st.error("⚠️ 系统尚未初始化，请检查配置。")
            else:
                with st.spinner("🤔 正在检索与生成回答..."):
                    result = ask_question(user_input, top_k=st.session_state.search_top_k)
                    st.session_state.chat_history.append(("user", user_input))
                    st.session_state.chat_history.append(("assistant", result['answer'], result['source'], result['used_context']))
                st.rerun()

    # 清除对话历史按钮
    if st.button("🧹 清除对话历史", use_container_width=True):
        st.session_state.chat_history = []
        clear_conversation_history()
        st.success("✅ 对话已清空")
        st.rerun()

    # 显示聊天历史
    for idx, msg in enumerate(st.session_state.chat_history):
        if len(msg) == 2:
            display_chat_message(msg[0], msg[1], msg_index=idx)
        elif len(msg) == 3:
            display_chat_message(msg[0], msg[1], msg[2], msg_index=idx)
        else:
            # 新格式: (role, content, sources, used_context)
            display_chat_message(msg[0], msg[1], msg[2], msg_index=idx, used_context=msg[3])


if __name__ == "__main__":
    main()
