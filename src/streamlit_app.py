import streamlit as st
from pathlib import Path
import sys
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag import ask_question, clear_conversation_history
from src.evaluator import RAGEvaluator
from src.reporter import EvaluationReporter
import src.auth as auth

st.set_page_config(page_title="RAG", layout="wide", page_icon="")

st.markdown("""
<style>
/* 极简黑白风格 + 关键交互元素着色 */

.stApp {
    background: #ffffff;
}

/* 标题 */
.main-header {
    font-size: 2rem;
    color: #000000;
    margin-bottom: 0.25rem;
    font-weight: 700;
    letter-spacing: -0.03em;
}

/* 导航标签 */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    padding: 0;
    border-bottom: 2px solid #000000;
}

.stTabs [data-baseweb="tab"] {
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    border-radius: 0;
    color: #6b7280;
    transition: all 0.15s;
    border-bottom: 3px solid transparent;
    margin-bottom: -2px;
}

.stTabs [aria-selected="true"] {
    color: #000000;
    border-bottom: 3px solid #000000;
}

/* 消息样式 */
.user-message {
    background: #000000;
    color: #ffffff;
    margin-left: 2rem;
    padding: 1rem 1.25rem;
    border-radius: 0;
}

.user-message strong {
    color: #ffffff;
    font-weight: 600;
}

.assistant-message {
    background: #ffffff;
    color: #000000;
    margin-right: 2rem;
    padding: 1rem 1.25rem;
    border-radius: 0;
    border: 2px solid #000000;
}

.assistant-message strong {
    color: #000000;
    font-weight: 600;
}

/* 来源信息 */
.source-info {
    background: #f9fafb;
    padding: 0.75rem 1rem;
    font-size: 0.875rem;
    border: 1px solid #e5e7eb;
    font-weight: 500;
}

/* 状态 */
.status-success {
    color: #000000;
    font-size: 0.875rem;
    font-weight: 600;
}

.status-warning {
    color: #6b7280;
    font-size: 0.875rem;
    font-weight: 500;
}

/* 侧边栏 */
.stSidebar [data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 2px solid #000000;
    padding: 1.5rem 1rem 1.5rem 1.5rem;
}

/* 通用按钮 - 黑白 */
.stButton > button {
    border-radius: 0;
    font-weight: 600;
    border: 2px solid #000000;
    transition: all 0.1s;
}

.stButton > button[kind="primary"] {
    background: #000000;
    color: #ffffff;
    border: 2px solid #000000;
}

/* LOGIN/REGISTER 按钮特殊样式 - 蓝色 */
.stButton > button[kind="primary"][data-testid*="login"],
.stButton > button[kind="primary"][data-testid*="register"] {
    background: #3b82f6 !important;
    border-color: #3b82f6 !important;
}

.stButton > button[kind="primary"][data-testid*="login"]:hover,
.stButton > button[kind="primary"][data-testid*="register"]:hover {
    background: #2563eb !important;
    border-color: #2563eb !important;
    transform: translate(-1px, -1px);
    box-shadow: 3px 3px 0 #3b82f6;
}

/* 其他按钮悬停效果 */
.stButton > button:hover {
    transform: translate(-1px, -1px);
    box-shadow: 3px 3px 0 #000000;
}

/* 输入框 */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    border-radius: 0;
    border: 2px solid #000000;
    font-weight: 500;
}

/* Expander */
.streamlit-expanderHeader {
    border-radius: 0 !important;
    font-weight: 600;
    border: 1px solid #000000;
    background: #f9fafb;
}

/* 历史项 */
.history-item {
    padding: 0.75rem 1rem;
    border-radius: 0;
    margin-bottom: 0.25rem;
    font-size: 0.875rem;
    border: 1px solid transparent;
    font-weight: 500;
}

.history-item:hover {
    background: #000000;
    color: #ffffff;
    border: 1px solid #000000;
}

.history-item.active {
    background: #000000;
    color: #ffffff;
    border: 1px solid #000000;
}

/* 间距 */
div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > [data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
}

/* 分隔线 */
hr {
    border-color: #000000;
    border-width: 1px;
    opacity: 1;
}

/* 侧边栏标题 */
.stSidebar h3 {
    font-size: 0.75rem;
    font-weight: 700;
    color: #000000;
    margin: 1.5rem 0 0.75rem 0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Slider - 蓝色渐变 */
.stSlider [data-testid="stSlider"] > div > div > div {
    background: linear-gradient(to right, #3b82f6 0%, #3b82f6 var(--value, 50%), #e5e7eb var(--value, 50%), #e5e7eb 100%);
}

.stSlider [data-testid="stSliderValue"] {
    font-weight: 700;
    color: #3b82f6;
}

.stSlider [data-testid="stThumb"] {
    background: #3b82f6 !important;
    border: 2px solid #1e40af !important;
}

/* Metric */
.stMetric {
    border: 2px solid #000000;
    padding: 1rem;
    background: #ffffff;
}

.stMetric [data-testid="stMetricValue"] {
    font-weight: 700;
    color: #000000;
}

/* Dataframe */
.stDataFrame {
    border: 2px solid #000000;
}

/* Selectbox */
.stSelectbox > div > div > select {
    border-radius: 0;
    border: 2px solid #000000;
}

/* Multiselect - 为不同指标添加颜色 */
.stMultiSelect > div > div > div {
    border-radius: 0;
    border: 2px solid #000000;
}

/* 选中项的颜色标签 */
.stMultiSelect [data-baseweb="tag"] {
    border-radius: 0;
    font-weight: 600;
    font-size: 0.8rem;
}

/* 根据指标内容设置不同颜色 */
.stMultiSelect [data-baseweb="tag"] {
    background: #3b82f6 !important;
    color: #ffffff !important;
    border: none;
}
</style>
""", unsafe_allow_html=True)


def display_chat_message(role, content, sources=None, msg_index=None, used_context=None):
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>YOU</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)
    else:
        context_hint = ""
        if used_context is not None:
            if used_context:
                context_hint = '<span class="status-success">[RAG]</span>'
            else:
                context_hint = '<span class="status-warning">[GEN]</span>'

        st.markdown(f"""
        <div class="assistant-message">
            <strong>AI</strong> {content}
            <div style="margin-top: 0.5rem;">{context_hint}</div>
        </div>
        """, unsafe_allow_html=True)

    if sources and used_context:
        with st.expander(f"SOURCE ({len(sources)})", expanded=False):
            for i, source in enumerate(sources, 1):
                content_full = source.get('content', source.get('content_preview', ''))
                similarity = source.get('similarity', 0)
                
                st.markdown(f"""
                <div class="source-info">
                    <strong>{i}. {source.get('source', 'unknown')}</strong>
                    <span style="background:#000000;color:#ffffff;padding:0.125rem 0.5rem;margin-left:0.5rem;font-size:0.75rem;">
                        {similarity:.3f}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                st.text_area("", content_full, height=100, key=f"source_{msg_index}_{i}", label_visibility="collapsed")


def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_conversation_id' not in st.session_state:
        st.session_state.current_conversation_id = None
    if 'search_top_k' not in st.session_state:
        st.session_state.search_top_k = 3
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = True
    if 'eval_results' not in st.session_state:
        st.session_state.eval_results = None
    if 'eval_df' not in st.session_state:
        st.session_state.eval_df = None
    if 'eval_triggered' not in st.session_state:
        st.session_state.eval_triggered = False


def render_auth_sidebar():
    if st.session_state.logged_in:
        return True
    
    st.caption("LOGIN TO SAVE HISTORY")
    
    auth_mode = st.radio("", ["LOGIN", "REGISTER"], horizontal=True, label_visibility="collapsed")
    
    if auth_mode == "LOGIN":
        with st.form("login"):
            username = st.text_input("USERNAME", "")
            password = st.text_input("PASSWORD", "", type="password")
            submit = st.form_submit_button("LOGIN", type="primary", use_container_width=True)
            
            if submit:
                if username and password:
                    success, user_info, msg = auth.login(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_info = user_info
                        st.rerun()
                    else:
                        st.error(msg)
    else:
        with st.form("register"):
            username = st.text_input("USERNAME", "")
            password = st.text_input("PASSWORD", "", type="password")
            confirm = st.text_input("CONFIRM", "", type="password")
            submit = st.form_submit_button("REGISTER", type="primary", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.warning("REQUIRED")
                elif len(username) < 4 or len(username) > 20:
                    st.warning("USERNAME: 4-20")
                elif len(password) < 6 or len(password) > 20:
                    st.warning("PASSWORD: 6-20")
                elif password != confirm:
                    st.error("MISMATCH")
                else:
                    success, msg = auth.register(username, password)
                    if success:
                        st.success("DONE")
                    else:
                        st.error(msg)
    
    return False


def chat_page():
    st.markdown('<h1 class="main-header">RAG</h1>', unsafe_allow_html=True)
    init_session_state()

    with st.sidebar:
        # 用户信息
        if st.session_state.logged_in:
            st.markdown(f"""
            <div style="background:#000000;color:#ffffff;padding:1rem;margin-bottom:1rem;font-weight:700;">
                {st.session_state.user_info['username']}
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("LOGOUT", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.user_info = None
                st.session_state.chat_history = []
                st.session_state.current_conversation_id = None
                st.rerun()
        else:
            render_auth_sidebar()

        # 系统状态
        st.markdown("""
        <div style="border:2px solid #000000;padding:1rem;margin:0.5rem 0;font-size:0.875rem;">
            <div style="font-weight:700;">SYSTEM</div>
            <div>STATUS: ONLINE</div>
            <div>RETRIEVAL: BGE-M3</div>
            <div>LLM: QWEN-2.5</div>
        </div>
        """, unsafe_allow_html=True)

        # 参数 - 蓝色滑块
        st.session_state.search_top_k = st.slider("TOP-K", 1, 10, st.session_state.search_top_k)

        # 对话管理
        if st.session_state.logged_in:
            if st.button("NEW CHAT", type="primary", use_container_width=True):
                conv_id = auth.create_conversation(st.session_state.user_info['id'], "NEW")
                if conv_id:
                    st.session_state.current_conversation_id = conv_id
                    st.session_state.chat_history = []
                    clear_conversation_history()
                    st.rerun()
            
            # 历史
            conversations = auth.get_user_conversations(st.session_state.user_info['id'])
            
            if conversations:
                for conv in conversations[:10]:
                    updated = datetime.fromisoformat(conv['updated_at']).strftime("%m/%d")
                    title = conv['title'][:12] if conv.get('title') else "UNTITLED"
                    
                    is_active = st.session_state.current_conversation_id == conv['id']
                    
                    col_btn, col_del = st.columns([5, 1])
                    with col_btn:
                        if st.button(f"{title}", key=f"conv_{conv['id']}", use_container_width=True):
                            messages = auth.get_conversation_messages(conv['id'])
                            st.session_state.current_conversation_id = conv['id']
                            st.session_state.chat_history = []
                            for msg in messages:
                                if msg['role'] == 'user':
                                    st.session_state.chat_history.append(("user", msg['content']))
                                else:
                                    st.session_state.chat_history.append(("assistant", msg['content'], msg['sources'], msg['used_context']))
                            st.rerun()
                    with col_del:
                        if st.button("×", key=f"del_{conv['id']}", use_container_width=True):
                            if auth.delete_conversation(conv['id'], st.session_state.user_info['id']):
                                if st.session_state.current_conversation_id == conv['id']:
                                    st.session_state.current_conversation_id = None
                                    st.session_state.chat_history = []
                                st.rerun()

        if st.button("CLEAR", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_conversation_id = None
            clear_conversation_history()
            st.rerun()

    # 主内容
    for idx, msg in enumerate(st.session_state.chat_history):
        if len(msg) == 2:
            display_chat_message(msg[0], msg[1], msg_index=idx)
        elif len(msg) == 3:
            display_chat_message(msg[0], msg[1], msg[2], msg_index=idx)
        else:
            display_chat_message(msg[0], msg[1], msg[2], msg_index=idx, used_context=msg[3])

    if user_input := st.chat_input(""):
        if user_input.strip():
            with st.spinner("..."):
                result = ask_question(user_input, top_k=st.session_state.search_top_k)
                
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", result['answer'], result['source'], result['used_context']))
                
                if st.session_state.logged_in:
                    if not st.session_state.current_conversation_id:
                        conv_id = auth.create_conversation(st.session_state.user_info['id'], user_input[:30])
                        st.session_state.current_conversation_id = conv_id
                    
                    auth.save_message(st.session_state.current_conversation_id, "user", user_input)
                    auth.save_message(st.session_state.current_conversation_id, "assistant", result['answer'], result['source'], result['used_context'])
                    
                    if st.session_state.current_conversation_id:
                        auth.update_conversation_title(st.session_state.current_conversation_id, st.session_state.user_info['id'], user_input[:30])
            
            st.rerun()


def evaluation_page():
    st.markdown('<h1 class="main-header">EVALUATION</h1>', unsafe_allow_html=True)
    init_session_state()

    with st.sidebar:
        # 测试集
        testset_dir = project_root / "src"
        testset_files = [f for f in testset_dir.glob("*.json") if f.name not in ["testset_template.json", "retrieval_*.json"]]

        if testset_files:
            selected_file = st.selectbox("DATASET", options=[f.name for f in testset_files], index=0)
        else:
            st.warning("NO DATASET")
            selected_file = None

        # 指标 - 蓝色标签
        available_metrics = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
        metric_labels = {
            "faithfulness": "FAITH",
            "answer_relevance": "RELEVANCE",
            "context_precision": "PRECISION",
            "context_recall": "RECALL"
        }
        
        selected_metrics = st.multiselect(
            "METRICS",
            options=[metric_labels[m] for m in available_metrics],
            default=[metric_labels[m] for m in available_metrics]
        )
        
        reverse_labels = {v: k for k, v in metric_labels.items()}
        selected_metrics_keys = [reverse_labels[m] for m in selected_metrics]

        if st.button("RUN", type="primary", use_container_width=True):
            st.session_state.eval_triggered = True

    # 主内容
    col1, col2 = st.columns([2, 1])

    with col1:
        if selected_file:
            testset_path = testset_dir / selected_file
            with open(testset_path, 'r', encoding='utf-8') as f:
                testset_data = json.load(f)

            st.caption(f"{len(testset_data)} QUESTIONS")

            for i, item in enumerate(testset_data[:2]):
                with st.expander(f"Q{i+1}: {item['question'][:30]}...", expanded=False):
                    st.text(f"Q: {item['question']}")
                    st.text(f"A: {item.get('ground_truth', item.get('reference', 'N/A'))}")

            if len(testset_data) > 2:
                st.caption(f"+{len(testset_data) - 2}")

    with col2:
        st.caption("METRICS")
        st.text("FAITH: answer based on context")
        st.text("REL: answer relevance")
        st.text("PREC: context relevance")
        st.text("REC: context coverage")

    # 执行
    if st.session_state.eval_triggered and selected_file:
        st.session_state.eval_triggered = False
        
        if not selected_metrics_keys:
            st.error("SELECT METRICS")
        else:
            try:
                with st.spinner("..."):
                    from src.rag import create_rag_chain
                    rag_chain = create_rag_chain()

                    evaluator = RAGEvaluator(
                        rag_chain=rag_chain,
                        model_name="my-qwen25",
                        base_url="http://localhost:11434"
                    )

                    eval_results = evaluator.evaluate(
                        testset_path=str(testset_path),
                        metrics=selected_metrics_keys,
                        save_dir=str(project_root / "evaluation_results")
                    )

                    st.session_state.eval_results = eval_results
                    st.session_state.eval_df = evaluator.get_dataframe()

            except Exception as e:
                st.error(f"ERROR: {str(e)}")

    # 结果
    if st.session_state.eval_results is not None:
        st.divider()
        
        summary = st.session_state.eval_results["summary"]
        metric_names = [metric_labels.get(k, k).upper() for k in summary.keys()]
        cols = st.columns(len(summary))

        for i, (metric, score) in enumerate(summary.items()):
            label = metric_labels.get(metric, metric).upper()
            if score >= 0.8:
                delta = "OK"
                delta_color = "normal"
            elif score >= 0.6:
                delta = "WARN"
                delta_color = "off"
            else:
                delta = "FAIL"
                delta_color = "inverse"

            cols[i].metric(label, f"{score:.3f}", delta, delta_color=delta_color)

        st.divider()

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.caption("SUMMARY")
            fig_summary = go.Figure()
            fig_summary.add_trace(go.Bar(
                x=metric_names,
                y=list(summary.values()),
                marker_color='#000000',
                marker_line_color='#000000',
                marker_line_width=2,
            ))
            fig_summary.update_layout(
                yaxis=dict(range=[0, 1]),
                height=280,
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            fig_summary.update_xaxes(showline=True, linewidth=2, linecolor='#000000')
            fig_summary.update_yaxes(showline=True, linewidth=2, linecolor='#000000')
            st.plotly_chart(fig_summary, use_container_width=True)

        with col_chart2:
            st.caption("DISTRIBUTION")
            if st.session_state.eval_df is not None:
                metrics_cols = [col for col in st.session_state.eval_df.columns
                               if col not in ['user_input', 'response', 'retrieved_contexts', 'reference']]

                if metrics_cols:
                    fig_box = go.Figure()
                    for idx, metric in enumerate(metrics_cols):
                        fig_box.add_trace(go.Box(
                            y=st.session_state.eval_df[metric],
                            name=metric_labels.get(metric, metric).upper(),
                            marker_color='#000000',
                            line_color='#000000',
                            line_width=2,
                        ))
                    fig_box.update_layout(
                        height=280,
                        margin=dict(l=10, r=10, t=10, b=10),
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    fig_box.update_xaxes(showline=True, linewidth=2, linecolor='#000000')
                    fig_box.update_yaxes(showline=True, linewidth=2, linecolor='#000000')
                    st.plotly_chart(fig_box, use_container_width=True)

        st.divider()

        st.caption("DETAILS")
        if st.session_state.eval_df is not None:
            display_cols = [col for col in st.session_state.eval_df.columns
                           if col not in ['retrieved_contexts', 'reference']]
            st.dataframe(st.session_state.eval_df[display_cols], use_container_width=True, height=400)

        st.divider()

        json_str = json.dumps(st.session_state.eval_results, ensure_ascii=False, indent=2)
        st.download_button(
            "DOWNLOAD",
            data=json_str,
            file_name=f"eval_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True
        )


def main():
    init_session_state()
    
    page = st.navigation([
        st.Page(chat_page, title="CHAT"),
        st.Page(evaluation_page, title="EVAL"),
    ])
    page.run()


if __name__ == "__main__":
    main()
