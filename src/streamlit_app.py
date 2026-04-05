import streamlit as st
from pathlib import Path
import sys
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag import ask_question, clear_conversation_history
from src.evaluator import RAGEvaluator
from src.reporter import EvaluationReporter
import src.auth as auth
from src.onboarding import get_onboarding_message, should_show_onboarding, mark_onboarding_complete, is_onboarding_complete, process_onboarding_response

# 尝试导入量化模块
try:
    from src.quantitative import (
        check_akquant_available,
        get_available_strategies,
        run_backtest,
        generate_report,
        parse_natural_language_strategy,
        should_trigger_quant,
        STRATEGY_TEMPLATES
    )
    QUANT_AVAILABLE = True
except ImportError as e:
    QUANT_AVAILABLE = False
    print(f"量化模块导入失败: {e}")

st.set_page_config(page_title="FinRAG-Advisor", layout="wide", page_icon="")

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
    margin-bottom: 0.5rem;
}

.user-message + .stMarkdown {
    background: #000000;
    color: #ffffff;
    margin-left: 2rem;
    padding: 0 1.25rem 1rem 1.25rem;
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
    margin-bottom: 0.5rem;
}

.assistant-message + .stMarkdown {
    background: #ffffff;
    color: #000000;
    margin-right: 2rem;
    padding: 0 1.25rem 1rem 1.25rem;
    border: 2px solid #000000;
    border-top: none;
    border-radius: 0 0 0 0;
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
    color: #3b82f6;
    background: #ffffff;
    border: 2px solid #3b82f6;
    padding: 0.125rem 0.5rem;
    margin-left: 0.5rem;
    font-size: 0.75rem;
    font-weight: 700;
    border-radius: 0;
}

.status-warning {
    color: #f59e0b;
    background: #ffffff;
    border: 2px solid #f59e0b;
    padding: 0.125rem 0.5rem;
    margin-left: 0.5rem;
    font-size: 0.75rem;
    font-weight: 700;
    border-radius: 0;
}

.status-compliance {
    color: #10b981;
    background: #ffffff;
    border: 2px solid #10b981;
    padding: 0.125rem 0.5rem;
    margin-left: 0.5rem;
    font-size: 0.75rem;
    font-weight: 700;
    border-radius: 0;
}

.status-compliance-risk {
    color: #ef4444;
    background: #ffffff;
    border: 2px solid #ef4444;
    padding: 0.125rem 0.5rem;
    margin-left: 0.5rem;
    font-size: 0.75rem;
    font-weight: 700;
    border-radius: 0;
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

/* LOGIN/REGISTER 表单按钮 - 蓝色新粗野主义 */
form[data-testid="stForm"] button[data-testid="stFormSubmitButton"] {
    background: #3b82f6 !important;
    color: #ffffff !important;
    border: 3px solid #000000 !important;
    border-radius: 0 !important;
    font-weight: 700 !important;
    box-shadow: 4px 4px 0 #000000 !important;
    transition: all 0.15s ease !important;
}

form[data-testid="stForm"] button[data-testid="stFormSubmitButton"]:hover {
    background: #2563eb !important;
    transform: translate(-2px, -2px) !important;
    box-shadow: 6px 6px 0 #000000 !important;
}

form[data-testid="stForm"] button[data-testid="stFormSubmitButton"]:active {
    transform: translate(0, 0) !important;
    box-shadow: 2px 2px 0 #000000 !important;
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

/* Slider - 蓝色新粗野主义 */
div[data-testid="stSlider"] {
    border: 2px solid #000000;
    padding: 0.5rem;
    background: #ffffff !important;
    box-shadow: 3px 3px 0 #000000;
    border-radius: 0 !important;
}

div[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(to right, #3b82f6 0%, #3b82f6 var(--progress, 50%), #e5e7eb var(--progress, 50%), #e5e7eb 100%) !important;
    height: 8px !important;
    border: 2px solid #000000 !important;
}

div[data-testid="stSlider"] div[data-testid="stNumberInput"] {
    background: #ffffff !important;
    border: 2px solid #000000 !important;
}

div[data-testid="stSlider"] div[data-testid="stNumberInput"] input {
    color: #3b82f6 !important;
    font-weight: 700 !important;
}

div[data-testid="stSlider"] [role="slider"] {
    background: #3b82f6 !important;
    border: 3px solid #000000 !important;
    box-shadow: 3px 3px 0 #000000 !important;
    width: 24px !important;
    height: 24px !important;
    border-radius: 0 !important;
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


def display_chat_message(role, content, sources=None, msg_index=None, used_context=None, compliance=None, kg_sources=None):
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>YOU</strong>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(content)
    else:
        context_hint = ""
        if used_context is not None:
            if used_context:
                context_hint = '<span class="status-success">[RAG]</span>'
            else:
                context_hint = '<span class="status-warning">[GEN]</span>'

        # 添加 KG 标识
        kg_hint = ""
        if kg_sources:
            kg_hint = '<span style="background:#8b5cf6;color:#ffffff;padding:0.125rem 0.5rem;margin-left:0.5rem;font-size:0.75rem;font-weight:700;">[KG]</span>'

        # 添加合规标识（在 RAG/GEN 右边）
        compliance_hint = ""
        if compliance:
            is_compliant = compliance.get("is_compliant")
            if is_compliant is True:
                compliance_hint = '<span class="status-compliance">[合规]</span>'
            elif is_compliant is False:
                compliance_hint = '<span class="status-compliance-risk">[风险]</span>'
            else:
                compliance_hint = '<span class="status-warning">[未知]</span>'

        st.markdown(f"""
        <div class="assistant-message">
            <strong>AI</strong> {context_hint} {kg_hint} {compliance_hint}
        </div>
        """, unsafe_allow_html=True)
        st.markdown(content)

    # 显示 KG 检索结果
    if kg_sources:
        with st.expander("KG 知识图谱检索", expanded=False):
            for kg_source in kg_sources:
                st.markdown(f"- 类型: {kg_source.get('type', 'unknown')} | 置信度: {kg_source.get('confidence', 0):.2f}")

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
                st.text_area("Source Content", content_full, height=100, key=f"source_{msg_index}_{i}", label_visibility="collapsed")


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
    if 'onboarding_shown' not in st.session_state:
        st.session_state.onboarding_shown = False
    if 'eval_df' not in st.session_state:
        st.session_state.eval_df = None
    if 'eval_triggered' not in st.session_state:
        st.session_state.eval_triggered = False


def render_auth_sidebar():
    if st.session_state.logged_in:
        return True
    
    st.caption("LOGIN TO SAVE HISTORY")
    
    auth_mode = st.radio("Auth Mode", ["LOGIN", "REGISTER"], horizontal=True, label_visibility="collapsed")
    
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
    st.markdown('<h1 class="main-header">FinRAG-Advisor</h1>', unsafe_allow_html=True)
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

    # 检查是否需要显示欢迎消息
    if (st.session_state.logged_in and 
        st.session_state.user_info and 
        not st.session_state.onboarding_shown and
        len(st.session_state.chat_history) == 0):
        
        user_id = st.session_state.user_info.get("id")
        if user_id and should_show_onboarding(user_id) and not st.session_state.onboarding_shown:
            # 显示欢迎消息
            onboarding = get_onboarding_message(user_id)
            st.session_state.chat_history.append(("assistant", onboarding["message"]))
            st.session_state.onboarding_shown = True
            st.rerun()

    # 主内容
    for idx, msg in enumerate(st.session_state.chat_history):
        if len(msg) == 2:
            display_chat_message(msg[0], msg[1], msg_index=idx)
        elif len(msg) == 3:
            display_chat_message(msg[0], msg[1], msg[2], msg_index=idx)
        elif len(msg) == 4:
            display_chat_message(msg[0], msg[1], msg[2], msg_index=idx, used_context=msg[3])
        elif len(msg) == 5:
            display_chat_message(msg[0], msg[1], msg[2], msg_index=idx, used_context=msg[3], compliance=msg[4])
        elif len(msg) >= 6:
            display_chat_message(msg[0], msg[1], msg[2], msg_index=idx, used_context=msg[3], compliance=msg[4], kg_sources=msg[5] if msg[5] else None)
        else:
            display_chat_message(msg[0], msg[1], msg[2], msg_index=idx, used_context=msg[3])

    if user_input := st.chat_input(""):
        if user_input.strip():
            st.session_state.chat_history.append(("user", user_input))
            
            # 检查是否在引导中
            in_onboarding = False
            if st.session_state.logged_in:
                user_id = st.session_state.user_info.get("id")
                if user_id and should_show_onboarding(user_id):
                    in_onboarding = True
                    # 处理引导回复
                    next_msg = process_onboarding_response(user_id, user_input)
                    st.session_state.chat_history.append(("assistant", next_msg["message"]))
                    
                    if next_msg.get("is_complete"):
                        # 引导完成，创建对话记录
                        conv_id = auth.create_conversation(user_id, "新对话")
                        st.session_state.current_conversation_id = conv_id
            else:
                in_onboarding = False
            
            # 非引导流程，正常问答
            if not in_onboarding:
                # 获取用户称呼
                user_name = None
                if st.session_state.logged_in:
                    user_id = st.session_state.user_info.get("id")
                    if user_id:
                        profile = auth.get_user_profile(user_id)
                        if profile and profile.get("display_name"):
                            user_name = profile["display_name"]

                with st.spinner("..."):
                    result = ask_question(user_input, top_k=st.session_state.search_top_k, user_name=user_name)

                    st.session_state.chat_history.append(("assistant", result['answer'], result['source'], result['used_context'], result.get('compliance'), result.get('kg_sources', [])))
                    
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
                        llm_provider="deepseek",  # 使用 DeepSeek API
                        model_name="deepseek-chat",
                        embed_provider="ollama",  # Embedding 仍用本地 Ollama
                        embed_model="my-bge-m3",
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

        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            json_str = json.dumps(st.session_state.eval_results, ensure_ascii=False, indent=2)
            st.download_button(
                "DOWNLOAD JSON",
                data=json_str,
                file_name=f"eval_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )

        with col_dl2:
            html_content = st.session_state.eval_df.to_html(index=False, escape=False)
            st.download_button(
                "DOWNLOAD HTML",
                data=html_content,
                file_name=f"eval_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
                use_container_width=True
            )


def quant_page():
    """量化策略回测页面"""
    st.markdown('<h1 class="main-header">QUANT</h1>', unsafe_allow_html=True)
    init_session_state()
    
    # 检查 AKQuant 是否可用
    if not QUANT_AVAILABLE:
        st.error("量化模块未安装，请确保 AKQuant 已正确安装")
        st.code("pip install -e ./akquant-main/python", language="bash")
        return
    
    with st.sidebar:
        st.markdown("### QUANT SETTINGS")
        
        # 检查 AKQuant 状态
        available, msg = check_akquant_available()
        if available:
            st.success(f"✓ {msg}")
        else:
            st.warning(f"⚠ {msg}")
        
        # 初始资金
        initial_cash = st.number_input("INITIAL CASH", value=100000.0, min_value=10000.0, step=10000.0)
        
        # 佣金
        commission = st.number_input("COMMISSION (%)", value=0.03, min_value=0.0, max_value=1.0, step=0.01) / 100
        
        st.divider()
        
        # 策略选择
        st.markdown("### STRATEGY")
        strategies = get_available_strategies()
        strategy_names = [s["name"] for s in strategies]
        selected_strategy_name = st.selectbox("SELECT STRATEGY", options=strategy_names, index=0)
        selected_strategy_id = strategies[strategy_names.index(selected_strategy_name)]["id"]
        
        # 策略描述
        st.caption(strategies[strategy_names.index(selected_strategy_name)]["description"])
        
        # 策略参数
        st.markdown("### PARAMETERS")
        strategy_params = {}
        
        if selected_strategy_id == "dual_ma":
            fast_window = st.slider("FAST WINDOW", 5, 30, 10)
            slow_window = st.slider("SLOW WINDOW", 10, 60, 30)
            strategy_params = {"fast_window": fast_window, "slow_window": slow_window}
            
        elif selected_strategy_id == "rsi":
            rsi_period = st.slider("RSI PERIOD", 5, 20, 14)
            oversold = st.slider("OVERSOLD", 10, 40, 30)
            overbought = st.slider("OVERBOUGHT", 60, 90, 70)
            strategy_params = {"rsi_period": rsi_period, "oversold": oversold, "overbought": overbought}
            
        elif selected_strategy_id == "macd":
            fast_period = st.slider("FAST PERIOD", 5, 20, 12)
            slow_period = st.slider("SLOW PERIOD", 15, 40, 26)
            signal_period = st.slider("SIGNAL PERIOD", 5, 15, 9)
            strategy_params = {"fast_period": fast_period, "slow_period": slow_period, "signal_period": signal_period}
            
        elif selected_strategy_id == "bollinger":
            window = st.slider("WINDOW", 10, 30, 20)
            num_std = st.slider("STD MULTIPLIER", 1.0, 3.0, 2.0, 0.25)
            strategy_params = {"window": window, "num_std": num_std}
        
        st.divider()
        
        # 回测按钮
        run_backtest_clicked = st.button("RUN BACKTEST", type="primary", use_container_width=True)
    
    # 主内容
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### DATA SOURCE")
        
        # 数据源选择
        data_source = st.radio("SOURCE", ["AKSHARE (股票)", "模拟数据 (MOCK)"], horizontal=True, index=1)
        
        symbol = "sh600000"  # 默认浦发银行
        
        if data_source == "AKSHARE (股票)":
            symbol = st.text_input("STOCK CODE (e.g. sh600000)", value="sh600000").lower()
        
        # 日期范围
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input("START", value=datetime(2021, 1, 1))
        with col_end:
            end_date = st.date_input("END", value=datetime(2023, 12, 31))
        
        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")
        
        # 基准对比选项
        enable_benchmark = st.checkbox("启用基准对比", value=False)
        benchmark_returns = None
        if enable_benchmark and data_source == "AKSHARE (股票)":
            st.caption("基准: 买入持有策略")
    
    with col_right:
        st.markdown("### QUICK TEMPLATES")
        
        # 快速模板
        templates = {
            "双均线 (10,30) + 浦发银行": {
                "strategy": "dual_ma", "params": {"fast_window": 10, "slow_window": 30},
                "symbol": "sh600000", "start": "20210101", "end": "20231231"
            },
            "RSI(14) + 贵州茅台": {
                "strategy": "rsi", "params": {"rsi_period": 14, "oversold": 30, "overbought": 70},
                "symbol": "sh600519", "start": "20220101", "end": "20231231"
            },
            "MACD + 中国平安": {
                "strategy": "macd", "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                "symbol": "sh601318", "start": "20220101", "end": "20231231"
            },
            "布林带(20,2) + 比亚迪": {
                "strategy": "bollinger", "params": {"window": 20, "num_std": 2.0},
                "symbol": "sz002594", "start": "20220101", "end": "20231231"
            }
        }
        
        for template_name, template_config in templates.items():
            if st.button(template_name, use_container_width=True):
                # 更新所有参数
                st.session_state.quant_template = template_config
        
        if "quant_template" in st.session_state:
            template = st.session_state.quant_template
            selected_strategy_id = template["strategy"]
            strategy_params = template["params"]
            symbol = template["symbol"]
            start_date_str = template["start"]
            end_date_str = template["end"]
            st.info(f"已加载: {template_name}")
    
    # 执行回测
    if run_backtest_clicked:
        with st.spinner("Running backtest..."):
            try:
                # 获取数据
                if data_source == "AKSHARE (股票)":
                    import akshare as ak
                    with st.spinner("Fetching data..."):
                        df = ak.stock_zh_a_daily(symbol=symbol, start_date=start_date_str, end_date=end_date_str)
                        df["symbol"] = symbol
                else:
                    # 生成模拟数据
                    import numpy as np
                    dates = pd.date_range(start=start_date_str[:4] + "-" + start_date_str[4:6] + "-" + start_date_str[6:],
                                         end=end_date_str[:4] + "-" + end_date_str[4:6] + "-" + end_date_str[6:])
                    n = len(dates)
                    np.random.seed(42)
                    returns = np.random.normal(0.0005, 0.02, n)
                    price = 100 * np.cumprod(1 + returns)
                    df = pd.DataFrame({
                        "date": dates,
                        "open": price,
                        "high": price * 1.01,
                        "low": price * 0.99,
                        "close": price,
                        "volume": 10000,
                        "symbol": symbol
                    })
                
                # 生成基准收益（买入持有）
                benchmark_returns = None
                if enable_benchmark:
                    benchmark_returns = (df.set_index("date")["close"].pct_change().fillna(0.0).rename("BENCHMARK"))
                
                st.success(f"✓ Data loaded: {len(df)} bars")
                
                # 创建流式回调
                from src.quantitative import create_streaming_callback
                stream_callback = create_streaming_callback()
                
                # 运行回测
                result = run_backtest(
                    data=df,
                    strategy_type=selected_strategy_id,
                    strategy_params=strategy_params,
                    symbol=symbol,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    initial_cash=initial_cash,
                    commission_rate=commission,
                    benchmark_data=benchmark_returns,
                    on_event=stream_callback
                )
                
                if result.get("success"):
                    metrics = result["metrics"]
                    
                    # 显示指标
                    st.divider()
                    st.markdown("### 📊 PERFORMANCE METRICS")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_return = metrics.get("total_return_pct", 0)
                    with col1:
                        delta_color = "normal" if total_return >= 0 else "inverse"
                        st.metric("TOTAL RETURN", f"{total_return:.2f}%", delta="↑" if total_return >= 0 else "↓", delta_color=delta_color)
                    
                    with col2:
                        st.metric("SHARPE RATIO", f"{metrics.get('sharpe_ratio', 0):.2f}")
                    
                    with col3:
                        st.metric("MAX DRAWDOWN", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
                    
                    with col4:
                        st.metric("TRADES", metrics.get("total_trades", 0))
                    
                    # 详细指标表格
                    st.markdown("#### DETAILED METRICS")
                    metrics_df = pd.DataFrame([
                        {"指标": "年化收益率", "数值": f"{metrics.get('annualized_return', 0):.2f}%"},
                        {"指标": "胜率", "数值": f"{metrics.get('win_rate', 0):.1f}%"},
                        {"指标": "夏普比率", "数值": f"{metrics.get('sharpe_ratio', 0):.2f}"},
                        {"指标": "最大回撤", "数值": f"{metrics.get('max_drawdown_pct', 0):.2f}%"},
                        {"指标": "总交易次数", "数值": metrics.get("total_trades", 0)},
                        {"指标": "最终资产", "数值": f"¥{metrics.get('final_value', 0):,.2f}"},
                    ])
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                    
                    # 生成并显示报告（支持基准对比）
                    st.divider()
                    st.markdown("### VISUALIZATION")
                    
                    report_path = generate_report(
                        result,
                        strategy_name=selected_strategy_name,
                        symbol=symbol,
                        market_data=df,
                        benchmark_data=benchmark_returns
                    )
                    
                    if report_path and Path(report_path).exists():
                        with open(report_path, 'r', encoding='utf-8') as f:
                            report_html = f.read()
                        st.components.v1.html(report_html, height=600, scrolling=True)
                        
                        # 下载报告按钮
                        with open(report_path, 'rb') as f:
                            st.download_button(
                                "DOWNLOAD REPORT (HTML)",
                                data=f,
                                file_name=Path(report_path).name,
                                mime="text/html",
                                use_container_width=True
                            )
                    else:
                        st.info("报告生成中，请在命令行查看...")
                        
                else:
                    st.error(f"Backtest failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # 使用说明
    with st.expander("📖 USAGE GUIDE", expanded=False):
        st.markdown("""
        ### 量化策略回测系统
        
        **1. 选择策略模板**
        - 双均线: 短期均线与长期均线交叉信号
        - RSI: 相对强弱指数超买超卖
        - MACD: 指数平滑移动平均线
        - 布林带: 价格通道突破策略
        
        **2. 配置参数**
        - 调整策略的周期参数
        - 设置初始资金和佣金
        
        **3. 运行回测**
        - 点击 "RUN BACKTEST" 开始回测
        - 查看绩效指标和可视化报告
        
        **4. 查看报告**
        - HTML 报告包含 K 线图和交易标记
        - 可下载保存
        """)

        st.code("pip install neo4j", language="bash")


def kg_page():
    """知识图谱页面"""
    st.markdown('<h1 class="main-header">KNOWLEDGE GRAPH</h1>', unsafe_allow_html=True)
    init_session_state()

    # 尝试导入 KG 模块
    try:
        from src.knowledge_graph import (
            KnowledgeGraphRetriever,
            get_kg_retriever,
            check_kg_status,
            EntityType,
            RelationType
        )
        KG_AVAILABLE = True
    except ImportError as e:
        KG_AVAILABLE = False
        st.error(f"知识图谱模块导入失败: {e}")

    with st.sidebar:
        st.markdown("### KG SETTINGS")

        # 检查状态
        if KG_AVAILABLE:
            status, msg = check_kg_status()
            if status:
                st.success(f"✓ {msg}")
            else:
                st.warning(f"⚠ {msg}")
        else:
            st.error("KG 模块未安装")

        st.divider()

        # 查询类型选择
        st.markdown("### QUERY TYPE")
        query_mode = st.radio(
            "SELECT MODE",
            ["关系查询", "影响分析", "实体搜索", "导入数据"],
            horizontal=False
        )

        st.divider()

        # 统计信息
        if KG_AVAILABLE and status:
            st.markdown("### STATISTICS")
            try:
                retriever = get_kg_retriever()
                # 获取统计（如果连接可用）
                st.info("图谱已连接")
            except Exception as e:
                st.warning(f"无法获取统计: {e}")

    # 主内容区域
    if not KG_AVAILABLE or not status:
        st.warning(
            "### 知识图谱未连接\n\n"
            "请确保:\n"
            "1. Neo4j 已启动 (docker run -p 7474:7474 -p 7687:7687 neo4j:5)\n"
            "2. 环境变量已配置 (.env 文件中的 NEO4J_URI, NEO4J_PASSWORD)\n"
            "3. pip install neo4j\n\n"
            "安装后重启应用。"
        )
        return

    # 查询功能
    if query_mode == "关系查询":
        st.markdown("### 关系查询")

        # 输入框
        query_input = st.text_input(
            "输入公司/行业名称",
            placeholder="例如: 苹果, 腾讯, 科技行业",
            help="查询公司或行业的关系网络"
        )

        # 快速示例
        st.markdown("**快速查询:**")
        col1, col2, col3 = st.columns(3)

        examples = {
            "苹果 Apple": "Apple",
            "微软 Microsoft": "Microsoft",
            "谷歌 Google": "Google"
        }

        for i, (label, name) in enumerate(examples.items()):
            col = [col1, col2, col3][i]
            if col.button(label, use_container_width=True):
                query_input = name

        if st.button("查询关系", type="primary", use_container_width=True) and query_input:
            with st.spinner("查询中..."):
                try:
                    retriever = get_kg_retriever()
                    result = retriever.query(f"{query_input}的关系有哪些？")

                    if result.entities:
                        st.success(f"✓ {result.explanation}")

                        # 显示实体列表
                        st.markdown("#### 相关实体")
                        entities_df = pd.DataFrame(result.entities)
                        if not entities_df.empty:
                            st.dataframe(entities_df, use_container_width=True, hide_index=True)

                        # 显示关系
                        if result.relations:
                            st.markdown("#### 关系网络")
                            relations_df = pd.DataFrame(result.relations)
                            if not relations_df.empty:
                                st.dataframe(relations_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("未找到相关结果")

                except Exception as e:
                    st.error(f"查询失败: {e}")

    elif query_mode == "影响分析":
        st.markdown("### 影响分析")

        # 事件输入
        event_input = st.text_input(
            "输入事件名称",
            placeholder="例如: 加息, 降息, 财报",
            help="分析特定事件的影响"
        )

        # 事件示例
        st.markdown("**常见事件:**")
        event_examples = ["加息", "降息", "贸易战", "财报季"]
        selected_event = st.selectbox("或选择事件", [""] + event_examples)

        if selected_event:
            event_input = selected_event

        if st.button("分析影响", type="primary", use_container_width=True) and event_input:
            with st.spinner("分析中..."):
                try:
                    retriever = get_kg_retriever()
                    result = retriever.query(f"{event_input}对哪些公司有影响？")

                    if result.entities:
                        st.success(f"✓ {result.explanation}")

                        # 显示受影响实体
                        st.markdown("#### 受影响的实体")
                        entities_df = pd.DataFrame(result.entities)
                        if not entities_df.empty:
                            st.dataframe(entities_df, use_container_width=True, hide_index=True)

                        # 可视化影响链
                        if result.paths:
                            st.markdown("#### 影响路径")
                            for i, path in enumerate(result.paths[:5], 1):
                                path_str = " → ".join([n.get("name", "") for n in path if n.get("name")])
                                if path_str:
                                    st.markdown(f"`{i}. {path_str}`")
                    else:
                        st.info("未找到影响路径")

                except Exception as e:
                    st.error(f"分析失败: {e}")

    elif query_mode == "实体搜索":
        st.markdown("### 实体搜索")

        # 搜索类型
        search_type = st.selectbox(
            "实体类型",
            ["全部", "公司 Company", "人物 Person", "行业 Sector", "事件 Event"]
        )

        type_map = {
            "全部": None,
            "公司 Company": "Company",
            "人物 Person": "Person",
            "行业 Sector": "Sector",
            "事件 Event": "Event"
        }

        # 搜索关键词
        search_input = st.text_input(
            "搜索关键词",
            placeholder="输入实体名称",
            help="搜索特定的实体"
        )

        if st.button("搜索", type="primary", use_container_width=True):
            with st.spinner("搜索中..."):
                try:
                    retriever = get_kg_retriever()

                    if search_input:
                        result = retriever.query(f"查找{search_input}相关的实体")
                    else:
                        result = retriever.query("查找金融相关的实体")

                    if result.entities:
                        # 按类型筛选
                        entities = result.entities
                        if type_map[search_type]:
                            entities = [e for e in entities if e.get("type") == type_map[search_type]]

                        st.success(f"找到 {len(entities)} 个实体")

                        if entities:
                            df = pd.DataFrame(entities)
                            st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.info("未找到相关实体")

                except Exception as e:
                    st.error(f"搜索失败: {e}")

    elif query_mode == "导入数据":
        st.markdown("### 导入数据")

        # 标签页切换
        tab_text, tab_folder, tab_cypher = st.tabs(["手动输入", "文件夹导入", "Cypher 查询"])

        with tab_text:
            st.info(
                "输入文本，LLM 将自动抽取实体和关系到知识图谱。\n\n"
                "**示例文本:**\n"
                "```\n"
                "苹果公司(Apple)是全球领先的科技公司，CEO为Tim Cook。\n"
                "苹果与微软在操作系统领域存在竞争关系。\n"
                "```"
            )

            doc_input = st.text_area(
                "输入文档内容",
                placeholder="在此输入要分析的金融文档...",
                height=200
            )

            if st.button("导入到图谱", type="primary", use_container_width=True, key="import_text"):
                if not doc_input:
                    st.warning("请输入文档内容")
                else:
                    with st.spinner("处理中..."):
                        try:
                            from src.knowledge_graph import import_from_documents

                            result = import_from_documents([doc_input])

                            if result.get("success"):
                                st.success(
                                    f"导入成功!\n\n"
                                    f"- 实体数: {result.get('total_entities', 0)}\n"
                                    f"- 关系数: {result.get('total_relations', 0)}"
                                )
                            else:
                                st.error(f"导入失败: {result.get('error', '未知错误')}")

                        except Exception as e:
                            st.error(f"导入失败: {e}")

        with tab_folder:
            kg_docs_dir = Path(__file__).parent.parent / "data" / "kg_docs"

            st.info(
                f"将 .md / .txt 文件放入 `data/kg_docs/` 文件夹，然后点击导入。\n\n"
                f"当前目录: `{kg_docs_dir}`"
            )

            # 显示已有文件
            if kg_docs_dir.exists():
                files = list(kg_docs_dir.glob("*.md")) + list(kg_docs_dir.glob("*.txt"))
                if files:
                    st.markdown(f"**已检测到 {len(files)} 个文件:**")
                    for f in sorted(files):
                        size_kb = f.stat().st_size / 1024
                        st.markdown(f"- `{f.name}` ({size_kb:.1f} KB)")
                else:
                    st.warning("文件夹中没有 .md 或 .txt 文件")
            else:
                st.warning("文件夹不存在，请创建 `data/kg_docs/` 目录")

            if st.button("从文件夹批量导入", type="primary", use_container_width=True, key="import_folder"):
                with st.spinner("正在读取文件并抽取实体..."):
                    try:
                        from src.knowledge_graph import import_from_directory

                        progress = st.progress(0, text="正在处理...")
                        result = import_from_directory(str(kg_docs_dir))
                        progress.progress(1.0, text="处理完成")

                        if result.get("success"):
                            st.success(
                                f"导入完成!\n\n"
                                f"- 处理文件: {result.get('processed_files', 0)} / {result.get('total_files', 0)}\n"
                                f"- 实体数: {result.get('total_entities', 0)}\n"
                                f"- 关系数: {result.get('total_relations', 0)}"
                            )
                            if result.get("errors"):
                                with st.expander("查看错误详情"):
                                    for err in result["errors"]:
                                        st.error(err)
                        else:
                            st.error(f"导入失败: {result.get('error', '未知错误')}")

                    except Exception as e:
                        st.error(f"导入失败: {e}")

        with tab_cypher:
            st.markdown("#### 手动 Cypher 查询")
            cypher_input = st.text_area(
                "Cypher 语句",
                placeholder="MATCH (n) RETURN n LIMIT 10",
                height=100
            )

            if st.button("执行", use_container_width=True):
                if cypher_input:
                    with st.spinner("执行中..."):
                        try:
                            from src.knowledge_graph import get_neo4j_connection

                            conn = get_neo4j_connection()
                            results = conn.execute_query(cypher_input)

                            if results:
                                st.success(f"返回 {len(results)} 条结果")
                                df = pd.DataFrame(results)
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.info("无结果")

                        except Exception as e:
                            st.error(f"查询失败: {e}")
                else:
                    st.warning("请输入 Cypher 语句")

    # 使用说明
    with st.expander("使用指南", expanded=False):
        st.markdown(
            "### 知识图谱功能\n\n"
            "**1. 关系查询**\n"
            "- 查询公司/行业的完整关系网络\n"
            "- 支持竞争对手、合作伙伴、上下游关系\n\n"
            "**2. 影响分析**\n"
            "- 分析事件对市场的影响路径\n"
            "- 例如：加息 -> 银行 -> 房地产\n\n"
            "**3. 实体搜索**\n"
            "- 按类型搜索实体\n"
            "- 支持公司、人物、行业、事件\n\n"
            "**4. 导入数据**\n"
            "- 从文本自动抽取实体和关系\n"
            "- 支持手动 Cypher 查询\n\n"
            "### 连接 Neo4j\n\n"
            "```bash\n"
            "# 启动 Neo4j\n"
            "docker run -p 7474:7474 -p 7687:7687 neo4j:5\n\n"
            "# 安装驱动\n"
            "pip install neo4j\n"
            "```"
        )


def market_page():
    """股票行情展示页面"""
    st.markdown('<h1 class="main-header">MARKET</h1>', unsafe_allow_html=True)
    init_session_state()
    
    # 检查 AKShare 是否可用
    try:
        import akshare as ak
        AKSHARE_AVAILABLE = True
    except ImportError:
        AKSHARE_AVAILABLE = False
        st.error("AKShare 未安装: pip install akshare")
        return
    
    with st.sidebar:
        st.markdown("### MARKET SETTINGS")
        
        # 常用股票/指数快捷选择
        st.markdown("**快捷选择:**")
        quick_stocks = {
            "上证指数": "sh000001",
            "深证成指": "sz399001",
            "创业板指": "sz399006",
            "沪深300": "sh000300",
            "科创50": "sh000688",
            "贵州茅台": "sh600519",
            "宁德时代": "sz300750",
            "比亚迪": "sz002594",
            "中国平安": "sh601318",
            "招商银行": "sz600036",
        }
        
        selected_quick = st.selectbox("热门标的", options=list(quick_stocks.keys()), index=0)
        default_symbol = quick_stocks[selected_quick]
        
        # 自定义股票代码
        st.markdown("**或输入代码:**")
        symbol_input = st.text_input(
            "股票代码", 
            value=default_symbol,
            placeholder="sh600000 或 600000",
            help="支持格式: sh600000, sz000001, 600000"
        ).lower().strip()
        
        # 时间周期选择
        st.markdown("**时间周期:**")
        period_options = {
            "日K": "daily",
            "周K": "weekly", 
            "月K": "monthly",
            "5分钟": "5",
            "15分钟": "15",
            "30分钟": "30",
            "60分钟": "60",
        }
        period = st.selectbox("周期", options=list(period_options.keys()), index=0)
        
        # 日期范围
        st.markdown("**日期范围:**")
        date_range = st.selectbox(
            "范围", 
            options=["近1月", "近3月", "近6月", "近1年", "近3年", "近5年"],
            index=3
        )
        
        # 技术指标
        st.markdown("**技术指标:**")
        show_ma = st.checkbox("均线 (MA)", value=True)
        show_volume = st.checkbox("成交量", value=True)
        show_macd = st.checkbox("MACD", value=False)
        show_rsi = st.checkbox("RSI", value=False)
        
        # 刷新按钮
        refresh = st.button("刷新数据", type="primary", use_container_width=True)
    
    # 解析股票代码
    def parse_symbol(code: str) -> tuple:
        """解析股票代码，返回 (symbol, is_index) 元组"""
        code = code.strip().lower()
        
        # 指数代码映射（这些是指数，不是股票）
        index_codes = {
            "sh000001": "sh000001",  # 上证指数
            "sz399001": "sz399001",  # 深证成指
            "sz399006": "sz399006",  # 创业板指
            "sh000300": "sh000300",  # 沪深300
            "sh000688": "sh000688",  # 科创50
            "sh000016": "sh000016",  # 上证50
            "sh000905": "sh000905",  # 中证500
        }
        
        if code in index_codes:
            return code, True
        
        if code.startswith(("sh", "sz")):
            # 判断是否为指数（0开头且6位数字）
            if len(code) == 8 and code[2:].isdigit():
                num = code[2:]
                # 沪深指数通常以000,399开头
                if num.startswith(("000", "399")):
                    return code, True
                return code, False
            return code, False
        
        # 自动判断（6位数字）
        if len(code) == 6 and code.isdigit():
            # 000xxx, 001xxx 通常是指数
            if code.startswith(("000", "399")):
                return f"sz{code}", True
            # 6开头是沪市
            elif code.startswith("6"):
                return f"sh{code}", False
            # 0, 3开头是深市
            elif code.startswith(("0", "3")):
                return f"sz{code}", False
        
        return code, False
    
    symbol, is_index = parse_symbol(symbol_input)
    
    # 根据日期范围计算开始日期
    from datetime import timedelta
    today = datetime.now()
    date_map = {
        "近1月": 30, "近3月": 90, "近6月": 180,
        "近1年": 365, "近3月": 1095, "近5年": 1825
    }
    days = date_map.get(date_range, 365)
    start_date = (today - timedelta(days=days)).strftime("%Y%m%d")
    end_date = today.strftime("%Y%m%d")
    
    # 主内容区域
    try:
        with st.spinner("获取行情数据..."):
            # 获取实时行情
            try:
                if symbol.startswith(("sh", "sz")):
                    df_realtime = ak.stock_zh_a_spot_em()
                    stock_info = df_realtime[df_realtime['代码'] == symbol[2:]].iloc[0]
                    
                    # 实时行情卡片
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    price = float(stock_info['最新价'])
                    change = float(stock_info['涨跌幅'])
                    volume = float(stock_info['成交量'])
                    amount = float(stock_info['成交额'])
                    high = float(stock_info['最高'])
                    low = float(stock_info['最低'])
                    open_price = float(stock_info['今开'])
                    prev_close = float(stock_info['昨收'])
                    
                    with col1:
                        st.metric("最新价", f"¥{price:.2f}", f"{change:+.2f}%")
                    with col2:
                        st.metric("最高", f"¥{high:.2f}")
                    with col3:
                        st.metric("最低", f"¥{low:.2f}")
                    with col4:
                        st.metric("成交量", f"{volume/10000:.2f}万")
                    with col5:
                        st.metric("成交额", f"¥{amount/100000000:.2f}亿")
                    
                    st.divider()
                    
                    # 详细信息
                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        st.text(f"今开: ¥{open_price:.2f}")
                        st.text(f"昨收: ¥{prev_close:.2f}")
                    with col_info2:
                        st.text(f"涨停: ¥{prev_close*1.1:.2f}")
                        st.text(f"跌停: ¥{prev_close*0.9:.2f}")
                    with col_info3:
                        st.text(f"市值: {stock_info.get('总市值', 'N/A')}")
                        st.text(f"流通市值: {stock_info.get('流通市值', 'N/A')}")
                        
            except Exception as e:
                st.warning(f"实时行情获取失败: {e}")
            
            # 获取历史K线数据
            with st.spinner("加载K线数据..."):
                try:
                    if period in ["daily", "weekly", "monthly"]:
                        # 根据是否为指数选择不同的API
                        if is_index:
                            # 指数使用专用接口
                            df = ak.stock_zh_index_daily(symbol=symbol)
                            # 过滤日期范围
                            df['date'] = pd.to_datetime(df['date'])
                            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                        else:
                            df = ak.stock_zh_a_daily(symbol=symbol, start_date=start_date, end_date=end_date)
                    else:
                        # 分钟级数据（仅交易日可用）
                        try:
                            if is_index:
                                st.warning("指数暂无分钟级数据，已自动切换为日线数据。")
                                df = ak.stock_zh_index_daily(symbol=symbol)
                                df['date'] = pd.to_datetime(df['date'])
                                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                            else:
                                df = ak.stock_zh_a_minute(symbol=symbol, period=period, adjust="qfq")
                        except (ValueError, Exception) as e:
                            st.warning(f"分钟K线数据获取失败: {str(e)[:50]}... 已自动切换为日线数据。")
                            try:
                                if is_index:
                                    df = ak.stock_zh_index_daily(symbol=symbol)
                                    df['date'] = pd.to_datetime(df['date'])
                                    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                                else:
                                    df = ak.stock_zh_a_daily(symbol=symbol, start_date=start_date, end_date=end_date)
                            except Exception as daily_error:
                                st.error(f"日线数据获取也失败: {str(daily_error)[:100]}")
                                return
                    
                    if df is None or df.empty:
                        st.error("获取数据失败，请检查股票代码")
                        return
                    
                    # 数据预处理
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    elif '日期' in df.columns:
                        df['date'] = pd.to_datetime(df['日期'])
                        df = df.drop('日期', axis=1)
                    
                    # 统一列名
                    df.columns = [c.lower() if c != 'date' else 'date' for c in df.columns]
                    
                    # K线图表
                    st.markdown("### K-LINE CHART")
                    
                    fig = go.Figure()
                    
                    # K线（蜡烛图）
                    fig.add_trace(go.Candlestick(
                        x=df['date'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name="K线",
                        increasing_line_color='#ef4444',
                        decreasing_line_color='#22c55e'
                    ))
                    
                    # 添加均线
                    if show_ma and 'close' in df.columns:
                        df['ma5'] = df['close'].rolling(window=5).mean()
                        df['ma10'] = df['close'].rolling(window=10).mean()
                        df['ma20'] = df['close'].rolling(window=20).mean()
                        
                        fig.add_trace(go.Scatter(
                            x=df['date'], y=df['ma5'], name="MA5",
                            line=dict(color='#f59e0b', width=1)
                        ))
                        fig.add_trace(go.Scatter(
                            x=df['date'], y=df['ma10'], name="MA10",
                            line=dict(color='#8b5cf6', width=1)
                        ))
                        fig.add_trace(go.Scatter(
                            x=df['date'], y=df['ma20'], name="MA20",
                            line=dict(color='#3b82f6', width=1)
                        ))
                    
                    # 设置布局
                    fig.update_layout(
                        height=500,
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=10, r=10, t=30, b=10),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        xaxis=dict(
                            showgrid=True, gridcolor='#f3f4f6',
                            showline=True, linewidth=1, linecolor='#000'
                        ),
                        yaxis=dict(
                            showgrid=True, gridcolor='#f3f4f6',
                            showline=True, linewidth=1, linecolor='#000'
                        ),
                        legend=dict(
                            orientation="h", yanchor="bottom",
                            y=1.02, xanchor="right", x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 成交量图表
                    if show_volume and 'volume' in df.columns:
                        st.markdown("### VOLUME")
                        
                        colors = ['#ef4444' if df['close'].iloc[i] >= df['open'].iloc[i] else '#22c55e' 
                                 for i in range(len(df))]
                        
                        fig_vol = go.Figure()
                        fig_vol.add_trace(go.Bar(
                            x=df['date'], y=df['volume'], marker_color=colors, name="成交量"
                        ))
                        fig_vol.update_layout(
                            height=200,
                            xaxis_rangeslider_visible=False,
                            margin=dict(l=10, r=10, t=10, b=10),
                            plot_bgcolor='white',
                            xaxis=dict(showgrid=True, gridcolor='#f3f4f6'),
                            yaxis=dict(showgrid=True, gridcolor='#f3f4f6')
                        )
                        st.plotly_chart(fig_vol, use_container_width=True)
                    
                    # MACD 指标
                    if show_macd and 'close' in df.columns:
                        st.markdown("### MACD")
                        
                        # 计算 MACD
                        df['ema12'] = df['close'].ewm(span=12).mean()
                        df['ema26'] = df['close'].ewm(span=26).mean()
                        df['dif'] = df['ema12'] - df['ema26']
                        df['dea'] = df['dif'].ewm(span=9).mean()
                        df['macd'] = (df['dif'] - df['dea']) * 2
                        
                        fig_macd = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                               vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
                        
                        # K线
                        fig_macd.add_trace(go.Candlestick(
                            x=df['date'], open=df['open'], high=df['high'],
                            low=df['low'], close=df['close'], name="K线"
                        ), row=1, col=1)
                        
                        # DIF 和 DEA
                        fig_macd.add_trace(go.Scatter(x=df['date'], y=df['dif'], name="DIF", line=dict(color='#3b82f6', width=1)), row=2, col=1)
                        fig_macd.add_trace(go.Scatter(x=df['date'], y=df['dea'], name="DEA", line=dict(color='#f59e0b', width=1)), row=2, col=1)
                        
                        # MACD 柱
                        colors_macd = ['#ef4444' if v >= 0 else '#22c55e' for v in df['macd']]
                        fig_macd.add_trace(go.Bar(x=df['date'], y=df['macd'], marker_color=colors_macd, name="MACD"), row=3, col=1)
                        
                        fig_macd.update_layout(height=400, showlegend=True, margin=dict(l=10, r=10, t=10, b=10))
                        st.plotly_chart(fig_macd, use_container_width=True)
                    
                    # RSI 指标
                    if show_rsi and 'close' in df.columns:
                        st.markdown("### RSI")
                        
                        # 计算 RSI
                        delta = df['close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        df['rsi'] = 100 - (100 / (1 + rs))
                        
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(
                            x=df['date'], y=df['rsi'], name="RSI(14)",
                            line=dict(color='#8b5cf6', width=2)
                        ))
                        
                        # 超买超卖线
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="超买")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#22c55e", annotation_text="超卖")
                        
                        fig_rsi.update_layout(
                            height=200, yaxis=dict(range=[0, 100]),
                            margin=dict(l=10, r=10, t=10, b=10)
                        )
                        st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    # 数据表格
                    with st.expander("查看数据详情", expanded=False):
                        st.dataframe(df.tail(50), use_container_width=True)
                        
                        # 下载数据
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "下载CSV",
                            data=csv,
                            file_name=f"{symbol}_kline.csv",
                            mime="text/csv"
                        )
                    
                except Exception as e:
                    st.error(f"加载失败: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    except Exception as market_error:
        st.error(f"市场页面加载失败: {str(market_error)}")
        import traceback
        st.code(traceback.format_exc())


def main():
    init_session_state()

    page = st.navigation([
        st.Page(chat_page, title="CHAT"),
        st.Page(market_page, title="MARKET"),
        st.Page(kg_page, title="KG"),
        st.Page(quant_page, title="QUANT"),
        st.Page(evaluation_page, title="EVAL"),
    ])
    page.run()

    # 右下角版权信息
    st.markdown(
        "<style>\n"
        ".footer-text {\n"
        "    position: fixed;\n"
        "    bottom: 10px;\n"
        "    right: 20px;\n"
        "    color: #9ca3af;\n"
        "    font-size: 0.75rem;\n"
        "    font-weight: 400;\n"
        "    z-index: 9999;\n"
        "}\n"
        "</style>\n"
        "<div class=\"footer-text\">本科毕设 - 刘宗奇</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
