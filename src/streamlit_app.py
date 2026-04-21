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
        get_available_rule_strategies,
        get_available_ml_strategies,
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

# 尝试导入 ML 策略模块
try:
    from src.ml_strategy import (
        get_available_ml_strategies as get_ml_strategies,
        create_ml_strategy_instance,
        ML_STRATEGY_TEMPLATES,
        DEFAULT_FEATURE_CONFIGS,
        should_trigger_ml,
    )
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    print(f"ML 策略模块导入失败: {e}")

# 尝试导入快照管理模块
try:
    from src.snapshot_manager import (
        get_snapshot_manager,
        list_snapshots,
        delete_snapshot,
    )
    SNAPSHOT_AVAILABLE = True
except ImportError as e:
    SNAPSHOT_AVAILABLE = False
    print(f"快照管理模块导入失败: {e}")

# 尝试导入多策略模块
try:
    from src.multi_strategy import (
        get_multi_strategy_simulator,
        list_saved_configs,
    )
    MULTI_STRATEGY_AVAILABLE = True
except ImportError as e:
    MULTI_STRATEGY_AVAILABLE = False
    print(f"多策略模块导入失败: {e}")

# 尝试导入锐思文本分析模块
try:
    from src.resset_data import (
        check_resset_available,
        get_resset_connection,
        get_cn_company_report,
        get_government_report,
        get_us_company_report,
        get_financial_news,
        get_research_report,
        get_forum_posts,
        get_real_estate_info,
        format_resset_content,
        CN_REPORT_TYPES,
        US_REPORT_TYPES,
        RESEARCH_TYPES,
        FORUM_TYPES,
        REAL_ESTATE_TYPES,
        REGION_CODES,
    )
    RESSET_AVAILABLE = True
except ImportError as e:
    RESSET_AVAILABLE = False
    print(f"锐思数据模块导入失败: {e}")

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


def display_chat_message(role, content, sources=None, msg_index=None, used_context=None, compliance=None, kg_sources=None, citation_summary=None):
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
        
        # 显示引用摘要
        if citation_summary:
            st.markdown(f"""
            <div style="margin-top:0.5rem;padding:0.5rem;background:#f8fafc;border-left:3px solid #3b82f6;border-radius:0.25rem;font-size:0.85rem;">
                {citation_summary}
            </div>
            """, unsafe_allow_html=True)

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
            submit = st.form_submit_button("LOGIN", type="primary", width='stretch')
            
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
            submit = st.form_submit_button("REGISTER", type="primary", width='stretch')
            
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
            
            if st.button("LOGOUT", width='stretch'):
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
            if st.button("NEW CHAT", type="primary", width='stretch'):
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
                        if st.button(f"{title}", key=f"conv_{conv['id']}", width='stretch'):
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
                        if st.button("×", key=f"del_{conv['id']}", width='stretch'):
                            if auth.delete_conversation(conv['id'], st.session_state.user_info['id']):
                                if st.session_state.current_conversation_id == conv['id']:
                                    st.session_state.current_conversation_id = None
                                    st.session_state.chat_history = []
                                st.rerun()

        if st.button("CLEAR", width='stretch'):
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
            display_chat_message(msg[0], msg[1], msg[2], msg_index=idx, used_context=msg[3], compliance=msg[4], kg_sources=msg[5] if msg[5] else None, citation_summary=msg[6] if len(msg) >= 7 else None)
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

                    st.session_state.chat_history.append(("assistant", result['answer'], result['source'], result['used_context'], result.get('compliance'), result.get('kg_sources', []), result.get('citation_summary')))
                    
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

        if st.button("RUN", type="primary", width='stretch'):
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
            st.plotly_chart(fig_summary, width='stretch')

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
                    st.plotly_chart(fig_box, width='stretch')

        st.divider()

        st.caption("DETAILS")
        if st.session_state.eval_df is not None:
            display_cols = [col for col in st.session_state.eval_df.columns
                           if col not in ['retrieved_contexts', 'reference']]
            st.dataframe(st.session_state.eval_df[display_cols], width='stretch', height=400)

        st.divider()

        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            json_str = json.dumps(st.session_state.eval_results, ensure_ascii=False, indent=2)
            st.download_button(
                "DOWNLOAD JSON",
                data=json_str,
                file_name=f"eval_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                width='stretch'
            )

        with col_dl2:
            html_content = st.session_state.eval_df.to_html(index=False, escape=False)
            st.download_button(
                "DOWNLOAD HTML",
                data=html_content,
                file_name=f"eval_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
                width='stretch'
            )


def quant_page():
    """量化策略回测页面（含规则策略、ML策略、快照管理、多策略模拟）"""
    st.markdown('<h1 class="main-header">QUANT</h1>', unsafe_allow_html=True)
    init_session_state()
    
    # 检查 AKQuant 是否可用
    if not QUANT_AVAILABLE:
        st.error("量化模块未安装，请确保 AKQuant 已正确安装")
        st.code("pip install -e ./akquant-main/python", language="bash")
        return
    
    # 子 Tab 布局
    tab_rule, tab_ml, tab_snapshot, tab_multi = st.tabs([
        "规则策略", "ML 策略训练", "快照管理", "多策略模拟盘"
    ])
    
    # ===========================
    # Tab 1: 规则策略（原有功能）
    # ===========================
    with tab_rule:
        with st.sidebar:
            st.markdown("### QUANT SETTINGS")
            
            available, msg = check_akquant_available()
            if available:
                st.success(f"✓ {msg}")
            else:
                st.warning(f"⚠ {msg}")
            
            initial_cash = st.number_input("INITIAL CASH", value=100000.0, min_value=10000.0, step=10000.0, key="rule_cash")
            commission = st.number_input("COMMISSION (%)", value=0.03, min_value=0.0, max_value=1.0, step=0.01, key="rule_comm") / 100
            
            st.divider()
            
            st.markdown("### STRATEGY")
            rule_strategies = get_available_rule_strategies() if QUANT_AVAILABLE else []
            strategy_names = [s["name"] for s in rule_strategies]
            
            if strategy_names:
                selected_strategy_name = st.selectbox("SELECT STRATEGY", options=strategy_names, index=0, key="rule_strategy")
                selected_strategy_id = rule_strategies[strategy_names.index(selected_strategy_name)]["id"]
                st.caption(rule_strategies[strategy_names.index(selected_strategy_name)]["description"])
            else:
                selected_strategy_id = "dual_ma"
                selected_strategy_name = "双均线策略"
            
            st.markdown("### PARAMETERS")
            strategy_params = {}
            
            if selected_strategy_id == "dual_ma":
                fast_window = st.slider("FAST WINDOW", 5, 30, 10, key="rule_fast")
                slow_window = st.slider("SLOW WINDOW", 10, 60, 30, key="rule_slow")
                strategy_params = {"fast_window": fast_window, "slow_window": slow_window}
            elif selected_strategy_id == "rsi":
                rsi_period = st.slider("RSI PERIOD", 5, 20, 14, key="rule_rsi_p")
                oversold = st.slider("OVERSOLD", 10, 40, 30, key="rule_os")
                overbought = st.slider("OVERBOUGHT", 60, 90, 70, key="rule_ob")
                strategy_params = {"rsi_period": rsi_period, "oversold": oversold, "overbought": overbought}
            elif selected_strategy_id == "macd":
                fast_period = st.slider("FAST PERIOD", 5, 20, 12, key="rule_macd_f")
                slow_period = st.slider("SLOW PERIOD", 15, 40, 26, key="rule_macd_s")
                signal_period = st.slider("SIGNAL PERIOD", 5, 15, 9, key="rule_macd_sig")
                strategy_params = {"fast_period": fast_period, "slow_period": slow_period, "signal_period": signal_period}
            elif selected_strategy_id == "bollinger":
                window = st.slider("WINDOW", 10, 30, 20, key="rule_boll_w")
                num_std = st.slider("STD MULTIPLIER", 1.0, 3.0, 2.0, 0.25, key="rule_boll_std")
                strategy_params = {"window": window, "num_std": num_std}
            
            st.divider()
            run_backtest_clicked = st.button("RUN BACKTEST", type="primary", width='stretch', key="rule_run")
        
        # 数据源
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("### DATA SOURCE")
            data_source = st.radio("SOURCE", ["AKSHARE (股票)", "模拟数据 (MOCK)"], horizontal=True, index=1, key="rule_data_src")
            symbol = "sh600000"
            
            if data_source == "AKSHARE (股票)":
                symbol = st.text_input("STOCK CODE", value="sh600000", key="rule_symbol").lower()
            
            col_start, col_end = st.columns(2)
            with col_start:
                start_date = st.date_input("START", value=datetime(2021, 1, 1), key="rule_start")
            with col_end:
                end_date = st.date_input("END", value=datetime(2023, 12, 31), key="rule_end")
            
            start_date_str = start_date.strftime("%Y%m%d")
            end_date_str = end_date.strftime("%Y%m%d")
            
            enable_benchmark = st.checkbox("启用基准对比", value=False, key="rule_bench")
        
        with col_right:
            st.markdown("### QUICK TEMPLATES")
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
                if st.button(template_name, width='stretch', key=f"tpl_{template_name}"):
                    st.session_state.quant_template = template_config
            
            if "quant_template" in st.session_state:
                template = st.session_state.quant_template
                selected_strategy_id = template["strategy"]
                strategy_params = template["params"]
                symbol = template["symbol"]
                start_date_str = template["start"]
                end_date_str = template["end"]
                st.info(f"已加载模板: {template_name}")
        
        # 执行回测
        if run_backtest_clicked:
            _execute_backtest(
                data_source, symbol, start_date_str, end_date_str,
                selected_strategy_id, strategy_params, selected_strategy_name,
                initial_cash, commission, enable_benchmark
            )
        
        with st.expander("📖 规则策略使用指南", expanded=False):
            st.markdown("""
            ### 规则策略回测
            - **双均线**: 短期均线与长期均线交叉信号
            - **RSI**: 相对强弱指数超买超卖
            - **MACD**: 指数平滑移动平均线
            - **布林带**: 价格通道突破策略
            """)
    
    # ===========================
    # Tab 2: ML 策略训练
    # ===========================
    with tab_ml:
        if not ML_AVAILABLE:
            st.warning("ML 策略模块未安装。请确保 `src/ml_strategy.py` 存在且 scikit-learn 已安装。")
            st.code("pip install scikit-learn xgboost lightgbm", language="bash")
            return
        
        st.markdown("### ML 策略 Walk-forward 训练")
        st.caption("使用机器学习模型预测涨跌，通过 Walk-forward Validation 防止未来函数泄露")
        
        col_ml_config, col_ml_data = st.columns([1, 2])
        
        with col_ml_config:
            st.markdown("#### 模型配置")
            
            ml_strategy_list = get_ml_strategies()
            ml_strategy_names = [s["name"] for s in ml_strategy_list]
            selected_ml_name = st.selectbox("选择 ML 模型", options=ml_strategy_names, index=0, key="ml_model_select")
            selected_ml_id = ml_strategy_list[ml_strategy_names.index(selected_ml_name)]["id"]
            
            st.caption(ml_strategy_list[ml_strategy_names.index(selected_ml_name)]["description"])
            
            st.divider()
            st.markdown("#### Walk-forward 参数")
            
            ml_train_window = st.number_input("训练窗口", value=50, min_value=20, max_value=500, key="ml_train_w")
            ml_test_window = st.number_input("测试窗口", value=20, min_value=5, max_value=100, key="ml_test_w")
            ml_rolling_step = st.number_input("滚动步长", value=10, min_value=1, max_value=50, key="ml_roll_step")
            
            st.divider()
            st.markdown("#### 特征配置")
            ml_feature_config = st.selectbox(
                "特征集", 
                options=list(DEFAULT_FEATURE_CONFIGS.keys()),
                index=0,
                key="ml_feature_cfg",
                help="basic: 收益率 | technical: +RSI/MACD/布林带 | extended: 全部特征"
            )
            
            # 显示特征详情
            feat_desc = {
                "basic": "ret1, ret2, ret3, ret5 (收益率特征)",
                "technical": "ret1, ret2, rsi_14, macd_diff, bollinger_pos (技术指标)",
                "extended": "全部特征: 收益率 + RSI + MACD + 布林带 + 量比 + 均线比",
            }
            st.caption(feat_desc.get(ml_feature_config, ""))
            
            # 模型特定参数
            template_info = ML_STRATEGY_TEMPLATES.get(selected_ml_id, {})
            model_type = template_info.get("model_type", "")
            
            ml_model_params = {}
            if model_type == "xgboost":
                ml_model_params["max_depth"] = st.number_input("最大深度", value=5, min_value=2, max_value=10, key="ml_xgb_depth")
                ml_model_params["n_estimators"] = st.number_input("树数量", value=100, min_value=10, max_value=500, key="ml_xgb_n")
            elif model_type == "lightgbm":
                ml_model_params["num_leaves"] = st.number_input("叶子数", value=31, min_value=10, max_value=100, key="ml_lgb_leaves")
                ml_model_params["n_estimators"] = st.number_input("树数量", value=100, min_value=10, max_value=500, key="ml_lgb_n")
            elif model_type == "random_forest":
                ml_model_params["n_estimators"] = st.number_input("树数量", value=100, min_value=10, max_value=500, key="ml_rf_n")
            elif model_type == "lstm":
                ml_model_params["hidden_dim"] = st.number_input("隐藏层维度", value=32, min_value=8, max_value=256, key="ml_lstm_hidden")
                ml_model_params["num_layers"] = st.number_input("LSTM 层数", value=2, min_value=1, max_value=4, key="ml_lstm_layers")
                ml_model_params["epochs"] = st.number_input("训练轮数", value=20, min_value=5, max_value=100, key="ml_lstm_epochs")
                ml_model_params["lr"] = st.number_input("学习率", value=0.001, min_value=0.0001, max_value=0.01, step=0.0001, format="%f", key="ml_lstm_lr")
            
            st.divider()
            
            # 数据源配置
            ml_data_source = st.radio("数据源", ["模拟数据 (MOCK)", "AKSHARE (股票)"], horizontal=True, index=0, key="ml_data_src")
            ml_symbol = "sh600000"
            if ml_data_source == "AKSHARE (股票)":
                ml_symbol = st.text_input("股票代码", value="sh600000", key="ml_symbol").lower()
            
            col_ml_start, col_ml_end = st.columns(2)
            with col_ml_start:
                ml_start_date = st.date_input("开始日期", value=datetime(2021, 1, 1), key="ml_start")
            with col_ml_end:
                ml_end_date = st.date_input("结束日期", value=datetime(2023, 12, 31), key="ml_end")
            
            ml_initial_cash = st.number_input("初始资金", value=100000.0, min_value=10000.0, key="ml_cash")
            ml_commission = st.number_input("佣金 (%)", value=0.03, min_value=0.0, max_value=1.0, step=0.01, key="ml_comm") / 100
            
            st.divider()
            
            # 快照选项
            ml_save_checkpoint = st.text_input("回测后保存快照名称（可选）", value="", placeholder="如: xgboost_phase1", key="ml_snapshot_name")
            
            run_ml_backtest = st.button("运行 ML 回测", type="primary", width='stretch', key="ml_run")
        
        with col_ml_data:
            st.markdown("#### ML 策略说明")
            
            ml_explanation = {
                "logistic_regression": "**逻辑回归**: 线性分类模型，适合特征与标签呈线性关系的场景。训练快速，不易过拟合。",
                "xgboost": "**XGBoost**: 梯度提升树，非线性模型，支持特征重要性分析。适合中等规模数据，准确率高。",
                "lightgbm": "**LightGBM**: 高效梯度提升树，训练速度比 XGBoost 更快，内存占用更少。适合大规模数据。",
                "random_forest": "**随机森林**: 集成学习，多棵决策树投票。鲁棒性强，不易过拟合，但训练较慢。",
                "lstm": "**LSTM**: 长短期记忆网络，深度学习序列模型。适合捕捉时序依赖，但训练耗时，需要 PyTorch。",
            }
            st.markdown(ml_explanation.get(model_type, ""))
            
            st.divider()
            st.markdown("#### Walk-forward Validation 原理")
            st.markdown("""
            Walk-forward 是时间序列正确的交叉验证方式：
            
            1. **训练窗口** (train_window): 使用最近 N 个 bar 训练模型
            2. **测试窗口** (test_window): 在接下来的 M 个 bar 上预测
            3. **滚动步长** (rolling_step): 每隔 K 个 bar 重新训练
            
            与传统 K-Fold 不同，Walk-forward 严格保证训练数据在测试数据之前，
            避免未来函数泄露 (look-ahead bias)。
            """)
        
        # 执行 ML 回测
        if run_ml_backtest:
            with st.spinner("运行 ML Walk-forward 回测..."):
                try:
                    # 获取数据
                    df = _load_market_data(ml_data_source, ml_symbol, ml_start_date, ml_end_date)
                    
                    if df is not None:
                        st.success(f"✓ 数据加载完成: {len(df)} bars")
                        
                        # 合并参数
                        ml_params = {
                            "train_window": ml_train_window,
                            "test_window": ml_test_window,
                            "rolling_step": ml_rolling_step,
                            "feature_config": ml_feature_config,
                            **ml_model_params,
                        }
                        
                        from src.quantitative import create_streaming_callback
                        stream_callback = create_streaming_callback()
                        
                        result = run_backtest(
                            data=df,
                            strategy_type=selected_ml_id,
                            strategy_params=ml_params,
                            symbol=ml_symbol,
                            start_date=ml_start_date.strftime("%Y%m%d"),
                            end_date=ml_end_date.strftime("%Y%m%d"),
                            initial_cash=ml_initial_cash,
                            commission_rate=ml_commission,
                            on_event=stream_callback,
                        )
                        
                        _display_backtest_result(result, selected_ml_name, ml_symbol, df)
                        
                        # 保存快照
                        if ml_save_checkpoint and result.get("success") and result.get("result_object"):
                            try:
                                from src.snapshot_manager import save_snapshot
                                res_obj = result["result_object"]
                                engine = res_obj.engine if hasattr(res_obj, 'engine') else res_obj
                                strategy = res_obj.strategy if hasattr(res_obj, 'strategy') else None
                                if strategy:
                                    path = save_snapshot(
                                        engine, strategy, ml_save_checkpoint,
                                        metadata={
                                            "strategy_type": selected_ml_id,
                                            "model_type": model_type,
                                            "train_window": ml_train_window,
                                            "test_window": ml_test_window,
                                            "symbols": [ml_symbol],
                                            "final_cash": result["metrics"].get("final_value", 0),
                                            "total_return_pct": result["metrics"].get("total_return_pct", 0),
                                        }
                                    )
                                    st.success(f"✓ 快照已保存: {ml_save_checkpoint}")
                            except Exception as e:
                                st.warning(f"快照保存失败: {e}")
                    
                except Exception as e:
                    st.error(f"ML 回测失败: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # ===========================
    # Tab 3: 快照管理
    # ===========================
    with tab_snapshot:
        if not SNAPSHOT_AVAILABLE:
            st.warning("快照管理模块未安装")
            return
        
        st.markdown("### 热启动快照管理")
        st.caption("保存回测引擎状态（持仓、订单、策略变量），在未来恢复继续运行")
        
        # 刷新
        if st.button("🔄 刷新快照列表", key="refresh_snapshots"):
            st.rerun()
        
        snapshots = list_snapshots()
        
        if not snapshots:
            st.info("暂无已保存的快照。运行 ML 回测时填写快照名称即可自动保存。")
        else:
            st.markdown(f"**共 {len(snapshots)} 个快照**")
            
            for snap in snapshots:
                with st.container():
                    col_info, col_action = st.columns([3, 1])
                    
                    with col_info:
                        name = snap.get("name", "unknown")
                        created = snap.get("created_at", "N/A")[:19]
                        strategy = snap.get("strategy_type", "N/A")
                        ret = snap.get("total_return_pct", 0)
                        size_mb = snap.get("file_size_mb", 0)
                        
                        ret_color = "green" if ret >= 0 else "red"
                        st.markdown(
                            f"**{name}** | {strategy} | "
                            f"<span style='color:{ret_color}'>{ret:.2f}%</span> | "
                            f"{size_mb:.2f}MB | {created}",
                            unsafe_allow_html=True
                        )
                    
                    with col_action:
                        col_resume, col_delete = st.columns(2)
                        
                        with col_resume:
                            if st.button("恢复", key=f"resume_{name}"):
                                st.session_state.resume_snapshot_name = name
                        
                        with col_delete:
                            if st.button("删除", key=f"delete_{name}"):
                                if delete_snapshot(name):
                                    st.success(f"已删除: {name}")
                                    st.rerun()
                                else:
                                    st.error("删除失败")
                
                st.divider()
        
        # 快照恢复面板
        if "resume_snapshot_name" in st.session_state:
            snap_name = st.session_state.resume_snapshot_name
            st.markdown(f"### 恢复快照: {snap_name}")
            
            col_res_data, col_res_config = st.columns(2)
            
            with col_res_data:
                res_data_source = st.radio("新数据源", ["模拟数据 (MOCK)", "AKSHARE (股票)"], horizontal=True, index=0, key="res_data_src")
                res_symbol = "sh600000"
                if res_data_source == "AKSHARE (股票)":
                    res_symbol = st.text_input("股票代码", value="sh600000", key="res_symbol").lower()
                
                col_res_start, col_res_end = st.columns(2)
                with col_res_start:
                    res_start_date = st.date_input("开始日期", value=datetime(2024, 1, 1), key="res_start")
                with col_res_end:
                    res_end_date = st.date_input("结束日期", value=datetime(2024, 6, 30), key="res_end")
            
            with col_res_config:
                st.markdown("#### 注意事项")
                st.warning("""
                热启动恢复需注意：
                - Instrument 需重新注册（费用配置不保存在快照中）
                - 数据连续性：Phase 2 开始时间应紧接 Phase 1 结束时间
                - 策略的 `on_start()` 中需通过 `is_restored` 避免覆盖已恢复状态
                """)
                
                res_commission = st.number_input("佣金费率", value=0.0003, format="%f", key="res_comm")
                res_stamp_tax = st.number_input("印花税率", value=0.001, format="%f", key="res_stamp")
                res_transfer = st.number_input("过户费率", value=0.00001, format="%f", key="res_transfer")
            
            if st.button("从快照恢复并运行", type="primary", key="run_resume"):
                with st.spinner("正在从快照恢复..."):
                    try:
                        df = _load_market_data(res_data_source, res_symbol, res_start_date, res_end_date)
                        
                        if df is not None:
                            from src.snapshot_manager import resume_snapshot
                            result = resume_snapshot(
                                snap_name, df,
                                symbols=res_symbol,
                                commission_rate=res_commission,
                                stamp_tax_rate=res_stamp_tax,
                                transfer_fee_rate=res_transfer,
                            )
                            
                            if result.get("success"):
                                metrics = result["metrics"]
                                st.success(f"✓ 快照恢复成功！")
                                
                                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                                with col_m1:
                                    st.metric("TOTAL RETURN", f"{metrics.get('total_return_pct', 0):.2f}%")
                                with col_m2:
                                    st.metric("SHARPE", f"{metrics.get('sharpe_ratio', 0):.2f}")
                                with col_m3:
                                    st.metric("MAX DD", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
                                with col_m4:
                                    st.metric("TRADES", metrics.get("total_trades", 0))
                            else:
                                st.error(f"恢复失败: {result.get('error', 'Unknown')}")
                    except Exception as e:
                        st.error(f"快照恢复出错: {str(e)}")
            
            if st.button("取消恢复", key="cancel_resume"):
                del st.session_state.resume_snapshot_name
                st.rerun()
    
    # ===========================
    # Tab 4: 多策略模拟盘
    # ===========================
    with tab_multi:
        if not MULTI_STRATEGY_AVAILABLE:
            st.warning("多策略模块未安装")
            return
        
        st.markdown("### 多策略模拟盘")
        st.caption("配置多个策略槽位，运行组合回测，计算跨策略指标")
        
        simulator = get_multi_strategy_simulator()
        
        col_slots, slot_result = st.columns([1, 2])
        
        with col_slots:
            st.markdown("#### 策略槽位配置")
            
            # 显示已有槽位
            current_slots = simulator.list_slots()
            if current_slots:
                for slot_data in current_slots:
                    slot_id = slot_data["slot_id"]
                    with st.expander(f"Slot: {slot_id}", expanded=False):
                        st.json(slot_data)
                        if st.button(f"移除 {slot_id}", key=f"remove_slot_{slot_id}"):
                            simulator.remove_slot(slot_id)
                            st.rerun()
            else:
                st.info("尚未添加策略槽位")
            
            st.divider()
            st.markdown("#### 添加策略槽位")
            
            new_slot_id = st.text_input("槽位 ID", value="alpha", key="new_slot_id")
            new_slot_strategy = st.selectbox(
                "策略",
                options=list(STRATEGY_TEMPLATES.keys()),
                format_func=lambda x: STRATEGY_TEMPLATES[x]["name"],
                key="new_slot_strategy"
            )
            new_slot_weight = st.number_input("资金权重", value=1.0, min_value=0.1, key="new_slot_weight")
            new_slot_max_size = st.number_input("最大下单量", value=10, min_value=1, key="new_slot_max_size")
            
            if st.button("添加槽位", key="add_slot"):
                try:
                    from src.quantitative import create_strategy_instance
                    strategy_class = create_strategy_instance(new_slot_strategy, {})
                    simulator.add_slot(
                        slot_id=new_slot_id,
                        strategy_class=strategy_class,
                        strategy_params={},
                        max_order_size=new_slot_max_size,
                        weight=new_slot_weight,
                        description=STRATEGY_TEMPLATES[new_slot_strategy]["name"],
                    )
                    st.success(f"已添加槽位: {new_slot_id}")
                    st.rerun()
                except Exception as e:
                    st.error(f"添加失败: {e}")
            
            st.divider()
            
            # 多策略回测
            st.markdown("#### 运行组合回测")
            multi_symbol = st.text_input("标的", value="sh600000", key="multi_symbol").lower()
            multi_initial_cash = st.number_input("初始资金", value=100000.0, key="multi_cash")
            
            if st.button("运行多策略回测", type="primary", key="run_multi"):
                if not current_slots:
                    st.warning("请先添加至少一个策略槽位")
                else:
                    with st.spinner("运行多策略回测..."):
                        try:
                            df = _load_market_data("模拟数据 (MOCK)", multi_symbol,
                                                   datetime(2022, 1, 1), datetime(2023, 12, 31))
                            
                            if df is not None:
                                result = simulator.run_simulation(
                                    data=df,
                                    symbols=multi_symbol,
                                    initial_cash=multi_initial_cash,
                                )
                                
                                st.session_state.multi_result = result
                        except Exception as e:
                            st.error(f"多策略回测失败: {e}")
        
        with slot_result:
            if "multi_result" in st.session_state:
                result = st.session_state.multi_result
                
                if result.get("success"):
                    metrics = result["metrics"]
                    cross = result.get("cross_slot_metrics", {})
                    
                    st.success(f"✓ 多策略回测完成 ({result.get('slot_count', 0)} 个槽位)")
                    
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        st.metric("组合收益率", f"{metrics.get('total_return_pct', 0):.2f}%")
                    with col_m2:
                        st.metric("组合夏普", f"{metrics.get('sharpe_ratio', 0):.2f}")
                    with col_m3:
                        st.metric("组合最大回撤", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
                    with col_m4:
                        st.metric("组合交易次数", metrics.get("total_trades", 0))
                    
                    # 各槽位表现
                    slot_perfs = cross.get("slot_performances", {})
                    if slot_perfs:
                        st.markdown("#### 各槽位表现")
                        perf_rows = []
                        for slot_id, perf in slot_perfs.items():
                            perf_rows.append({
                                "槽位": slot_id,
                                "收益率%": f"{perf.get('return_pct', 0):.2f}",
                                "夏普": f"{perf.get('sharpe_ratio', 0):.2f}",
                                "最大回撤%": f"{perf.get('max_drawdown_pct', 0):.2f}",
                                "胜率%": f"{perf.get('win_rate', 0):.1f}",
                            })
                        st.dataframe(pd.DataFrame(perf_rows), width='stretch', hide_index=True)
                else:
                    st.error(f"多策略回测失败: {result.get('error', 'Unknown')}")


def _load_market_data(data_source, symbol, start_date, end_date):
    """加载市场数据的辅助函数"""
    import numpy as np
    
    start_date_str = start_date.strftime("%Y%m%d") if hasattr(start_date, 'strftime') else str(start_date)
    end_date_str = end_date.strftime("%Y%m%d") if hasattr(end_date, 'strftime') else str(end_date)
    
    if data_source == "AKSHARE (股票)":
        import akshare as ak
        df = ak.stock_zh_a_daily(symbol=symbol, start_date=start_date_str, end_date=end_date_str)
        df["symbol"] = symbol
    else:
        # 生成模拟数据
        dates = pd.date_range(
            start=start_date_str[:4] + "-" + start_date_str[4:6] + "-" + start_date_str[6:],
            end=end_date_str[:4] + "-" + end_date_str[4:6] + "-" + end_date_str[6:]
        )
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
    
    return df


def _execute_backtest(data_source, symbol, start_date_str, end_date_str,
                      strategy_type, strategy_params, strategy_name,
                      initial_cash, commission, enable_benchmark):
    """执行回测的辅助函数"""
    with st.spinner("Running backtest..."):
        try:
            df = _load_market_data(data_source, symbol, start_date_str, end_date_str)
            
            if df is not None:
                benchmark_returns = None
                if enable_benchmark:
                    benchmark_returns = (df.set_index("date")["close"].pct_change().fillna(0.0).rename("BENCHMARK"))
                
                st.success(f"✓ Data loaded: {len(df)} bars")
                
                from src.quantitative import create_streaming_callback
                stream_callback = create_streaming_callback()
                
                result = run_backtest(
                    data=df,
                    strategy_type=strategy_type,
                    strategy_params=strategy_params,
                    symbol=symbol,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    initial_cash=initial_cash,
                    commission_rate=commission,
                    benchmark_data=benchmark_returns,
                    on_event=stream_callback
                )
                
                _display_backtest_result(result, strategy_name, symbol, df, benchmark_returns)
            else:
                st.error("数据加载失败")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def _display_backtest_result(result, strategy_name, symbol, df, benchmark_returns=None):
    """显示回测结果的辅助函数"""
    if result.get("success"):
        metrics = result["metrics"]
        
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
        st.dataframe(metrics_df, width='stretch', hide_index=True)
        
        # 生成报告
        st.divider()
        st.markdown("### VISUALIZATION")
        
        report_path = generate_report(
            result,
            strategy_name=strategy_name,
            symbol=symbol,
            market_data=df,
            benchmark_data=benchmark_returns
        )
        
        if report_path and Path(report_path).exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                report_html = f.read()
            st.components.v1.html(report_html, height=600, scrolling=True)
            
            with open(report_path, 'rb') as f:
                st.download_button(
                    "DOWNLOAD REPORT (HTML)",
                    data=f,
                    file_name=Path(report_path).name,
                    mime="text/html",
                    width='stretch'
                )
        else:
            st.info("报告生成中，请在命令行查看...")
            
    else:
        st.error(f"Backtest failed: {result.get('error', 'Unknown error')}")



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
            if col.button(label, width='stretch'):
                query_input = name

        if st.button("查询关系", type="primary", width='stretch') and query_input:
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
                            st.dataframe(entities_df, width='stretch', hide_index=True)

                        # 显示关系
                        if result.relations:
                            st.markdown("#### 关系网络")
                            relations_df = pd.DataFrame(result.relations)
                            if not relations_df.empty:
                                st.dataframe(relations_df, width='stretch', hide_index=True)
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

        if st.button("分析影响", type="primary", width='stretch') and event_input:
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
                            st.dataframe(entities_df, width='stretch', hide_index=True)

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

        if st.button("搜索", type="primary", width='stretch'):
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
                            st.dataframe(df, width='stretch', hide_index=True)
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

            if st.button("导入到图谱", type="primary", width='stretch', key="import_text"):
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

            if st.button("从文件夹批量导入", type="primary", width='stretch', key="import_folder"):
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

            if st.button("执行", width='stretch'):
                if cypher_input:
                    with st.spinner("执行中..."):
                        try:
                            from src.knowledge_graph import get_neo4j_connection

                            conn = get_neo4j_connection()
                            results = conn.execute_query(cypher_input)

                            if results:
                                st.success(f"返回 {len(results)} 条结果")
                                df = pd.DataFrame(results)
                                st.dataframe(df, width='stretch')
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
        refresh = st.button("刷新数据", type="primary", width='stretch')
    
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
                    
                    st.plotly_chart(fig, width='stretch')
                    
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
                        st.plotly_chart(fig_vol, width='stretch')
                    
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
                        st.plotly_chart(fig_macd, width='stretch')
                    
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
                        st.plotly_chart(fig_rsi, width='stretch')
                    
                    # 数据表格
                    with st.expander("查看数据详情", expanded=False):
                        st.dataframe(df.tail(50), width='stretch')
                        
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


def resset_page():
    """锐思文本分析数据页面"""
    st.markdown('<h1 class="main-header">RESSET</h1>', unsafe_allow_html=True)

    if not RESSET_AVAILABLE:
        st.error("锐思文本分析模块未安装")
        st.code("pip install https://rtas.resset.com/txtPath/resset-0.9.8-py3-none-any.whl", language="bash")
        return

    # 检查连接状态
    available, msg = check_resset_available()
    if available:
        st.success(f"✓ {msg}")
    else:
        st.warning(f"⚠ {msg}")
        return

    # 子 Tab 布局
    tab_cn, tab_gov, tab_us, tab_other = st.tabs([
        "中国上市公司", "政府工作报告", "美国上市公司", "资讯/研究/股吧"
    ])

    # ===========================
    # Tab 1: 中国上市公司财经文本
    # ===========================
    with tab_cn:
        st.markdown("### 中国上市公司财经文本")

        col_cfg, col_result = st.columns([1, 2])

        with col_cfg:
            st.markdown("#### 查询配置")
            cn_stock_code = st.text_input("股票代码", value="000002", key="resset_cn_code", help="6位数字，如 000002")
            cn_data_type = st.selectbox("数据类型", options=CN_REPORT_TYPES, index=0, key="resset_cn_type")
            cn_year = st.number_input("报告年份", value=2022, min_value=2001, max_value=2026, key="resset_cn_year")
            cn_content = st.radio("内容类型", ["part（剔除表格图片）", "all（全文）"], index=0, key="resset_cn_content")
            cn_content_type = "part" if "part" in cn_content else "all"

            if st.button("查询中国上市公司数据", type="primary", key="resset_cn_query"):
                with st.spinner("正在获取数据..."):
                    data = get_cn_company_report(
                        stock_code=cn_stock_code,
                        data_type=cn_data_type,
                        year=str(cn_year),
                        content_type=cn_content_type,
                    )
                    st.session_state.resset_cn_data = data

        with col_result:
            if "resset_cn_data" in st.session_state:
                data = st.session_state.resset_cn_data
                if data:
                    st.success(f"获取到 {len(data)} 条记录")
                    for i, item in enumerate(data):
                        title = item.get("title", "")
                        if isinstance(title, list):
                            title = title[0] if title else ""
                        name = item.get("name", "")
                        if isinstance(name, list):
                            name = name[0] if name else ""
                        charnum = item.get("charnum", "")
                        if isinstance(charnum, list):
                            charnum = charnum[0] if charnum else ""

                        with st.expander(f"📄 {title} ({name}, {charnum}字)", expanded=(i == 0)):
                            content = item.get("part_content") or item.get("all_content") or ""
                            if isinstance(content, list):
                                content = content[0] if content else ""
                            if content:
                                st.text_area("内容", value=content[:5000], height=300, key=f"resset_cn_content_{i}", disabled=True)
                                if len(content) > 5000:
                                    st.caption(f"... 内容共 {len(content)} 字，已截断显示")
                else:
                    st.info("未查询到数据")

    # ===========================
    # Tab 2: 政府工作报告
    # ===========================
    with tab_gov:
        st.markdown("### 政府工作报告")

        col_gov_cfg, col_gov_result = st.columns([1, 2])

        with col_gov_cfg:
            st.markdown("#### 查询配置")
            gov_region_options = list(REGION_CODES.keys())
            gov_region_selected = st.selectbox("行政区域", options=gov_region_options, index=0, key="resset_gov_region")
            gov_region_code = REGION_CODES.get(gov_region_selected, "100100")
            gov_year = st.number_input("报告年份", value=2022, min_value=1954, max_value=2026, key="resset_gov_year")

            if st.button("查询政府工作报告", type="primary", key="resset_gov_query"):
                with st.spinner("正在获取数据..."):
                    data = get_government_report(region_code=gov_region_code, year=str(gov_year))
                    st.session_state.resset_gov_data = data

        with col_gov_result:
            if "resset_gov_data" in st.session_state:
                data = st.session_state.resset_gov_data
                if data:
                    st.success(f"获取到 {len(data)} 条记录")
                    for i, item in enumerate(data):
                        title = item.get("title", "")
                        if isinstance(title, list):
                            title = title[0] if title else ""
                        with st.expander(f"📄 {title}", expanded=(i == 0)):
                            content = item.get("part_content") or item.get("all_content") or ""
                            if isinstance(content, list):
                                content = content[0] if content else ""
                            if content:
                                st.text_area("内容", value=content[:5000], height=300, key=f"resset_gov_content_{i}", disabled=True)
                                if len(content) > 5000:
                                    st.caption(f"... 内容共 {len(content)} 字，已截断显示")
                else:
                    st.info("未查询到数据")

    # ===========================
    # Tab 3: 美国上市公司
    # ===========================
    with tab_us:
        st.markdown("### 美国上市公司财经文本")

        col_us_cfg, col_us_result = st.columns([1, 2])

        with col_us_cfg:
            st.markdown("#### 查询配置")
            us_stock_code = st.text_input("美股代码", value="AMZN", key="resset_us_code", help="如 AMZN, AAPL, MSFT")
            us_data_type = st.selectbox("数据类型", options=US_REPORT_TYPES, index=0, key="resset_us_type",
                                        format_func=lambda x: {"10K": "10K - 年报", "10Q": "10Q - 季报", "424B": "424B - 招股说明书"}.get(x, x))
            us_year = st.number_input("报告年份", value=2022, min_value=1987, max_value=2026, key="resset_us_year")

            if st.button("查询美国上市公司数据", type="primary", key="resset_us_query"):
                with st.spinner("正在获取数据..."):
                    data = get_us_company_report(
                        stock_code=us_stock_code,
                        data_type=us_data_type,
                        year=str(us_year),
                    )
                    st.session_state.resset_us_data = data

        with col_us_result:
            if "resset_us_data" in st.session_state:
                data = st.session_state.resset_us_data
                if data:
                    st.success(f"获取到 {len(data)} 条记录")
                    for i, item in enumerate(data):
                        title = item.get("title", "")
                        if isinstance(title, list):
                            title = title[0] if title else ""
                        with st.expander(f"📄 {title}", expanded=(i == 0)):
                            content = item.get("part_content") or item.get("all_content") or ""
                            if isinstance(content, list):
                                content = content[0] if content else ""
                            if content:
                                st.text_area("内容", value=content[:5000], height=300, key=f"resset_us_content_{i}", disabled=True)
                else:
                    st.info("未查询到数据")

    # ===========================
    # Tab 4: 资讯/研究/股吧/房产
    # ===========================
    with tab_other:
        st.markdown("### 资讯 / 研究报告 / 股吧 / 房产")

        sub_tab_news, sub_tab_research, sub_tab_forum, sub_tab_realestate = st.tabs([
            "财经资讯", "研究报告", "股吧评论", "房产信息"
        ])

        with sub_tab_news:
            news_year = st.number_input("年份", value=2022, min_value=2017, max_value=2023, key="resset_news_year")
            if st.button("查询财经资讯", type="primary", key="resset_news_query"):
                with st.spinner("正在获取数据..."):
                    data = get_financial_news(year=str(news_year))
                    st.session_state.resset_news_data = data

            if "resset_news_data" in st.session_state:
                data = st.session_state.resset_news_data
                if data:
                    st.success(f"获取到 {len(data)} 条记录")
                    for i, item in enumerate(data[:10]):
                        title = item.get("title", "")
                        if isinstance(title, list):
                            title = title[0] if title else ""
                        with st.expander(f"📰 {title}", expanded=False):
                            content = item.get("part_content") or ""
                            if isinstance(content, list):
                                content = content[0] if content else ""
                            if content:
                                st.text(content[:3000])
                else:
                    st.info("未查询到数据")

        with sub_tab_research:
            res_type = st.selectbox("研究类型", options=RESEARCH_TYPES, index=3, key="resset_res_type")
            res_year = st.number_input("年份", value=2022, min_value=2017, max_value=2023, key="resset_res_year")
            if st.button("查询研究报告", type="primary", key="resset_res_query"):
                with st.spinner("正在获取数据..."):
                    data = get_research_report(data_type=res_type, year=str(res_year))
                    st.session_state.resset_res_data = data

            if "resset_res_data" in st.session_state:
                data = st.session_state.resset_res_data
                if data:
                    st.success(f"获取到 {len(data)} 条记录")
                    for i, item in enumerate(data[:10]):
                        info_title = item.get("InfoTitle") or item.get("title", "")
                        if isinstance(info_title, list):
                            info_title = info_title[0] if info_title else ""
                        with st.expander(f"📊 {info_title}", expanded=False):
                            content = item.get("Content") or item.get("part_content") or ""
                            if isinstance(content, list):
                                content = content[0] if content else ""
                            if content:
                                st.text(content[:3000])
                else:
                    st.info("未查询到数据")

        with sub_tab_forum:
            forum_type = st.selectbox("数据源", options=FORUM_TYPES, index=0, key="resset_forum_type")
            forum_year = st.number_input("年份", value=2022, min_value=2000, max_value=2023, key="resset_forum_year")
            forum_max = st.slider("最大获取条数", value=10, min_value=1, max_value=50, key="resset_forum_max")
            if st.button("查询股吧评论", type="primary", key="resset_forum_query"):
                with st.spinner("正在获取数据（两步获取，可能需要较长时间）..."):
                    data = get_forum_posts(data_type=forum_type, year=str(forum_year), max_items=forum_max)
                    st.session_state.resset_forum_data = data

            if "resset_forum_data" in st.session_state:
                data = st.session_state.resset_forum_data
                if data:
                    st.success(f"获取到 {len(data)} 条记录")
                    import pandas as pd
                    rows = []
                    for item in data[:50]:
                        t = item.get("title", "")
                        if isinstance(t, list):
                            t = t[0] if t else ""
                        rows.append({
                            "标题": t,
                            "发布时间": str(item.get("release_time", ""))[:10],
                            "阅读量": item.get("Reading_volume", item.get("hits", "")),
                            "评论数": item.get("Reply_num", ""),
                        })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("未查询到数据")

        with sub_tab_realestate:
            estate_type = st.selectbox("数据类型", options=REAL_ESTATE_TYPES, index=8, key="resset_estate_type")
            estate_year = st.number_input("年份", value=2022, min_value=2017, max_value=2023, key="resset_estate_year")
            if st.button("查询房产信息", type="primary", key="resset_estate_query"):
                with st.spinner("正在获取数据..."):
                    data = get_real_estate_info(data_type=estate_type, year=str(estate_year))
                    st.session_state.resset_estate_data = data

            if "resset_estate_data" in st.session_state:
                data = st.session_state.resset_estate_data
                if data:
                    st.success(f"获取到 {len(data)} 条记录")
                    for i, item in enumerate(data[:10]):
                        title = item.get("title", "")
                        with st.expander(f"🏠 {title}", expanded=False):
                            col_a, col_b = st.columns(2)
                            col_a.metric("起拍价", f"¥{item.get('starting_price', 'N/A')}")
                            col_b.metric("评估价", f"¥{item.get('valuation_price', 'N/A')}")
                            st.caption(f"位置: {item.get('location', 'N/A')} | 状态: {item.get('auction_status', 'N/A')}")
                            content = item.get("announcement") or ""
                            if content:
                                st.text(content[:2000])
                else:
                    st.info("未查询到数据")

    # 底部信息
    with st.expander("📖 锐思文本分析 API 使用说明", expanded=False):
        st.markdown("""
        ### 支持的数据类型
        | 类型 | 数据 | 年份范围 |
        |------|------|---------|
        | 中国上市公司 | 年度报告、季度报告、问询函、招股说明书等 | 2001-至今 |
        | 政府工作报告 | 国务院、各省市政府工作报告 | 1954-至今 |
        | 美国上市公司 | 10K年报、10Q季报、424B招股说明书 | 1987-至今 |
        | 财经资讯 | 新闻资讯 | 2017-2023 |
        | 研究报告 | 宏观分析、行业分析、公司研究等 | 2017-2023 |
        | 股吧评论 | 东方财富、雪球 | 2000-2023 |
        | 房产信息 | 各平台拍卖公告 | 2017-2023 |

        ### 在 CHAT 页面使用
        直接输入包含关键词的问题（如"帮我查看万科2022年年报"），系统会自动触发锐思数据获取。

        大模型也可以自主判断是否需要调用锐思 API —— 当它认为问题需要上述文本数据时，会自动发起调用。
        """)

    # ===========================
    # 数据填充 RAG 知识库
    # ===========================
    st.divider()
    st.markdown("### 将锐思数据填充到 RAG 知识库")
    st.caption("从锐思 API 获取文本数据并注入到向量数据库，使 RAG 检索可以检索到这些数据")

    ingest_col1, ingest_col2 = st.columns([1, 2])

    with ingest_col1:
        ingest_type = st.selectbox("数据类型", options=[
            "cn_report", "gov_report", "us_report",
            "financial_news", "research", "forum", "real_estate"
        ], format_func=lambda x: {
            "cn_report": "中国上市公司报告",
            "gov_report": "政府工作报告",
            "us_report": "美国上市公司报告",
            "financial_news": "财经新闻资讯",
            "research": "研究报告",
            "forum": "股吧评论",
            "real_estate": "房产拍卖信息",
        }.get(x, x), key="resset_ingest_type")

        ingest_code = st.text_input("股票/区域代码", value="000002", key="resset_ingest_code",
                                     help="中国上市公司: 6位股票代码; 政府: 区域代码; 美国: 英文代码")
        ingest_subtype = st.text_input("报告子类型", value="年度报告", key="resset_ingest_subtype",
                                        help="如: 年度报告, 10K, 行业分析, 东方财富 等")
        ingest_year = st.number_input("年份", value=2023, min_value=2000, max_value=2026, key="resset_ingest_year")

        if st.button("填充到 RAG 知识库", type="primary", key="resset_ingest_btn"):
            with st.spinner("正在获取数据并注入 RAG 知识库..."):
                try:
                    from src.rag import ingest_resset_data_to_rag
                    result = ingest_resset_data_to_rag(
                        data_type=ingest_type,
                        stock_code=ingest_code,
                        report_type=ingest_subtype,
                        year=str(ingest_year),
                        region_code=ingest_code if ingest_type == "gov_report" else "100100",
                    )
                    st.session_state.resset_ingest_result = result
                except Exception as e:
                    st.error(f"填充失败: {e}")

    with ingest_col2:
        if "resset_ingest_result" in st.session_state:
            result = st.session_state.resset_ingest_result
            if result.get("success"):
                st.success(f"✓ 成功注入 {result['documents_ingested']} 条文档到 RAG 知识库")
                st.info(f"标签: {result['label']} | 获取数据: {result['total_data_fetched']} 条")
            else:
                st.error(f"填充失败: {result.get('error', 'Unknown')}")


def main():
    init_session_state()

    page = st.navigation([
        st.Page(chat_page, title="CHAT"),
        st.Page(market_page, title="MARKET"),
        st.Page(kg_page, title="KG"),
        st.Page(resset_page, title="RESSET"),
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
