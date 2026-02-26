import streamlit as st
from pathlib import Path
from src.rag import ask_question, cache_manager, text_processor

# =========================
# 🔹 页面样式定义
# =========================
st.set_page_config(page_title="RAG 知识库问答系统", layout="wide")

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
def display_chat_message(role, content, sources=None):
    """显示用户和助手消息"""
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>🧑 您:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <strong>📖 智能助手:</strong> {content}
        </div>
        """, unsafe_allow_html=True)

    if sources:
        with st.expander(f"📄 参考文档片段 ({len(sources)}个)", expanded=False):
            for i, source in enumerate(sources, 1):
                similarity_color = "#4caf50" if source.get('similarity', 0) > 0.5 else "#ff9800"
                content_preview = source.get('content', source.get('content_preview', ''))
                st.markdown(f"""
                <div class="source-info">
                    <strong>📄 片段 {i}: {source.get('source', 'unknown')}</strong>
                    <span style="background:{similarity_color};color:white;padding:0.2rem 0.5rem;border-radius:0.25rem;">
                        相似度: {source.get('similarity', 0):.3f}
                    </span>
                    <br><em>📝 内容预览:</em><br>{content_preview[:150]}...
                </div>
                """, unsafe_allow_html=True)


# =========================
# 🔹 初始化系统状态
# =========================
def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'search_top_k' not in st.session_state:
        st.session_state.search_top_k = 8
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
            <strong>💬 对话模型:</strong> llama3.2:3b
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.header("🔧 搜索参数设置")
        st.session_state.search_top_k = st.slider("最大返回文档数", 3, 20, st.session_state.search_top_k)

        st.divider()
        if st.button("🗑️ 清除对话历史"):
            st.session_state.chat_history = []
            st.success("✅ 对话已清空")
            st.rerun()

        if st.button("🗑️ 清除向量缓存"):
            cache_manager.clear_cache()
            st.success("✅ 缓存已清除")


    # ========== 主体内容 ==========
    st.header("💬 智能对话助手")

    # 对话输入
    user_input = st.text_input("请输入您的问题：", placeholder="例如：Who won the Nobel Prize in Physics 2024?")
    col_send, col_clear = st.columns([1, 1])
    with col_send:
        send_clicked = st.button("🚀 发送", use_container_width=True)
    with col_clear:
        clear_clicked = st.button("🧹 清空", use_container_width=True)

    if clear_clicked:
        st.session_state.chat_history = []
        st.rerun()

    if send_clicked and user_input.strip():
        if not st.session_state.system_ready:
            st.error("⚠️ 系统尚未初始化，请检查配置。")
        else:
            with st.spinner("🤔 正在检索与生成回答..."):
                result = ask_question(user_input, top_k=st.session_state.search_top_k)
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", result['answer'], result['source']))

    # 显示聊天历史
    for msg in st.session_state.chat_history:
        if len(msg) == 2:
            display_chat_message(msg[0], msg[1])
        else:
            display_chat_message(msg[0], msg[1], msg[2])


if __name__ == "__main__":
    main()
