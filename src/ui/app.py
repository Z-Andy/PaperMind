"""
Streamlit 前端界面：多 Agent 协作研究平台交互 UI。
"""
import streamlit as st
import requests
import sys
from pathlib import Path

# 项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 页面配置
st.set_page_config(
    page_title="多Agent协作研究平台",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API 地址
API_BASE = "http://127.0.0.1:8000"

# 预设研究领域
DOMAINS = [
    "人工智能", "大语言模型", "机器学习",
    "计算机视觉", "多智能体系统", "信息检索",
    "检索增强生成(RAG)",
]

# 预设示例问题
EXAMPLE_QUESTIONS = [
    "最近的LLM推理优化方法有哪些？帮我做一个方法论对比",
    "多Agent系统的协作策略有哪些主流方案？各自的优缺点是什么？",
    "RAG技术中，检索增强对生成质量的影响有多大？有什么量化指标？",
    "强化学习在LLM对齐中的应用有哪些最新进展？",
    "帮我梳理Transformer架构近年来的重要改进方向",
]


def init_session():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "stats" not in st.session_state:
        st.session_state.stats = None


def call_api(endpoint: str, method: str = "GET", json_data: dict = None) -> dict:
    """调用后端 API"""
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "POST":
            resp = requests.post(url, json=json_data, timeout=120)
        else:
            resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("无法连接到后端服务，请先启动 FastAPI 服务")
        return None
    except requests.exceptions.Timeout:
        st.warning("后端响应超时，系统可能正在初始化中，请稍候...")
        return None
    except Exception as e:
        st.error(f"API 调用失败: {e}")
        return None


# ---- 侧边栏 ----

def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("🔬 研究平台")
        st.markdown("---")

        # 系统状态
        st.subheader("系统状态")
        stats = call_api("/stats")
        if stats:
            data = stats.get("data", {})
            vs = data.get("vector_store", {})
            papers = data.get("papers", {})

            col1, col2 = st.columns(2)
            with col1:
                st.metric("知识库大小", f"{vs.get('total_chunks', 0)} 块")
            with col2:
                total_papers = sum(v for v in papers.values())
                st.metric("已下载论文", f"{total_papers} 篇")

            if papers:
                st.caption("各领域论文数:")
                for domain, count in papers.items():
                    st.caption(f"  {domain}: {count} 篇")

        st.markdown("---")

        # 爬取控制
        st.subheader("知识库管理")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 更新知识库", use_container_width=True):
                with st.spinner("正在爬取最新论文..."):
                    result = call_api("/crawl", "POST", {"domain": None})
                    if result:
                        st.success(result.get("result", "完成"))
                        st.rerun()

        with col2:
            if st.button("📥 论文入链", use_container_width=True):
                with st.spinner("正在将论文向量化入链..."):
                    result = call_api("/ingest", "POST")
                    if result:
                        st.success(f"入链成功: {result.get('result', {})}")
                        st.rerun()

        # 领域选择
        selected_domain = st.selectbox(
            "单独爬取领域",
            options=["全部"] + DOMAINS,
            index=0,
        )

        if selected_domain != "全部":
            if st.button(f"爬取 [{selected_domain}]", use_container_width=True):
                with st.spinner(f"正在爬取 {selected_domain}..."):
                    result = call_api("/crawl", "POST",
                                     {"domain": selected_domain})
                    if result:
                        st.success("完成")
                        st.rerun()

        st.markdown("---")

        # 定时调度
        st.subheader("定时爬虫")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ 启动定时", use_container_width=True):
                call_api("/scheduler/start", "POST")
                st.success("已启动")
        with col2:
            if st.button("⏹ 停止定时", use_container_width=True):
                call_api("/scheduler/stop", "POST")
                st.success("已停止")

        st.caption("默认每24小时自动爬取全部领域")

        st.markdown("---")
        st.caption("Multi-Agent Research Platform v1.0")


# ---- 主界面 ----

def render_main():
    """渲染主界面"""
    st.title("🤖 多Agent协作 · 学术研究助手")

    st.markdown(
        """
        <style>
        .agent-tag {
            display: inline-block;
            padding: 2px 8px;
            margin: 2px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .tag-orchestrator { background: #FFE0B2; color: #E65100; }
        .tag-retriever { background: #BBDEFB; color: #0D47A1; }
        .tag-analyst { background: #C8E6C9; color: #1B5E20; }
        .tag-reviewer { background: #F8BBD0; color: #880E4F; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Agent 协作流程说明
    with st.expander("🏗️ 协作流程", expanded=False):
        cols = st.columns(4)
        steps = [
            ("🎯 Orchestrator", "orchestrator", "任务分解\n编排调度"),
            ("🔍 Retriever", "retriever", "知识库检索\n多查询融合"),
            ("📊 Analyst", "analyst", "深度分析\n方法对比"),
            ("✅ Reviewer", "reviewer", "质量审查\n准确性校验"),
        ]
        for col, (title, tag, desc) in zip(cols, steps):
            with col:
                st.markdown(
                    f'<span class="agent-tag tag-{tag}">{title}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(desc)

    st.markdown("---")

    # 对话历史
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 输入区域
    st.markdown("---")

    # 快捷问题
    with st.expander("💡 示例问题"):
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for i, (col, q) in enumerate(zip(cols, EXAMPLE_QUESTIONS)):
            with col:
                if st.button(f"示例{i + 1}", key=f"q_{i}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()

    # 输入框
    if "pending_question" in st.session_state:
        prompt = st.session_state.pop("pending_question")
    else:
        prompt = None

    user_input = st.chat_input(
        "输入你的学术问题，多Agent将协作回答...",
        key="chat_input",
    )

    if user_input:
        prompt = user_input

    if prompt:
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用后端
        with st.chat_message("assistant"):
            with st.spinner("🤔 Agent 团队正在协作中..."):
                result = call_api(
                    "/query",
                    "POST",
                    {"question": prompt, "enable_review": True},
                )

            if result:
                answer = result.get("answer", "抱歉，处理过程中出现错误。")
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

                # 显示统计信息
                if result.get("stats"):
                    with st.expander("📈 系统运行数据"):
                        st.json(result["stats"])


# ---- 主流程 ----

def main():
    init_session()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
