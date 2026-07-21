"""
CrewAI 多 Agent 协作系统：替代手写 Agent 流程。
使用 CrewAI 框架管理 Agent 角色、任务编排和协作。
"""
import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from crewai import Agent, Task, Crew, Process, LLM

from src.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, CONVERSATIONS_DIR,
)
from src.crawler import ArxivCrawler
from src.crawler.scheduler import CrawlScheduler
from src.metrics import get_metrics
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import Retriever
from src.agents.tools import RetrievePapersTool

logger = logging.getLogger(__name__)

# 查询缓存最大条目数
_MAX_CACHE_SIZE = 64


class CrewMultiAgentSystem:
    """基于 CrewAI 的多 Agent 协作系统"""

    def __init__(self):
        # ---- 基础组件（不变） ----
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.rag_pipeline = RAGPipeline(self.vector_store, self.embedder)
        self.retriever = Retriever(self.vector_store, self.embedder)
        self.crawler = ArxivCrawler()

        # ---- 预加载重排序模型（避免首次查询加载延迟） ----
        logger.info("预加载 Cross-Encoder 重排序模型...")
        self.retriever.preload_models()

        # ---- CrewAI LLM ----
        self.llm = LLM(
            model=f"openai/{LLM_MODEL}",
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
        )

        # ---- 工具（Retriever Agent 保留用于兼容，但流式查询不再走 LLM 检索） ----
        self.retrieve_tool = RetrievePapersTool(retriever=self.retriever)

        # ---- 创建 Crew ----
        self._build_crew()

        # ---- 爬虫调度（不变） ----
        self.scheduler = CrawlScheduler(
            crawl_func=self._auto_crawl_and_ingest
        )

        # ---- 多轮对话存储 ----
        self._conversations: dict[str, list[dict]] = {}

        # ---- 查询结果缓存 ----
        self._query_cache: OrderedDict[str, str] = OrderedDict()

        logger.info("CrewAI 多 Agent 协作系统初始化完成")

    def check_health(self) -> dict:
        """
        启动健康检查：验证 LLM API 和知识库状态。

        Returns:
            {"llm": true/false/"unknown", "kb_size": int, "issues": [...]}
        """
        issues = []

        # 1. 知识库检查
        kb_size = self.vector_store.get_count()

        # 2. LLM API 可用性检查（轻量测试）
        llm_ok = None
        try:
            client = __import__('openai').OpenAI(
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
            )
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                timeout=15,
            )
            llm_ok = bool(resp.choices[0].message.content)
        except Exception as e:
            llm_ok = False
            issues.append(f"LLM API 连接失败: {str(e)[:100]}")

        logger.info(
            f"健康检查: LLM={'✅' if llm_ok else '❌'}, "
            f"知识库={kb_size}块"
        )

        return {
            "llm": llm_ok,
            "kb_size": kb_size,
            "kb_status": "就绪" if kb_size > 0 else "空（需先更新知识库）",
            "issues": issues,
        }

    def _build_crew(self):
        """构建 CrewAI Agent 和 Task"""

        # ========================
        # Agent 定义
        # ========================

        self.retriever_agent = Agent(
            role="知识检索专家",
            goal="从论文知识库中检索与用户问题最相关的文献片段，"
                 "提取 2-3 个关键词进行多查询融合检索，"
                 "返回格式化的检索结果，包含论文来源和相关性分数。",
            backstory=(
                "你是一位专业的学术文献检索专家，擅长从大规模论文"
                "数据库中快速定位最相关的研究成果。你会从用户问题中"
                "提炼核心概念，使用多角度检索确保召回率。"
            ),
            tools=[self.retrieve_tool],
            llm=self.llm,
            verbose=True,
        )

        self.analyst_agent = Agent(
            role="学术研究分析师",
            goal="基于检索到的文献进行深度分析，提取关键发现，"
                 "横向对比不同论文的方法和结论，识别研究趋势，"
                 "输出结构化的分析报告。",
            backstory=(
                "你是一位严谨的学术研究分析师，拥有深厚的 AI/ML "
                "领域知识。你善于从多篇论文中提炼共同主题、对比方法"
                "优劣、发现研究空白。你从不编造不存在的信息，"
                "所有结论都基于实际文献。"
            ),
            llm=self.llm,
            verbose=True,
        )

        self.reviewer_agent = Agent(
            role="研究质量审查员",
            goal="审查分析报告的质量，从事实准确性、逻辑完整性、"
                 "覆盖全面性、实用性四个维度评分，指出问题并给出"
                 "具体改进建议。",
            backstory=(
                "你是一位严格的研究质量审查员，对学术报告有敏锐的"
                "洞察力。你能发现逻辑漏洞、识别过度推断、指出遗漏的"
                "重要方面。你的审查帮助确保每份分析报告都达到出版级"
                "质量标准。"
            ),
            llm=self.llm,
            verbose=True,
        )

        self.synthesizer_agent = Agent(
            role="研究综合编辑",
            goal="综合检索结果、分析报告和审查意见，"
                 "输出一份结构清晰、有引用依据、可操作的最终回答。"
                 "如果分析和审查之间存在矛盾，需要指出并给出判断。",
            backstory=(
                "你是一位资深研究编辑，擅长将多份专业报告融合成"
                "一份条理分明、易于理解的综合性文档。你确保最终输出"
                "既严谨又实用，让读者能快速抓住要点并知道如何应用。"
            ),
            llm=self.llm,
            verbose=True,
        )

        # ========================
        # Task 定义
        # ========================

        self.retrieve_task = Task(
            description=(
                "用户提问：{question}\n\n"
                "请从知识库中检索与该问题最相关的论文文献。"
                "从问题中提取 2-3 个检索关键词，使用 "
                "retrieve_papers 工具进行多查询融合检索。\n\n"
                "返回格式：\n"
                "1. 使用的检索关键词\n"
                "2. 检索结果数量\n"
                "3. 每条结果的论文来源、内容和相关性\n"
                "如果未找到相关文献，如实告知。"
            ),
            expected_output=(
                "格式化的检索结果，包含关键词、命中数量和各条文献"
                "的详细信息（来源、内容摘要、相关性分数）。"
            ),
            agent=self.retriever_agent,
        )

        self.analyze_task = Task(
            description=(
                "用户原始提问：{question}\n\n"
                "请基于检索到的文献，进行深度分析，输出结构化的"
                "分析报告。报告必须包含以下四部分：\n"
                "1. **核心发现** - 最重要的结论和共识\n"
                "2. **方法对比** - 不同论文采用的方法及优劣\n"
                "3. **趋势洞察** - 研究发展方向和潜在突破点\n"
                "4. **关键引用** - 标注信息出自哪篇论文\n\n"
                "注意：只基于已有文献分析，不编造不存在的信息。"
            ),
            expected_output=(
                "包含核心发现、方法对比、趋势洞察、关键引用四部分"
                "的结构化分析报告。"
            ),
            agent=self.analyst_agent,
        )

        self.review_task = Task(
            description=(
                "请对以上分析报告进行质量审查。从以下四个维度评分（1-10）：\n"
                "- **事实准确度**：引用和数据是否可靠\n"
                "- **逻辑完整性**：推理链条是否完整\n"
                "- **覆盖全面性**：是否遗漏重要方面\n"
                "- **实用性**：结论是否有实际价值\n\n"
                "输出审查报告：列出具体问题、给出各维度评分、"
                "提供改进建议。"
            ),
            expected_output=(
                "包含具体问题列表、四维度评分和改 进建议的审查报告。"
            ),
            agent=self.reviewer_agent,
        )

        self.synthesize_with_review_task = Task(
            description=(
                "用户原始提问：{question}\n\n"
                "请综合以下三份材料，输出最终优化答案：\n"
                "1. 检索到的文献信息\n"
                "2. 分析报告\n"
                "3. 审查意见\n\n"
                "要求：\n"
                "- 结构清晰，分点阐述\n"
                "- 吸收审查意见中的改进建议\n"
                "- 如分析与审查有矛盾，明确指出并判断\n"
                "- 在回答末尾标注参考的论文来源"
            ),
            expected_output=(
                "综合性的最终回答，吸收审查意见，标注引用来源。"
            ),
            agent=self.synthesizer_agent,
        )

        self.synthesize_no_review_task = Task(
            description=(
                "用户原始提问：{question}\n\n"
                "请基于检索结果和分析报告，输出最终答案。\n"
                "要求：结构清晰、有引用依据。"
            ),
            expected_output=(
                "基于检索和分析的结构化回答，标注引用来源。"
            ),
            agent=self.synthesizer_agent,
        )

    def _auto_crawl_and_ingest(self, domains: dict, download: bool = True) -> dict:
        """自动爬取 + 自动入链"""
        results = self.crawler.crawl_all_domains(domains, download)
        if results:
            total = sum(len(v) for v in results.values())
            if total > 0:
                self.rag_pipeline.ingest_papers()
        return results

    # ========================
    # 查询缓存
    # ========================

    @staticmethod
    def _cache_key(question: str, enable_review: bool) -> str:
        """生成查询缓存键"""
        raw = f"{question}|{enable_review}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_from_cache(self, question: str, enable_review: bool) -> Optional[str]:
        """从 LRU 缓存中获取结果"""
        key = self._cache_key(question, enable_review)
        if key in self._query_cache:
            # 移动到末尾（标记为最近使用）
            self._query_cache.move_to_end(key)
            logger.info(f"[Cache] 命中缓存: {question[:50]}...")
            return self._query_cache[key]
        return None

    def _set_to_cache(self, question: str, enable_review: bool, result: str):
        """将结果写入 LRU 缓存"""
        key = self._cache_key(question, enable_review)
        self._query_cache[key] = result
        self._query_cache.move_to_end(key)
        # 淘汰最旧的条目
        while len(self._query_cache) > _MAX_CACHE_SIZE:
            self._query_cache.popitem(last=False)

    # ========================
    # 多轮对话管理（含持久化、查询改写、上下文预算）
    # ========================

    # 上下文预算（字符数）：总 prompt 控制在 LLM 窗口安全范围内
    _HISTORY_BUDGET_CHARS = 2000   # 对话历史最大字符数
    _RETRIEVAL_BUDGET_CHARS = 4000  # 检索结果最大字符数（已有的限制）
    _MAX_HISTORY_ROUNDS = 10       # 最多保留轮数

    def _get_conversation(self, conversation_id: str) -> list[dict]:
        """获取对话历史（内存 → 文件回退加载）"""
        if conversation_id in self._conversations:
            return self._conversations[conversation_id]

        # 尝试从文件加载
        filepath = CONVERSATIONS_DIR / f"{conversation_id}.json"
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._conversations[conversation_id] = data
                logger.info(f"[Conv] 从文件加载对话: {conversation_id[-8:]}, "
                            f"{len(data)} 条消息")
                return data
            except Exception as e:
                logger.warning(f"[Conv] 加载失败 {filepath}: {e}")

        # 新建空对话
        self._conversations[conversation_id] = []
        return self._conversations[conversation_id]

    def _save_conversation(self, conversation_id: str):
        """持久化对话到 JSON 文件"""
        conv = self._conversations.get(conversation_id, [])
        if not conv:
            return
        filepath = CONVERSATIONS_DIR / f"{conversation_id}.json"
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(conv, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[Conv] 保存失败: {e}")

    def _add_to_conversation(
        self, conversation_id: str, role: str, content: str
    ):
        """添加消息到对话历史（内存 + 文件持久化）"""
        conv = self._get_conversation(conversation_id)
        conv.append({"role": role, "content": content})
        # 保留最近 N 轮
        max_messages = self._MAX_HISTORY_ROUNDS * 2
        if len(conv) > max_messages:
            self._conversations[conversation_id] = conv[-max_messages:]
        # 异步持久化（fire-and-forget，不阻塞查询）
        self._save_conversation(conversation_id)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略估算 token 数（中英混合：约 1 字符 ≈ 0.6 token）"""
        if not text:
            return 0
        # 中文约占 0.7 token/char，英文约占 0.25 token/char
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 0.7 + other_chars * 0.25)

    def _format_history_with_budget(
        self, history: list[dict], max_chars: int
    ) -> str:
        """
        按预算格式化对话历史：从最新到最旧纳入，按句子边界截断。
        超出预算的旧轮次自动丢弃。
        """
        if not history:
            return ""

        lines = ["【对话历史】"]
        used = len(lines[0])

        # 从最新到最旧遍历（反转）
        for msg in reversed(history):
            role_label = "用户" if msg["role"] == "user" else "助手"
            content = msg["content"]

            # 按句子边界截断（中英文句号、问号、感叹号、换行）
            if len(content) > 500:
                truncated = content[:500]
                # 回退到最近的句子边界
                match = re.search(
                    r'[。！？.!?\n](?=[^。！？.!?\n]*$)',
                    truncated
                )
                if match and match.start() > 200:
                    content = truncated[:match.end()]
                else:
                    content = truncated + "..."

            line = f"{role_label}: {content}"
            if used + len(line) > max_chars:
                # 空间不够 → 标记省略并停止
                lines.append("...（更早的对话已省略）")
                break

            lines.append(line)
            used += len(line)

        return "\n".join(lines)

    async def _rewrite_for_retrieval(
        self, question: str, history: list[dict]
    ) -> str:
        """
        利用对话历史将省略式追问改写为自包含的检索查询。

        例如：用户先问"LoRA的优缺点？"，再问"那显存占用呢？"
        → 改写为 "LoRA微调方法的显存占用"

        仅使用轻量 LLM 调用（<1s），失败时回退到原问题。
        """
        if not history:
            return question

        # 只用最近 4 条消息（2 轮）做上下文
        recent = history[-4:]
        history_str = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:200]}"
            for m in recent
        )

        rewrite_prompt = (
            "你是一个查询改写助手。根据对话历史，将用户的追问改写为一个完整的、"
            "可以独立用于论文检索的查询语句。只输出改写后的查询，不要任何解释。\n\n"
            f"对话历史：\n{history_str}\n\n"
            f"用户追问：{question}\n\n"
            "改写后的检索查询："
        )

        try:
            import openai
            client = openai.OpenAI(
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
            )
            t0 = time.time()
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": rewrite_prompt}],
                max_tokens=80,
                temperature=0.0,
                timeout=10,
            )
            rewritten = resp.choices[0].message.content.strip()
            elapsed = time.time() - t0
            logger.info(
                f"[Rewrite] {question[:30]}... → {rewritten[:50]}... "
                f"({elapsed:.1f}s)"
            )
            return rewritten if rewritten else question

        except Exception as e:
            logger.warning(f"[Rewrite] 查询改写失败，回退原问题: {e}")
            return question

    # ========================
    # 直接检索（不经过 LLM）
    # ========================

    def _retrieve_directly(self, question: str) -> str:
        """
        直接调用检索器，不经过 LLM Agent。
        省掉 1 次 LLM API 调用（原 Retriever Agent 的开销）。
        """
        t0 = time.time()
        # 使用问题本身作为检索查询（混合检索 + 重排序）
        results = self.retriever.retrieve(
            question,
            top_k=8,
            use_hybrid=True,
            use_rerank=True,
        )
        elapsed = time.time() - t0
        logger.info(
            f"[DirectRetrieve] 命中 {len(results)} 条, 耗时 {elapsed:.2f}s"
        )
        return self.retriever.format_context(results, max_chars=4000)

    # ========================
    # 异步辅助
    # ========================

    @staticmethod
    async def _run_agent_step(agent, task, inputs, step_name):
        """执行单个 Agent 步骤并返回结果文本"""
        from crewai import Task as CTask

        t0 = time.time()
        logger.info(f"[Stream] {step_name} 开始...")

        # 如果传入的是字符串描述，构造 Task
        if isinstance(task, str):
            task = CTask(
                description=task,
                expected_output="",
                agent=agent,
            )

        mini_crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )
        output = await mini_crew.kickoff_async(inputs=inputs)
        elapsed = time.time() - t0
        logger.info(f"[Stream] {step_name} 完成, 耗时 {elapsed:.1f}s")
        return output.raw if hasattr(output, "raw") else str(output)

    # ========================
    # 用户接口
    # ========================

    def query(
        self,
        question: str,
        enable_review: bool = True,
        conversation_id: Optional[str] = None,
    ) -> str:
        """
        用户提问接口（非流式），使用 CrewAI 编排多 Agent 协作。

        Args:
            question: 用户问题
            enable_review: 是否启用审查 Agent
            conversation_id: 多轮对话 ID（可选）

        Returns:
            综合回答
        """
        # 检查缓存
        cached = self._get_from_cache(question, enable_review)
        if cached:
            return cached

        logger.info(f"[CrewAI Query] {question[:80]}...")

        # 获取并加载对话历史
        history = []
        if conversation_id:
            history = self._get_conversation(conversation_id)

        # 查询改写 + 检索
        retrieval_question = question
        if history:
            import asyncio as _asyncio
            loop = _asyncio.new_event_loop()
            try:
                retrieval_question = loop.run_until_complete(
                    self._rewrite_for_retrieval(question, history)
                )
            finally:
                loop.close()

        # 直接检索
        retrieve_text = self._retrieve_directly(retrieval_question)

        # 按预算构建对话历史上下文
        history_text = ""
        if history:
            history_text = self._format_history_with_budget(
                history, self._HISTORY_BUDGET_CHARS
            )

        # 构建分析任务
        from crewai import Task as CTask
        analyze_desc = (
            self.analyze_task.description +
            (f"\n{history_text}" if history_text else "") +
            f"\n\n【检索到的文献】\n{retrieve_text}"
        )
        analyze_t = CTask(
            description=analyze_desc,
            expected_output=self.analyze_task.expected_output,
            agent=self.analyst_agent,
        )

        if enable_review:
            crew = Crew(
                agents=[self.analyst_agent, self.reviewer_agent,
                        self.synthesizer_agent],
                tasks=[analyze_t, self.review_task,
                       self.synthesize_with_review_task],
                process=Process.sequential,
                verbose=True,
            )
        else:
            crew = Crew(
                agents=[self.analyst_agent, self.synthesizer_agent],
                tasks=[analyze_t, self.synthesize_no_review_task],
                process=Process.sequential,
                verbose=True,
            )

        import asyncio
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                crew.kickoff_async(inputs={"question": question})
            )
        finally:
            loop.close()

        final_text = result.raw if hasattr(result, "raw") else str(result)

        # 保存到缓存
        self._set_to_cache(question, enable_review, final_text)

        # 保存到对话历史
        if conversation_id:
            self._add_to_conversation(conversation_id, "user", question)
            self._add_to_conversation(conversation_id, "assistant", final_text)

        return final_text

    async def query_stream(
        self,
        question: str,
        enable_review: bool = True,
        conversation_id: Optional[str] = None,
    ):
        """
        流式查询：逐 Agent 返回进度和结果。

        优化点：
        - 检索直接执行（不经过 LLM Agent），省 1 次 LLM 调用
        - 支持多轮对话上下文
        - 支持结果缓存

        Yields:
            {"type": "progress", "agent": "...", "status": "..."}
            {"type": "agent_done", "agent": "...", "preview": "..."}
            {"type": "result", "content": "..."}
        """
        logger.info(f"[Stream Query] {question[:80]}...")
        metrics = get_metrics()
        metrics.count("query_total")
        t_total = time.time()

        # 检查缓存
        cached = self._get_from_cache(question, enable_review)
        if cached:
            logger.info("[Stream] 缓存命中，直接返回")
            metrics.count("cache_hit")
            metrics.record_timing("total", time.time() - t_total)
            yield {"type": "progress", "agent": "Cache",
                   "status": "命中缓存，直接返回结果"}
            yield {"type": "result", "content": cached}
            return

        # 检查知识库是否为空
        if self.vector_store.get_count() == 0:
            yield {"type": "result",
                   "content": "知识库为空，尚未导入任何论文。请先在左侧边栏点击「更新知识库」或「论文入链」。"}
            return

        # 获取并加载对话历史
        history = []
        if conversation_id:
            history = self._get_conversation(conversation_id)

        # ============================================================
        # Step 0: 查询改写（利用对话历史补全省略式追问）
        # ============================================================
        retrieval_question = question
        if history:
            t_rewrite = time.time()
            retrieval_question = await self._rewrite_for_retrieval(
                question, history
            )
            metrics.record_timing("rewrite", time.time() - t_rewrite)

        # ============================================================
        # Step 1: 直接检索（使用改写后的查询，不经过 LLM）
        # ============================================================
        yield {"type": "progress", "agent": "Retriever",
               "status": "正在混合检索知识库（向量 + BM25 + 重排序）..."}

        import asyncio
        t_retrieve = time.time()
        retrieve_text = await asyncio.to_thread(
            self._retrieve_directly, retrieval_question
        )
        metrics.record_timing("retrieve", time.time() - t_retrieve)
        yield {"type": "agent_done", "agent": "Retriever",
               "preview": retrieve_text[:200]}

        # ============================================================
        # Step 2: Analyze
        # ============================================================
        yield {"type": "progress", "agent": "Analyst",
               "status": "正在深度分析检索到的文献..."}

        # 按预算构建对话历史上下文
        history_text = ""
        if history:
            history_text = self._format_history_with_budget(
                history, self._HISTORY_BUDGET_CHARS
            )

        analyze_desc = (
            self.analyze_task.description +
            (f"\n{history_text}" if history_text else "") +
            f"\n\n【检索到的文献】\n{retrieve_text}"
        )
        t_analyze = time.time()
        analyze_text = await self._run_agent_step(
            self.analyst_agent, analyze_desc,
            {"question": question}, "Analyst"
        )
        metrics.record_timing("analyze", time.time() - t_analyze)
        metrics.count("llm_call")
        yield {"type": "agent_done", "agent": "Analyst",
               "preview": analyze_text[:200]}

        # ============================================================
        # Step 3: Review（可选）
        # ============================================================
        if enable_review:
            yield {"type": "progress", "agent": "Reviewer",
                   "status": "正在审查分析报告的质量..."}

            review_desc = (
                self.review_task.description +
                f"\n\n【待审查的分析报告】\n{analyze_text}"
            )
            t_review = time.time()
            review_text = await self._run_agent_step(
                self.reviewer_agent, review_desc,
                {}, "Reviewer"
            )
            metrics.record_timing("review", time.time() - t_review)
            metrics.count("llm_call")
            yield {"type": "agent_done", "agent": "Reviewer",
                   "preview": review_text[:200]}

            # ---- Step 4: Synthesize ----
            yield {"type": "progress", "agent": "Synthesizer",
                   "status": "正在综合所有结果生成最终回答..."}

            synth_desc = (
                self.synthesize_with_review_task.description +
                (f"\n{history_text}" if history_text else "") +
                f"\n\n【检索结果】\n{retrieve_text}\n\n"
                f"【分析报告】\n{analyze_text}\n\n"
                f"【审查意见】\n{review_text}"
            )
        else:
            yield {"type": "progress", "agent": "Synthesizer",
                   "status": "正在综合结果生成回答..."}
            synth_desc = (
                self.synthesize_no_review_task.description +
                (f"\n{history_text}" if history_text else "") +
                f"\n\n【检索结果】\n{retrieve_text}\n\n"
                f"【分析报告】\n{analyze_text}"
            )

        t_synth = time.time()
        final_text = await self._run_agent_step(
            self.synthesizer_agent, synth_desc,
            {"question": question}, "Synthesizer"
        )
        metrics.record_timing("synthesize", time.time() - t_synth)
        metrics.count("llm_call")
        metrics.record_timing("total", time.time() - t_total)

        # 记录查询明细
        metrics.log_query({
            "question": question[:100],
            "conversation_id": conversation_id[-8:] if conversation_id else None,
            "enable_review": enable_review,
            "rewritten": retrieval_question != question,
            "total_s": round(time.time() - t_total, 2),
        })

        # 保存到缓存
        self._set_to_cache(question, enable_review, final_text)

        # 保存到对话历史
        if conversation_id:
            self._add_to_conversation(conversation_id, "user", question)
            self._add_to_conversation(conversation_id, "assistant", final_text)

        yield {"type": "result", "content": final_text}

    def crawl_domain(self, domain_name: str, max_results: int = 20) -> str:
        """手动触发指定领域爬取"""
        from src.config import RESEARCH_DOMAINS

        if domain_name not in RESEARCH_DOMAINS:
            domain_name = next(
                (k for k in RESEARCH_DOMAINS if domain_name in k),
                domain_name,
            )

        category = RESEARCH_DOMAINS.get(domain_name, "cs.AI")
        papers = self.crawler.crawl_domain(
            domain_name, category, download=True, max_results=max_results
        )
        if papers:
            self.rag_pipeline.ingest_papers()

        return (
            f"领域 [{domain_name}] 爬取完成！\n"
            f"  新增论文: {len(papers)} 篇\n"
            f"  向量库总量: {self.vector_store.get_count()} 条"
        )

    def update_knowledge_base(self, domain: Optional[str] = None, max_results: int = 20) -> str:
        """手动触发知识库更新（爬取 + 入链）"""
        from src.config import RESEARCH_DOMAINS

        if domain:
            domains = {domain: RESEARCH_DOMAINS.get(domain, "cs.AI")}
        else:
            domains = RESEARCH_DOMAINS

        results = self.crawler.crawl_all_domains(domains, download=True)
        total = sum(len(v) for v in results.values())

        if total > 0:
            ingest_result = self.rag_pipeline.ingest_papers()
            return (
                f"知识库更新完成!\n"
                f"  新增论文: {total} 篇\n"
                f"  文本块: {ingest_result.get('chunks', 0)} 个\n"
                f"  向量库总量: {self.vector_store.get_count()} 条"
            )
        return "未发现新论文，知识库已是最新。"

    def get_stats(self) -> dict:
        """获取系统统计信息"""
        return {
            "vector_store": {
                "total_chunks": self.vector_store.get_count(),
                "domains": self.vector_store.get_domain_stats(),
            },
            "papers": {
                domain: len(list(pdfs))
                for domain, pdfs in self.crawler.get_stored_papers().items()
            },
            "scheduler": {
                "running": self.scheduler.scheduler.running if self.scheduler else False,
                "last_crawl": (
                    str(self.scheduler.last_crawl_time)
                    if self.scheduler and self.scheduler.last_crawl_time
                    else None
                ),
                "last_result": (
                    self.scheduler.last_result_summary if self.scheduler else None
                ),
            },
        }

    def start_scheduler(self):
        """启动定时爬虫"""
        self.scheduler.start()

    def stop_scheduler(self):
        """停止定时爬虫"""
        self.scheduler.stop()


# 全局单例
_system: Optional[CrewMultiAgentSystem] = None


def get_system() -> CrewMultiAgentSystem:
    """获取系统单例"""
    global _system
    if _system is None:
        _system = CrewMultiAgentSystem()
    return _system


# 别名，保持向后兼容
MultiAgentSystem = CrewMultiAgentSystem
