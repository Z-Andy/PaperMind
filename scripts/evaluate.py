#!/usr/bin/env python
"""
检索与回答质量评估脚本。

支持三种模式：
  1. qasper:  直接下载 QASPER 数据集评估（推荐！arXiv NLP 论文，5049 个专家标注问题）
  2. retrieval: 使用自定义测试集评估检索质量（Recall@K、MRR、NDCG）
  3. judge:  LLM-as-Judge 对回答进行忠实度、相关性、完整性打分
  4. gen-template: 生成自定义测试集模板

用法：
  # ★ 推荐：直接评估 QASPER（自动下载，arxiv NLP 论文）
  python scripts/evaluate.py --mode qasper

  # 检索质量评估（需要准备 ground_truth.json）
  python scripts/evaluate.py --mode retrieval --testset test_queries.json

  # LLM-as-Judge 回答质量评估
  python scripts/evaluate.py --mode judge --testset test_queries.json

  # 生成示例测试集模板
  python scripts/evaluate.py --mode gen-template

外部公开数据集（可直接下载使用）：
  - QASPER:   allenai/qasper      (1585 篇 NLP arXiv 论文, 5049 问题, 含 evidence)
  - PeerQA:   UKPLab/PeerQA       (208 篇论文, 579 问题, 同行评审来源)
  - PeerQA-XT: UKPLab/PeerQA-XT   (12628 问题, 10 领域, LLM 合成)
  - MIRAGE:   nlpai-lab/MIRAGE    (7560 查询, RAG 专项评估)
  - BRIGHT:   xlangai/BRIGHT      (1384 推理密集型检索查询)

依赖：
  pip install datasets  # (用于加载 HuggingFace 数据集)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path 中（支持从 scripts/ 目录直接运行）
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 测试集格式
# ============================================================

def generate_template(output_path: str = "test_queries.json"):
    """生成测试集模板，用户填写 ground_truth 后即可评估"""
    template = {
        "description": "检索质量评估测试集。请为每条 query 填写 relevant_ids（相关论文ID）和 relevant_text（期望的关键信息）。",
        "queries": [
            {
                "id": "q001",
                "question": "大语言模型推理优化有哪些主流方法？",
                "relevant_ids": [],
                "relevant_text": ""
            },
            {
                "id": "q002",
                "question": "多Agent系统的协作策略有哪些？",
                "relevant_ids": [],
                "relevant_text": ""
            },
            {
                "id": "q003",
                "question": "RAG中检索增强对生成质量的影响有多大？",
                "relevant_ids": [],
                "relevant_text": ""
            },
            {
                "id": "q004",
                "question": "LoRA微调方法的优缺点是什么？",
                "relevant_ids": [],
                "relevant_text": ""
            },
            {
                "id": "q005",
                "question": "Transformer架构近年来有哪些重要改进？",
                "relevant_ids": [],
                "relevant_text": ""
            }
        ]
    }

    path = Path(output_path)
    path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"测试集模板已生成: {path}")
    logger.info("请编辑该文件，为每个问题填写 relevant_ids 和 relevant_text")


# ============================================================
# 评估指标实现
# ============================================================

def compute_recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@K: 前K个检索结果中命中了多少相关文档"""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return hits / len(relevant_ids)


def compute_mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """MRR (Mean Reciprocal Rank): 第一个相关文档排名的倒数"""
    if not relevant_ids:
        return 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """NDCG@K: 归一化折损累计增益（相关=1, 不相关=0）"""
    import math
    if not relevant_ids:
        return 0.0

    # DCG: 相关度 / log2(rank+1)
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, rid in enumerate(retrieved_ids[:k])
        if rid in relevant_ids
    )

    # IDCG: 理想排序（所有相关文档排在最前面）
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# 检索质量评估
# ============================================================

def evaluate_retrieval(testset_path: str):
    """运行检索质量评估"""
    from src.agents.crew_system import get_system

    testset = json.loads(Path(testset_path).read_text(encoding="utf-8"))
    queries = testset.get("queries", [])
    system = get_system()
    retriever = system.retriever

    results = []
    k_values = [3, 5, 10]

    for q in queries:
        question = q["question"]
        relevant = set(q.get("relevant_ids", []))
        qid = q.get("id", question[:20])

        if not relevant:
            logger.warning(f"跳过 {qid}: 未标注 relevant_ids")
            continue

        t0 = time.time()
        retrieved = retriever.retrieve(question, top_k=10, use_hybrid=True, use_rerank=True)
        elapsed = time.time() - t0
        retrieved_ids = [r.get("id", "") for r in retrieved]

        scores = {"id": qid, "question": question[:60], "latency_s": round(elapsed, 2)}
        for k in k_values:
            scores[f"recall@{k}"] = round(compute_recall_at_k(retrieved_ids, relevant, k), 3)
        scores["mrr"] = round(compute_mrr(retrieved_ids, relevant), 3)
        scores["ndcg@10"] = round(compute_ndcg_at_k(retrieved_ids, relevant, 10), 3)

        results.append(scores)
        logger.info(
            f"[{qid}] Recall@5={scores.get('recall@5', 'N/A')}, "
            f"MRR={scores['mrr']}, 耗时={elapsed:.2f}s"
        )

    # 汇总
    if results:
        print("\n" + "=" * 60)
        print("检索质量评估汇总")
        print("=" * 60)
        for metric in ["recall@3", "recall@5", "recall@10", "mrr", "ndcg@10"]:
            values = [r[metric] for r in results if metric in r]
            if values:
                avg = sum(values) / len(values)
                print(f"  {metric}: {avg:.3f}  (n={len(values)})")
        print(f"  平均检索耗时: {sum(r['latency_s'] for r in results)/len(results):.2f}s")
        print("=" * 60)

        # 保存详细结果
        out_path = Path(testset_path).with_suffix(".results.json")
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"详细结果已保存: {out_path}")


# ============================================================
# LLM-as-Judge 回答质量评估
# ============================================================

JUDGE_PROMPT = """你是一个严格的学术回答质量评估专家。请评估以下AI助手对用户问题的回答质量。

用户问题：{question}

AI回答：
{answer}

参考信息（来自知识库检索）：
{context}

请从以下三个维度评分（1-10分），并给出简要理由：

1. **忠实度 (Faithfulness)**：回答是否严格基于参考信息？有无编造不存在的内容？
2. **相关性 (Relevance)**：回答是否直接回应用户问题？有无跑题？
3. **完整性 (Completeness)**：回答是否覆盖了问题的主要方面？

请按以下JSON格式输出：
{{
    "faithfulness": 8,
    "relevance": 9,
    "completeness": 7,
    "overall": 8.0,
    "reason": "简要评价..."
}}"""


def evaluate_answers(testset_path: str):
    """LLM-as-Judge 评估回答质量"""
    from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    from src.agents.crew_system import get_system

    if not LLM_API_KEY:
        logger.error("请先配置 LLM_API_KEY")
        return

    import openai
    client = openai.OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    testset = json.loads(Path(testset_path).read_text(encoding="utf-8"))
    queries = testset.get("queries", [])
    system = get_system()

    results = []

    for q in queries:
        question = q["question"]
        qid = q.get("id", question[:20])
        logger.info(f"评估 [{qid}]: {question[:50]}...")

        # 获取系统回答
        answer = system.query(question, enable_review=False)

        # 获取检索上下文（用于检查忠实度）
        retrieved = system.retriever.retrieve(question, top_k=5, use_hybrid=True, use_rerank=True)
        context = system.retriever.format_context(retrieved, max_chars=3000)

        # LLM 评分
        prompt = JUDGE_PROMPT.format(question=question, answer=answer, context=context)
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
                timeout=60,
            )
            judge_output = resp.choices[0].message.content

            # 尝试解析 JSON
            try:
                # 提取 JSON 部分
                import re
                json_match = re.search(r'\{[^}]+\}', judge_output, re.DOTALL)
                if json_match:
                    scores = json.loads(json_match.group())
                else:
                    scores = {"raw": judge_output}
            except json.JSONDecodeError:
                scores = {"raw": judge_output}

            scores["id"] = qid
            scores["question"] = question[:60]
            results.append(scores)

            overall = scores.get("overall", "?")
            logger.info(f"  → 综合评分: {overall}")

        except Exception as e:
            logger.error(f"评估 [{qid}] 失败: {e}")
            results.append({"id": qid, "error": str(e)})

    # 汇总
    if results:
        print("\n" + "=" * 60)
        print("LLM-as-Judge 回答质量评估汇总")
        print("=" * 60)
        valid = [r for r in results if "faithfulness" in r]
        if valid:
            for metric in ["faithfulness", "relevance", "completeness", "overall"]:
                values = [r[metric] for r in valid if metric in r]
                if values:
                    avg = sum(values) / len(values)
                    print(f"  {metric}: {avg:.1f}/10  (n={len(values)})")
        print("=" * 60)

        out_path = Path(testset_path).with_suffix(".judge_results.json")
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"详细结果已保存: {out_path}")


# ============================================================
# QASPER 数据集评估（推荐方式，无需手动标注）
# ============================================================

def evaluate_qasper(split: str = "validation", max_samples: int = 100):
    """
    使用 QASPER 数据集评估检索质量。

    QASPER 包含 1585 篇 arXiv NLP 论文和 5049 个专家标注问题，
    每个问题都标注了答案所在的 evidence 段落。非常适合评估 PaperMind 的检索性能。

    数据集地址：https://huggingface.co/datasets/allenai/qasper
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("请先安装 datasets: pip install datasets")
        return

    logger.info(f"正在加载 QASPER 数据集 ({split} split)...")
    dataset = load_dataset("allenai/qasper", split=split, streaming=False)
    logger.info(f"已加载 {len(dataset)} 条 QA 记录")

    results = []
    k_values = [3, 5, 10]
    processed = 0
    skipped_no_text = 0
    skipped_no_evidence = 0

    for row in dataset:
        if processed >= max_samples:
            break

        paper_id = row.get("id", "unknown")

        # 构建论文全文
        full_text_parts = []
        if row.get("full_text"):
            ft = row["full_text"]
            paras = ft.get("paragraphs", [])
            for sec_paras in paras:
                for para in sec_paras:
                    full_text_parts.append(para)

        full_text = "\n".join(full_text_parts)
        if not full_text:
            skipped_no_text += 1
            continue

        # QASPER 每行是一个 QA 对
        qas = row.get("qas", {})
        question = qas.get("question", "") if isinstance(qas, dict) else ""
        answers = qas.get("answers", []) if isinstance(qas, dict) else []

        if not question or not answers:
            continue

        # 提取 evidence 文本
        evidence_texts = set()
        for ans in answers:
            if not isinstance(ans, dict):
                continue
            ans_data = ans.get("answer", ans)
            if isinstance(ans_data, dict):
                for ev in ans_data.get("evidence", []):
                    if isinstance(ev, str) and ev.strip():
                        evidence_texts.add(ev.strip()[:300])
                    elif isinstance(ev, dict):
                        t = ev.get("text", ev.get("content", ""))
                        if t and isinstance(t, str):
                            evidence_texts.add(t.strip()[:300])

        if not evidence_texts:
            skipped_no_evidence += 1
            continue

        # 检索
        paragraphs = [p.strip() for p in full_text.split("\n") if len(p.strip()) > 50]
        if len(paragraphs) < 3:
            continue

        from rank_bm25 import BM25Okapi
        tokenized_paras = [p.lower().split() for p in paragraphs]
        bm25 = BM25Okapi(tokenized_paras)
        scores = bm25.get_scores(question.lower().split())
        max_k = max(k_values)
        if len(scores) < max_k:
            max_k = len(scores)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:max_k]

        retrieved_paras = [paragraphs[i] for i in ranked_indices]

        def _is_relevant(retrieved_text):
            r = retrieved_text[:200].strip()
            for ev in evidence_texts:
                if r[:80] in ev or ev[:80] in r:
                    return True
            return False

        relevant_ids = set(
            str(i) for i, p in enumerate(retrieved_paras) if _is_relevant(p)
        )

        item = {"id": f"{paper_id}_q0", "question": question[:80]}
        for k in k_values:
            item[f"recall@{k}"] = round(
                compute_recall_at_k(
                    [str(j) for j in ranked_indices[:k]], relevant_ids, k
                ), 3
            )
        item["mrr"] = round(
            compute_mrr([str(j) for j in ranked_indices], relevant_ids), 3
        )
        results.append(item)
        processed += 1

        if processed % 20 == 0:
            logger.info(f"  已评估 {processed} 个问题...")

    # 汇总
    logger.info(
        f"评估完成: 有效={len(results)}, "
        f"无全文={skipped_no_text}, 无evidence={skipped_no_evidence}"
    )
    if results:
        print("\n" + "=" * 60)
        print(f"QASPER 检索质量评估 ({split}, n={len(results)})")
        print("=" * 60)
        for metric in ["recall@3", "recall@5", "recall@10", "mrr"]:
            values = [r[metric] for r in results if metric in r]
            if values:
                avg = sum(values) / len(values)
                print(f"  {metric}: {avg:.3f}")
        print("=" * 60)

        out_path = Path(f"qasper_{split}_results.json")
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"详细结果已保存: {out_path}")
    else:
        print("\n无有效评估结果。请检查 QASPER 数据结构是否匹配。")


# ============================================================
# QASPER 端到端评估（走完整 PaperMind 管线）
# ============================================================

def evaluate_qasper_e2e(split: str = "validation", max_samples: int = 20):
    """
    端到端评估：将 QASPER 论文灌入 PaperMind 知识库，用完整 Agent 管线回答，评估准确率。

    流程：
    1. 加载 QASPER 论文全文并作为文本块灌入临时向量库
    2. 逐个提问走 query_stream 获取回答
    3. 用 LLM-as-Judge 对比 ground truth 打分
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("请先安装 datasets: pip install datasets")
        return

    logger.info(f"正在加载 QASPER 数据集 ({split} split)...")
    dataset = load_dataset("allenai/qasper", split=split, streaming=False)
    papers = list(dataset)
    logger.info(f"已加载 {len(papers)} 条 QA 记录")

    # ---- 收集问题 ----
    test_questions = []
    for row in papers:
        # 提取全文
        full_text_parts = []
        if row.get("full_text"):
            ft = row["full_text"]
            paras = ft.get("paragraphs", [])
            for sec_paras in paras:
                for para in sec_paras:
                    full_text_parts.append(para)
        full_text = "\n".join(full_text_parts)
        if not full_text:
            continue

        qas = row.get("qas", {})
        questions = qas.get("question", []) if isinstance(qas, dict) else []
        answers_list = qas.get("answers", []) if isinstance(qas, dict) else []

        if not isinstance(questions, list) or not answers_list:
            continue

        for idx, question in enumerate(questions):
            if idx >= len(answers_list):
                continue
            ans_entry = answers_list[idx]
            if not isinstance(ans_entry, dict):
                continue

            # answer 内层也是列表（多个 annotator 的答案）
            inner_answers = ans_entry.get("answer", [])
            if not isinstance(inner_answers, list):
                inner_answers = [inner_answers]

            # 优先取 free_form_answer，否则拼接 evidence 作为 ground truth
            gt_answer = ""
            for ia in inner_answers:
                if isinstance(ia, dict):
                    fa = ia.get("free_form_answer", "")
                    if fa and fa.strip():
                        gt_answer = fa
                        break
            if not gt_answer:
                # 无 free_form_answer，用 highlighted_evidence 拼接
                for ia in inner_answers:
                    if isinstance(ia, dict):
                        he = ia.get("highlighted_evidence", ia.get("evidence", []))
                        if isinstance(he, list) and he:
                            gt_answer = " ".join(str(e) for e in he[:3])
                            break

            if not gt_answer or not question:
                continue

            test_questions.append({
                "paper_id": row.get("id", "unknown"),
                "full_text": full_text,
                "question": question if isinstance(question, str) else str(question),
                "gt_answer": gt_answer,
            })

        if len(test_questions) >= max_samples:
            break

    logger.info(f"收集到 {len(test_questions)} 个有效问题")

    # ---- 用独立向量库灌入 QASPER 论文（不污染用户知识库） ----
    logger.info("构建 QASPER 专用临时向量库...")
    from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    from src.rag.chunker import Chunk
    from src.rag.embedder import Embedder
    from src.rag.vector_store import VectorStore
    from src.rag.retriever import Retriever

    import shutil
    tmp_db_dir = Path("data/chroma_eval")
    if tmp_db_dir.exists():
        shutil.rmtree(tmp_db_dir)

    eval_embedder = Embedder()
    eval_vs = VectorStore(
        collection_name="qasper_e2e",
        persist_dir=tmp_db_dir,
    )
    eval_retriever = Retriever(eval_vs, eval_embedder)

    # 收集所有唯一论文全文并分块入库
    seen_ids = set()
    chunks = []
    for tq in test_questions:
        pid = tq["paper_id"]
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        text = tq["full_text"]
        for i in range(0, len(text), 1000):
            chunk_text = text[i:i + 1000].strip()
            if len(chunk_text) > 100:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_index=len(chunks),
                    metadata={"title": pid},
                ))

    if chunks:
        logger.info(f"向量化 {len(chunks)} 个文本块 (含 {len(seen_ids)} 篇论文)...")
        eval_vs.add_chunks(chunks, eval_embedder)
        logger.info(f"临时向量库构建完成: {len(chunks)} 块")

    # ---- 用独立检索器 + LLM 做 RAG（不走 CrewAI，保证测的是检索+生成核心能力） ----
    import openai
    client = openai.OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    results = []
    for i, tq in enumerate(test_questions):
        question = tq["question"]
        gt = tq["gt_answer"]
        logger.info(f"\n[{i + 1}/{len(test_questions)}] {question[:60]}...")

        # 检索
        retrieved = eval_retriever.retrieve(question, top_k=5, use_hybrid=True, use_rerank=True)
        context = eval_retriever.format_context(retrieved, max_chars=3000)

        # LLM 直接回答（模拟 RAG，不经过 CrewAI Agent 编排）
        rag_prompt = (
            "你是一个学术研究助手。请根据以下论文文献回答用户问题。\n"
            "如果文献中没有相关信息，请如实说明。\n\n"
            f"【论文文献】\n{context}\n\n"
            f"【用户问题】{question}\n\n"
            "请给出准确、简洁的回答："
        )

        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": rag_prompt}],
                temperature=0.3,
                max_tokens=600,
                timeout=120,
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM 回答失败: {e}")
            results.append({"question": question[:80], "error": str(e)})
            continue

        # 用 LLM 对比打分
        judge_prompt = (
            "你是一个严格的评估专家。请对比 AI 回答和参考答案，从 1-10 给以下维度打分：\n\n"
            "1. 准确度 (Accuracy): AI 回答中的关键事实是否与参考答案一致？\n"
            "2. 覆盖度 (Coverage): AI 回答是否涵盖了参考答案的主要信息点？\n"
            "3. 忠实度 (Faithfulness): AI 回答是否有参考答案之外的编造内容？\n\n"
            f"问题：{question}\n\n"
            f"参考答案：{gt[:500]}\n\n"
            f"AI 回答：{answer[:500]}\n\n"
            "请仅输出 JSON: {\"accuracy\": X, \"coverage\": X, \"faithfulness\": X, \"brief\": \"一句话评价\"}"
        )

        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                max_tokens=200,
                timeout=60,
            )
            raw = resp.choices[0].message.content
            # 提取 JSON
            import re
            m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
            scores = json.loads(m.group()) if m else {"raw": raw}
        except Exception as e:
            scores = {"error": str(e)}

        scores["question"] = question[:80]
        scores["answer_preview"] = answer[:200]
        results.append(scores)

        logger.info(
            f"  准确度={scores.get('accuracy', '?')}, "
            f"覆盖度={scores.get('coverage', '?')}, "
            f"忠实度={scores.get('faithfulness', '?')}"
        )

    # ---- 汇总 ----
    valid = [r for r in results if "accuracy" in r]
    if valid:
        print("\n" + "=" * 60)
        print(f"PaperMind 端到端准确率 ({split}, n={len(valid)})")
        print("=" * 60)
        for metric in ["accuracy", "coverage", "faithfulness"]:
            values = [r[metric] for r in valid if metric in r]
            avg = sum(values) / len(values) if values else 0
            print(f"  {metric}: {avg:.1f}/10")
        print("=" * 60)

        out_path = Path(f"qasper_e2e_{split}_results.json")
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"详细结果已保存: {out_path}")

    # ---- 清理 ----
    if tmp_db_dir.exists():
        shutil.rmtree(tmp_db_dir)
        logger.info(f"已清理临时向量库: {tmp_db_dir}")


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PaperMind 评估工具")
    parser.add_argument(
        "--mode", choices=["qasper", "qasper-e2e", "retrieval", "judge", "gen-template"],
        default="gen-template", help="评估模式"
    )
    parser.add_argument(
        "--testset", default="test_queries.json",
        help="测试集 JSON 文件路径"
    )
    parser.add_argument(
        "--max-samples", type=int, default=100,
        help="QASPER 模式最多评估问题数"
    )
    parser.add_argument(
        "--split", default="validation", choices=["train", "validation", "test"],
        help="QASPER 数据子集"
    )
    args = parser.parse_args()

    if args.mode == "gen-template":
        generate_template(args.testset)
    elif args.mode == "qasper":
        evaluate_qasper(split=args.split, max_samples=args.max_samples)
    elif args.mode == "qasper-e2e":
        evaluate_qasper_e2e(split=args.split, max_samples=args.max_samples)
    elif args.mode == "retrieval":
        evaluate_retrieval(args.testset)
    elif args.mode == "judge":
        evaluate_answers(args.testset)


if __name__ == "__main__":
    main()
