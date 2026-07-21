"""
章节感知分块器：按论文结构（章节→子章节→段落）分块，
而非按固定字符数硬切。
"""
import re
import logging
from dataclasses import dataclass
from typing import Optional

from src.config import CHUNK_SIZE

logger = logging.getLogger(__name__)

# 每块最大字符数（超长段落按句子切分时的上限）
MAX_CHUNK_CHARS = CHUNK_SIZE * 2  # 2000，章节块可以比字符块大一些
PARAGRAPH_CHUNK_LIMIT = CHUNK_SIZE * 3  # 3000，超长段落的切分上限


@dataclass
class Chunk:
    """文档分块"""
    text: str
    chunk_index: int
    metadata: dict


class SectionChunker:
    """按论文章节结构分块，保持内容语义完整性"""

    # ---- 章节标题检测规则 ----
    SECTION_PATTERNS = [
        # 编号标题 (最优先): "1. Introduction", "3.2.1 Model Design"
        re.compile(r'^(\d+(?:\.\d+)*)\s{1,3}(.+)$'),
        # 全大写短行: "ABSTRACT", "INTRODUCTION"
        re.compile(r'^([A-Z][A-Z\s\-]{4,40})$'),
        # 关键词标题
        re.compile(
            r'^(Abstract|Introduction|Related\s+Work|Background|'
            r'Method|Methodology|Approach|Experiments?|Evaluation|'
            r'Results?|Discussion|Conclusion|Future\s+Work|'
            r'References?|Bibliography|Appendix|Appendices|'
            r'Acknowledgments?|Acknowledgements?|'
            r'Limitations?|Ethics\s+Statement|'
            r'Preliminaries?|Notation|'
            r'Problem\s+Formulation|Proposed\s+Method|'
            r'System\s+Overview|Architecture|'
            r'Training|Inference|Implementation\s+Details|'
            r'Ablation\s+Study|Ablation\s+Studies|'
            r'Quantitative\s+Analysis|Qualitative\s+Analysis|'
            r'Hyperparameter|Hyperparameters?|'
            r'Baselines?|Benchmarks?|'
            r'Dataset|Datasets?|Data\s+Collection|'
            r'Case\s+Study|Case\s+Studies|'
            r'Related\s+Literature|Literature\s+Review|'
            r'Problem\s+Statement|'
            r'Contributions?|Our\s+Contributions?)$',
            re.IGNORECASE,
        ),
    ]

    def __init__(self, max_chunk_chars: int = None):
        self.max_chunk_chars = max_chunk_chars or MAX_CHUNK_CHARS

    # ============================================================
    # 公共接口
    # ============================================================

    def chunk_text(self, text: str, metadata: dict = None) -> list[Chunk]:
        """将文本按章节结构拆分为块"""
        if not text or not text.strip():
            return []

        if metadata is None:
            metadata = {}

        # Step 1: 解析为章节树
        sections = self._parse_sections(text)

        if not sections:
            # 没有检测到章节结构，回退到按段落分块
            return self._fallback_chunk(text, metadata)

        # Step 2: 构建分块，每块带章节路径
        chunks = self._build_section_chunks(sections, metadata)

        logger.info(
            f"章节分块完成: {len(sections)}个章节 → {len(chunks)}个块"
        )
        return chunks

    def chunk_documents(self, documents: list) -> list[Chunk]:
        """批量分块多个文档"""
        all_chunks = []
        for doc in documents:
            meta = {
                "source": doc.file_path,
                "title": doc.title,
                **doc.metadata,
            }
            chunks = self.chunk_text(doc.content, metadata=meta)
            all_chunks.extend(chunks)

        logger.info(
            f"分块完成: {len(documents)}篇论文 → {len(all_chunks)}个文本块"
        )
        return all_chunks

    # ============================================================
    # 章节解析
    # ============================================================

    def _parse_sections(self, text: str) -> list[dict]:
        """
        解析文本为章节结构。
        返回: [{"level": 1, "number": "1", "title": "Introduction",
                "paragraphs": ["para1", "para2"]}, ...]
        """
        lines = text.split('\n')
        # 过滤掉页眉页脚噪声（以 "--- 第X页 ---" 开头的行）
        clean_lines = [
            l for l in lines
            if not l.startswith("--- 第") and l.strip()
        ]

        # 检测所有标题行
        headings = self._detect_headings(clean_lines)
        if not headings:
            return []

        # 按标题分组：每个标题及其后续文本
        sections = []
        for i, (line_idx, level, number, heading_text) in enumerate(headings):
            # 确定该章节覆盖的行范围
            start = line_idx + 1
            if i + 1 < len(headings):
                end = headings[i + 1][0] - 1
            else:
                end = len(clean_lines) - 1

            # 收集该章节下的段落
            body_lines = clean_lines[start:end + 1]
            paragraphs = self._extract_paragraphs(body_lines)

            if not paragraphs:
                continue

            full_title = f"{number} {heading_text}" if number else heading_text
            sections.append({
                "level": level,
                "number": number,
                "title": heading_text,
                "full_title": full_title.strip(),
                "paragraphs": paragraphs,
            })

        return sections

    def _detect_headings(self, lines: list[str]) -> list[tuple]:
        """
        检测所有标题行。
        返回: [(line_index, level, number_str, title_text), ...]
        """
        headings = []

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or len(stripped) < 2:
                continue

            for pattern in self.SECTION_PATTERNS:
                m = pattern.match(stripped)
                if m:
                    # 编号标题: "3.2.1 Model Design"
                    if pattern == self.SECTION_PATTERNS[0]:
                        number_str = m.group(1)
                        title_text = m.group(2)
                        level = number_str.count('.') + 1
                        # 过滤纯数字行（不是标题）
                        if len(title_text) < 3 or title_text.isdigit():
                            continue
                        headings.append((idx, level, number_str, title_text))
                        break

                    # 全大写短行: "INTRODUCTION"
                    elif pattern == self.SECTION_PATTERNS[1]:
                        headings.append((idx, 1, "", stripped.title()))
                        break

                    # 关键词标题: "Experiments"
                    elif pattern == self.SECTION_PATTERNS[2]:
                        # 估算层级（常见一级标题）
                        level = 1
                        key_upper = m.group(1).upper()
                        if key_upper in ("ABSTRACT", "INTRODUCTION", "CONCLUSION",
                                          "REFERENCES", "BIBLIOGRAPHY",
                                          "ACKNOWLEDGMENTS", "ACKNOWLEDGEMENTS",
                                          "APPENDIX"):
                            level = 1
                        headings.append((idx, level, "", stripped))
                        break

        # 按行号重排序
        headings.sort(key=lambda x: x[0])
        return headings

    def _extract_paragraphs(self, lines: list[str]) -> list[str]:
        """将行列表合并为段落列表（段落间以空行分隔）"""
        paragraphs = []
        current = []

        for line in lines:
            if line.strip():
                current.append(line.strip())
            else:
                if current:
                    paragraphs.append(' '.join(current))
                    current = []

        if current:
            paragraphs.append(' '.join(current))

        return paragraphs

    # ============================================================
    # 分块构建
    # ============================================================

    def _build_section_chunks(
        self, sections: list[dict], base_metadata: dict
    ) -> list[Chunk]:
        """将章节内容和段落构建为分块"""
        chunks = []

        for sec in sections:
            section_path = sec["full_title"]
            prefix = f"[{section_path}]\n"
            prefix_len = len(prefix)

            # 将段落分组，每组不超过 max_chunk_chars
            groups = self._group_paragraphs(
                sec["paragraphs"], prefix_len
            )

            for group in groups:
                body = '\n\n'.join(group)
                chunk_text = prefix + body
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_index=len(chunks),
                    metadata={
                        **base_metadata,
                        "section": section_path,
                        "section_level": sec["level"],
                        "section_number": sec.get("number", ""),
                        "paragraph_count": len(group),
                        "char_count": len(chunk_text),
                        "chunk_index": len(chunks),
                    },
                ))

        return chunks

    def _group_paragraphs(
        self, paragraphs: list[str], prefix_len: int
    ) -> list[list[str]]:
        """
        将段落分组为适当大小的块。
        短段落合并，超长段落按句子切分。
        """
        groups = []
        current_group = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)

            # 超长段落：按句子切分
            if para_len > PARAGRAPH_CHUNK_LIMIT:
                # 先把当前组保存
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_len = 0

                # 按句子切分超长段落
                sub_parts = self._split_long_paragraph(para, prefix_len)
                for part in sub_parts:
                    groups.append([part])
                continue

            # 正常段落：尝试合并
            if current_len + para_len + prefix_len <= self.max_chunk_chars:
                current_group.append(para)
                current_len += para_len
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [para]
                current_len = para_len

        if current_group:
            groups.append(current_group)

        return groups

    def _split_long_paragraph(
        self, paragraph: str, prefix_len: int
    ) -> list[str]:
        """将超长段落按句子边界切分"""
        target = self.max_chunk_chars - prefix_len
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)

        parts = []
        current = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) <= target:
                current.append(sent)
                current_len += len(sent) + 1  # +1 for space
            else:
                if current:
                    parts.append(' '.join(current))
                # 单句超长的情况，直接按字符切
                if len(sent) > target:
                    parts.append(sent[:target] + "...")
                    current = []
                    current_len = 0
                else:
                    current = [sent]
                    current_len = len(sent)

        if current:
            parts.append(' '.join(current))

        return parts

    def _fallback_chunk(self, text: str, metadata: dict) -> list[Chunk]:
        """未检测到章节结构时，按段落分块"""
        paragraphs = self._extract_paragraphs(text.split('\n'))
        groups = self._group_paragraphs(paragraphs, 0)

        chunks = []
        for group in groups:
            chunks.append(Chunk(
                text='\n\n'.join(group),
                chunk_index=len(chunks),
                metadata={
                    **metadata,
                    "section": "(无章节结构)",
                    "section_level": 0,
                    "char_count": sum(len(p) for p in group),
                    "chunk_index": len(chunks),
                },
            ))

        logger.info(f"回退分块: {sum(len(g) for g in groups)}段落 → {len(chunks)}块")
        return chunks


# 保持向后兼容的别名
TextChunker = SectionChunker
