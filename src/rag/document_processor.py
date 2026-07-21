"""
文档处理器：解析 PDF 论文，提取文本、表格和图片说明。
支持 Gemini 多模态模型对图片进行智能描述。
"""
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import fitz  # pymupdf

from src.rag.image_describer import ImageDescriber

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """解析后的文档"""
    file_path: str
    content: str
    title: str = ""
    page_count: int = 0
    metadata: dict = field(default_factory=dict)


class DocumentProcessor:
    """PDF 文档解析器，支持文本、表格、图片说明提取"""

    # 图片标题关键词（中英文）
    CAPTION_KEYWORDS = [
        "Figure", "Fig.", "Fig ", "figure", "fig.", "fig ",
        "图", "Figure", "FIGURE",
        "Table", "TABLE", "表",
    ]

    @staticmethod
    def parse_pdf(file_path: Path | str) -> Optional[Document]:
        """
        解析单个 PDF 文件，提取文本 + 表格 + 图片说明。

        Args:
            file_path: PDF 文件路径

        Returns:
            Document 对象，解析失败返回 None
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return None

        try:
            doc = fitz.open(str(file_path))
            full_text_parts = []
            title = ""
            total_images = 0
            total_tables = 0
            total_described = 0

            # 初始化图片描述器（如果配置了 GEMINI_API_KEY）
            describer = ImageDescriber()

            for i, page in enumerate(doc):
                page_content, img_count, tbl_count = (
                    DocumentProcessor._extract_page_content(page, describer)
                )
                total_images += img_count
                total_tables += tbl_count

                if page_content.strip():
                    full_text_parts.append(
                        f"--- 第{i + 1}页 ---\n{page_content}"
                    )

                # 从第一页提取标题（字体最大的文本）
                if i == 0:
                    blocks = page.get_text("dict")["blocks"]
                    max_size = 0
                    for block in blocks:
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    if (
                                        span["size"] > max_size
                                        and span["text"].strip()
                                    ):
                                        max_size = span["size"]
                                        title = span["text"].strip()

            if not title and full_text_parts:
                title = full_text_parts[0].strip()[:100]

            content = "\n\n".join(full_text_parts)
            total_pages = len(doc)
            doc.close()

            logger.info(
                f"解析完成: {file_path.name} | {total_pages}页 | "
                f"{len(content)}字符 | 图片: {total_images} | 表格: {total_tables}"
                f"{' | Gemini图片描述: 已启用' if describer.available else ''}"
            )
            return Document(
                file_path=str(file_path),
                content=content,
                title=title,
                page_count=total_pages,
                metadata={
                    "filename": file_path.stem,
                    "arxiv_id": file_path.stem,
                    "image_count": total_images,
                    "table_count": total_tables,
                },
            )

        except Exception as e:
            logger.error(f"解析失败 {file_path.name}: {e}")
            return None

    @staticmethod
    def _extract_page_content(page, describer=None) -> tuple[str, int, int]:
        """
        提取单页内容：文本 + 表格 + 图片说明，
        自动检测单栏/双栏布局并按正确阅读顺序排列。

        Args:
            page: PyMuPDF Page 对象
            describer: 可选的 ImageDescriber，用于多模态图片理解

        Returns:
            (content_str, image_count, table_count)
        """
        blocks = page.get_text("dict")["blocks"]
        page_width = page.rect.width
        content_items: list[dict] = []

        # ---- 1. 收集文本块和图片块（带位置信息） ----
        # Type 1 = 光栅图像，但学术论文中矢量图形不会被标为 type 1
        seen_image_bboxes = set()

        for block in blocks:
            bbox = block["bbox"]   # (x0, y0, x1, y1)
            y0, x0, x1 = bbox[1], bbox[0], bbox[2]

            if block["type"] == 0:  # 文本块
                lines_text = []
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        lines_text.append(span["text"])
                text = "".join(lines_text).strip()
                if text:
                    content_items.append({
                        "type": "text",
                        "y": y0,
                        "x0": x0,
                        "x1": x1,
                        "content": text,
                    })

            elif block["type"] == 1:  # 光栅图片块
                content_items.append({
                    "type": "image",
                    "y": y0,
                    "x0": x0,
                    "x1": x1,
                    "y1": bbox[3],
                    "caption": None,
                    "description": None,
                })
                seen_image_bboxes.add((x0, y0, x1, bbox[3]))

        # ---- 1.5 补充检测：page.get_images() 扫描所有嵌入图像 ----
        # 这对矢量图无效（矢量图不是"图像"而是绘图指令），
        # 但能捕获被文字环绕的嵌入光栅图
        for img_info in page.get_images(full=True):
            try:
                # 获取图像在页面上的位置
                img_rects = page.get_image_rects(img_info[7])
                for rect in img_rects:
                    key = (rect.x0, rect.y0, rect.x1, rect.y1)
                    # 去重：已通过 type=1 捕获的不重复
                    if key in seen_image_bboxes:
                        continue
                    seen_image_bboxes.add(key)
                    content_items.append({
                        "type": "image",
                        "y": rect.y0,
                        "x0": rect.x0,
                        "x1": rect.x1,
                        "y1": rect.y1,
                        "caption": None,
                        "description": None,
                    })
            except Exception:
                pass  # 旧版 PyMuPDF 可能不支持 get_image_rects

        # ---- 2. 收集表格 ----
        table_count = 0
        try:
            tables_result = page.find_tables()
            if tables_result:
                table_list = None
                if hasattr(tables_result, "tables"):
                    table_list = tables_result.tables
                else:
                    table_list = list(tables_result)

                if table_list:
                    for table in table_list:
                        markdown = DocumentProcessor._table_to_markdown(table)
                        if markdown:
                            tbl_bbox = table.bbox
                            content_items.append({
                                "type": "table",
                                "y": tbl_bbox[1],
                                "x0": tbl_bbox[0],
                                "x1": tbl_bbox[2],
                                "content": markdown,
                            })
                            table_count += 1
        except Exception as e:
            logger.debug(f"表格提取异常: {e}")

        # ---- 3. 布局感知排序（替代简单 y 排序） ----
        DocumentProcessor._sort_by_layout(content_items, page_width)

        # ---- 3.5 矢量图间隙检测 ----
        # 文本块间的大空隙（>120pt）很可能是矢量图/表格，
        # 在间隙中插入占位符
        text_items = [it for it in content_items if it["type"] == "text"]
        new_images = []
        for i in range(len(text_items) - 1):
            gap = text_items[i + 1]["y"] - text_items[i]["y"]
            if gap > 120:
                mid_y = (text_items[i]["y"] + text_items[i + 1]["y"]) / 2
                new_images.append({
                    "type": "image",
                    "y": mid_y,
                    "x0": 50,
                    "x1": page_width - 50,
                    "y1": mid_y + 5,
                    "caption": None,
                    "description": None,
                })
        if new_images:
            content_items.extend(new_images)
            content_items.sort(key=lambda x: x["y"])

        # ---- 4. 匹配图片标题 ----
        image_count = sum(
            1 for item in content_items if item["type"] == "image"
        )
        DocumentProcessor._match_image_captions(content_items)

        # ---- 4.5 多模态图片描述（如果启用了 Gemini） ----
        if describer is not None and describer.available and image_count > 0:
            for item in content_items:
                if item["type"] != "image":
                    continue
                bbox = (item["x0"], item["y"], item["x1"], item["y1"])
                caption = item.get("caption", "")
                desc = describer.describe_image(page, bbox, caption)
                if desc:
                    item["description"] = desc
                    logger.debug(f"图片描述: {desc[:60]}...")

        # ---- 5. 拼装最终文本 ----
        parts = []
        for item in content_items:
            if item["type"] == "image":
                caption = item.get("caption", "")
                description = item.get("description", "")
                if description:
                    # Gemini 成功描述了图片内容
                    heading = f"[图片描述]"
                    if caption:
                        heading += f" 标题: {caption}"
                    parts.append(f"{heading}\n{description}")
                elif caption:
                    parts.append(f"[图片说明: {caption}]")
                else:
                    parts.append(f"[图片]")
            elif item["type"] == "text":
                if item.get("is_caption"):
                    continue
                parts.append(item["content"])
            elif item["type"] == "table":
                parts.append(item["content"])

        result = "\n".join(p for p in parts if p)
        return result, image_count, table_count

    # ---- 多栏检测与排序 ----

    @staticmethod
    def _detect_columns(content_items: list[dict], page_width: float):
        """
        检测页面是否为多栏布局，并找到栏分割线。

        原理：统计所有文本块的 x 中心点分布，如果存在明显双峰
        （两组 x 中心之间间距 > 45pt），则判定为双栏。

        Returns:
            (is_multi_column, divider_x)
            单栏时返回 (False, None)
        """
        # 只取文本块的 x 中心点
        x_centers = []
        for item in content_items:
            if item["type"] == "text":
                x_center = (item["x0"] + item["x1"]) / 2
                x_centers.append(x_center)

        if len(x_centers) < 4:
            return False, None

        x_centers.sort()

        # 寻找最大间隙
        best_gap = 0
        best_divider = None
        best_left_n = 0

        for i in range(len(x_centers) - 1):
            gap = x_centers[i + 1] - x_centers[i]
            if gap > best_gap and gap > 40:
                left_n = i + 1
                right_n = len(x_centers) - left_n
                # 两边至少各有 2 个文本块才认为是栏
                if left_n >= 2 and right_n >= 2:
                    best_gap = gap
                    best_divider = (x_centers[i] + x_centers[i + 1]) / 2
                    best_left_n = left_n

        if best_divider:
            logger.debug(
                f"检测到双栏布局: divider={best_divider:.0f}, "
                f"左栏{best_left_n}块, 右栏{len(x_centers) - best_left_n}块"
            )
            return True, best_divider

        return False, None

    @staticmethod
    def _sort_by_layout(content_items: list[dict], page_width: float):
        """
        按版面结构排序：

        - 单栏页面 → 按 y 坐标从上到下
        - 多栏页面 → 先通栏内容（标题/摘要/跨栏图表），
          再左栏（从上到下），再右栏（从上到下）
        """
        if not content_items:
            return

        is_multi, divider_x = DocumentProcessor._detect_columns(
            content_items, page_width
        )

        if not is_multi:
            # 单栏：直接按 y 排序
            content_items.sort(key=lambda x: x["y"])
            return

        # ---- 多栏：分类 + 排序 ----
        full_width_items = []
        left_items = []
        right_items = []

        for item in content_items:
            item_width = item["x1"] - item["x0"]
            x_center = (item["x0"] + item["x1"]) / 2

            # 跨栏判断：宽度 > 页宽 65%，或横跨分割线两侧
            if item_width > page_width * 0.65:
                full_width_items.append(item)
            elif x_center < divider_x:
                left_items.append(item)
            else:
                right_items.append(item)

        # 各分组内按 y 排序
        full_width_items.sort(key=lambda x: x["y"])
        left_items.sort(key=lambda x: x["y"])
        right_items.sort(key=lambda x: x["y"])

        # 合并：按 y 位置将通栏内容插入到栏内容之间
        sorted_items = []
        fi = 0  # full-width 指针
        ci = 0  # column (左+右合并) 指针

        # 将左右栏按 y 交错合并（同 y 时左在前）
        column_items = []
        li = ri = 0
        while li < len(left_items) and ri < len(right_items):
            if left_items[li]["y"] <= right_items[ri]["y"]:
                column_items.append(left_items[li])
                li += 1
            else:
                column_items.append(right_items[ri])
                ri += 1
        column_items.extend(left_items[li:])
        column_items.extend(right_items[ri:])

        # 将通栏块按 y 插入到列块之间
        while fi < len(full_width_items) and ci < len(column_items):
            if full_width_items[fi]["y"] < column_items[ci]["y"]:
                sorted_items.append(full_width_items[fi])
                fi += 1
            else:
                sorted_items.append(column_items[ci])
                ci += 1
        sorted_items.extend(full_width_items[fi:])
        sorted_items.extend(column_items[ci:])

        content_items.clear()
        content_items.extend(sorted_items)

    @staticmethod
    def _match_image_captions(content_items: list[dict]):
        """
        为图片块匹配标题：查找图片下方紧邻的文本块，
        如果以 Figure/Fig/图 等关键词开头，则认作标题。
        """
        for i, item in enumerate(content_items):
            if item["type"] != "image":
                continue

            # 搜索图片后 1~3 个文本块
            for j in range(i + 1, min(i + 4, len(content_items))):
                next_item = content_items[j]
                if next_item["type"] != "text":
                    continue

                candidate = next_item.get("content", "")
                if not candidate:
                    continue

                # 检查是否以标题关键词开头
                if any(
                    candidate.strip().startswith(kw)
                    for kw in DocumentProcessor.CAPTION_KEYWORDS
                ):
                    item["caption"] = candidate
                    # 标记为已用作标题，避免重复出现在正文中
                    next_item["is_caption"] = True
                    break

    @staticmethod
    def _table_to_markdown(table) -> str:
        """
        将 PyMuPDF 表格转为 Markdown 格式字符串。

        Args:
            table: pymupdf.table.Table 对象

        Returns:
            Markdown 表格字符串，提取失败返回空字符串
        """
        try:
            data = table.extract()
            if not data or len(data) < 1:
                return ""

            max_cols = max(len(row) for row in data)
            lines = []

            for row_idx, row in enumerate(data):
                # 补齐到最大列数，清理换行符
                cleaned = []
                for cell in row:
                    c = str(cell).replace("\n", " ").strip() if cell is not None else ""
                    cleaned.append(c)
                while len(cleaned) < max_cols:
                    cleaned.append("")

                lines.append("| " + " | ".join(cleaned) + " |")
                if row_idx == 0:
                    lines.append("| " + " | ".join(["---"] * max_cols) + " |")

            return "[表格]\n" + "\n".join(lines)
        except Exception as e:
            logger.debug(f"表格格式化异常: {e}")
            return ""

    @staticmethod
    def parse_directory(papers_dir: Path | str) -> list[Document]:
        """
        批量解析目录下所有 PDF 文件。

        Returns:
            成功解析的文档列表
        """
        papers_dir = Path(papers_dir)
        if not papers_dir.exists():
            logger.warning(f"目录不存在: {papers_dir}")
            return []

        documents = []
        pdf_files = list(papers_dir.rglob("*.pdf"))
        logger.info(f"开始批量解析: {len(pdf_files)} 个 PDF 文件")

        for pdf_path in pdf_files:
            doc = DocumentProcessor.parse_pdf(pdf_path)
            if doc:
                domain = pdf_path.parent.name
                doc.metadata["domain"] = domain
                documents.append(doc)

        logger.info(f"批量解析完成: {len(documents)}/{len(pdf_files)} 成功")
        return documents
