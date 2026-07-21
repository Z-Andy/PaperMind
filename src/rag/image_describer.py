"""
图片描述器：使用 Gemini 多模态模型理解论文中的图表。
"""
import io
import logging
from typing import Optional

import fitz  # pymupdf

from src.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

DESCRIBE_PROMPT = """You are analyzing a figure from an academic paper.
Describe what this figure shows in detail. Include:
1. The type of figure (architecture diagram, chart, table, algorithm flow, etc.)
2. Key elements and their relationships
3. Any notable data, numbers, or trends visible
4. The main conclusion a reader should draw from this figure

Write in the same language as the paper. Be concise but thorough."""


class ImageDescriber:
    """使用 Gemini 多模态模型描述 PDF 中的图片"""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model or GEMINI_MODEL
        self._client = None
        self._available = bool(self.api_key)

        if self._available:
            logger.info(f"图片描述器已启用: model={self.model_name}")
        else:
            logger.info("未配置 GEMINI_API_KEY，图片描述功能禁用")

    @property
    def client(self):
        """延迟初始化 Gemini 客户端"""
        if self._client is None and self._available:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                logger.error(
                    "需要安装 google-genai 包: uv pip install google-genai"
                )
                self._available = False
            except Exception as e:
                logger.error(f"Gemini 客户端初始化失败: {e}")
                self._available = False
        return self._client

    @property
    def available(self) -> bool:
        return self._available and self.client is not None

    def describe_image(
        self,
        page: fitz.Page,
        bbox: tuple,
        caption: str = "",
    ) -> Optional[str]:
        """
        渲染页面指定区域为图片，发送给 Gemini 生成描述。

        Args:
            page: PyMuPDF Page 对象
            bbox: 图片区域 (x0, y0, x1, y1)
            caption: 图片已有的标题文字（辅助上下文）

        Returns:
            图片描述文字，失败返回 None
        """
        if not self.available:
            return None

        try:
            # ---- 1. 渲染页面区域为 PNG ----
            x0, y0, x1, y1 = bbox

            # 稍微扩大裁剪区域（5% padding），避免裁到内容边缘
            pad_x = (x1 - x0) * 0.05
            pad_y = (y1 - y0) * 0.05
            clip = fitz.Rect(
                max(0, x0 - pad_x),
                max(0, y0 - pad_y),
                min(page.rect.width, x1 + pad_x),
                min(page.rect.height, y1 + pad_y),
            )

            # 分辨率：矩阵系数 2.0 = 144 DPI 左右
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, clip=clip)
            img_bytes = pix.tobytes("png")

            # ---- 2. 调用 Gemini ----
            prompt = DESCRIBE_PROMPT
            if caption:
                prompt += f"\n\nThe figure caption is: \"{caption}\""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    prompt,
                    {"inline_data": {"mime_type": "image/png", "data": img_bytes}},
                ],
            )

            description = response.text.strip()
            logger.debug(f"图片描述生成成功: {description[:80]}...")
            return description

        except Exception as e:
            logger.warning(f"图片描述失败: {e}")
            return None

    def describe_all_images(
        self,
        page: fitz.Page,
        image_items: list[dict],
    ) -> dict:
        """
        批量描述页面中所有图片。

        Args:
            page: PyMuPDF Page 对象
            image_items: content_items 中 type="image" 的项目列表

        Returns:
            以 id(image_item) 为 key 的描述字典
        """
        descriptions = {}
        for item in image_items:
            bbox = (item["x0"], item["y0"], item["x1"], item.get("y1", item["y0"] + 200))
            caption = item.get("caption", "")
            desc = self.describe_image(page, bbox, caption)
            if desc:
                descriptions[id(item)] = desc
        return descriptions
