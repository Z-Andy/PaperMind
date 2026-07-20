"""
向量化模块：支持本地 sentence-transformers 模型和远程 API 两种方式。
"""
import logging
from typing import Optional

import numpy as np
from openai import OpenAI

from src.config import (
    EMBEDDING_PROVIDER, LOCAL_EMBEDDING_MODEL,
    EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


class Embedder:
    """文本向量化器"""

    def __init__(
        self,
        model: str = None,
        provider: str = EMBEDDING_PROVIDER,
        api_key: str = EMBEDDING_API_KEY,
        base_url: str = EMBEDDING_BASE_URL,
    ):
        self.provider = provider
        self.model_name = model or (
            LOCAL_EMBEDDING_MODEL if provider == "local" else EMBEDDING_MODEL
        )
        self._model = None
        self._dimension: Optional[int] = None

        if provider == "local":
            self._init_local_model()
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)

        logger.info(f"Embedder 初始化: provider={provider}, model={self.model_name}")

    def _init_local_model(self):
        """初始化本地 sentence-transformers 模型"""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"加载本地 Embedding 模型: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"本地模型加载完成，维度: {self._dimension}")
        except ImportError:
            raise ImportError(
                "使用本地模型需要安装 sentence-transformers，请执行: "
                "pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"本地模型加载失败: {e}")
            raise

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        批量将文本转换为向量。

        Args:
            texts: 文本列表
            batch_size: 每批处理的文本数

        Returns:
            向量列表，每个向量为 float 列表
        """
        if not texts:
            return []

        if self.provider == "local":
            return self._embed_local(texts, batch_size)
        else:
            return self._embed_api(texts, batch_size)

    def _embed_local(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """使用本地模型向量化"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # sentence-transformers 自动处理 batching
                embeddings = self._model.encode(
                    batch,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                # numpy array → list[float]
                if isinstance(embeddings, np.ndarray):
                    embeddings = embeddings.tolist()
                all_embeddings.extend(embeddings)

                if self._dimension is None and all_embeddings:
                    self._dimension = len(all_embeddings[0])

                logger.debug(f"本地向量化批次 {i // batch_size + 1}: {len(batch)}条")
            except Exception as e:
                logger.error(f"本地向量化失败: {e}")
                dim = self._dimension or 384
                for _ in batch:
                    all_embeddings.append([0.0] * dim)

        logger.info(f"向量化完成: {len(texts)}条文本 → {len(all_embeddings)}个向量")
        return all_embeddings

    def _embed_api(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """使用远程 API 向量化"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                )
                batch_embeddings = [d.embedding for d in response.data]
                all_embeddings.extend(batch_embeddings)

                if self._dimension is None and batch_embeddings:
                    self._dimension = len(batch_embeddings[0])

                logger.debug(f"API向量化批次 {i // batch_size + 1}: {len(batch)}条")
            except Exception as e:
                logger.error(f"API向量化失败: {e}")
                dim = self._dimension or 1536
                for _ in batch:
                    all_embeddings.append([0.0] * dim)

        logger.info(f"向量化完成: {len(texts)}条文本 → {len(all_embeddings)}个向量")
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """将单个查询文本向量化"""
        embeddings = self.embed_texts([query])
        return embeddings[0] if embeddings else []

    @property
    def dimension(self) -> int:
        if self._dimension:
            return self._dimension
        if self.provider == "local":
            return 384  # bge-small 默认维度
        return 1536
