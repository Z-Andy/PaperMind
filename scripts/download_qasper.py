#!/usr/bin/env python
"""
下载 QASPER 数据集到 data/qasper/ 目录。
QASPER: 1585 篇 arXiv NLP 论文 + 5049 个专家标注问题。

用法：
  # 方式1：通过 HuggingFace 镜像下载（国内推荐）
  set HF_ENDPOINT=https://hf-mirror.com
  python scripts/download_qasper.py

  # 方式2：直接下载
  python scripts/download_qasper.py

来源: https://huggingface.co/datasets/allenai/qasper
"""

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
QASPER_DIR = PROJECT_ROOT / "data" / "qasper"

# 国内用户设置环境变量加速下载
HF_MIRROR = "https://hf-mirror.com"


def main():
    QASPER_DIR.mkdir(parents=True, exist_ok=True)

    # 检查是否已设置镜像
    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if not hf_endpoint:
        logger.info(
            "未设置 HF_ENDPOINT 环境变量。"
            "如在国内，建议先运行:\n"
            "  set HF_ENDPOINT=https://hf-mirror.com"
        )
        # 自动尝试镜像
        logger.info("自动使用镜像下载...")
        os.environ["HF_ENDPOINT"] = HF_MIRROR

    # 检查依赖
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "缺少 datasets 库。请先安装:\n"
            "  pip install datasets pyarrow\n"
            "国内用户建议使用镜像:\n"
            "  pip install datasets pyarrow -i https://pypi.tuna.tsinghua.edu.cn/simple"
        )
        sys.exit(1)

    logger.info("正在从 HuggingFace 加载 QASPER 数据集...")
    logger.info(f"  HF_ENDPOINT = {os.environ.get('HF_ENDPOINT', '默认')}")

    for split in ["train", "validation"]:
        output_file = QASPER_DIR / f"qasper_{split}.jsonl"

        if output_file.exists() and output_file.stat().st_size > 1000:
            logger.info(f"  已存在，跳过: {output_file.name}")
            continue

        logger.info(f"  加载 {split} split...")
        try:
            dataset = load_dataset(
                "allenai/qasper", split=split,
                cache_dir=str(QASPER_DIR / "cache"),
                trust_remote_code=True,
            )
        except Exception as e:
            logger.error(f"加载 {split} 失败: {e}")
            continue

        # 保存为 JSONL（每行一个 JSON 对象）
        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for paper in dataset:
                f.write(json.dumps(paper, ensure_ascii=False) + "\n")
                count += 1

        size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"  已保存: {output_file.name} ({count} 篇论文, {size_mb:.1f} MB)")

    # 验证
    files = list(QASPER_DIR.glob("*.jsonl"))
    if files:
        logger.info(f"\n下载完成！共 {len(files)} 个文件")
        for f in files:
            logger.info(f"  {f.name}")
        logger.info(f"\n评估命令: python scripts/evaluate.py --mode qasper")
    else:
        logger.error("下载失败，请检查网络。国内用户请确保已设置 HF_ENDPOINT 环境变量")


if __name__ == "__main__":
    main()
