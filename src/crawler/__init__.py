"""
arXiv 论文爬虫：根据领域配置自动抓取论文元数据并下载 PDF。
"""
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

import arxiv
import httpx

from src.config import (
    PAPERS_DIR, ARXIV_REQUEST_DELAY, ARXIV_MAX_RESULTS_PER_FETCH,
    RESEARCH_DOMAINS
)

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """论文元数据"""
    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    published: datetime
    categories: list[str]
    pdf_url: str
    domain: str  # 爬取时属于哪个研究领域
    local_pdf_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "summary": self.summary,
            "published": self.published.isoformat(),
            "categories": self.categories,
            "pdf_url": self.pdf_url,
            "domain": self.domain,
            "local_pdf_path": self.local_pdf_path,
        }


class ArxivCrawler:
    """arXiv 论文爬虫"""

    def __init__(self, papers_dir: Path = PAPERS_DIR):
        self.papers_dir = Path(papers_dir)
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=ARXIV_REQUEST_DELAY,
            num_retries=3,
        )

    def search_papers(
        self,
        domain: str,
        category: str,
        max_results: int = ARXIV_MAX_RESULTS_PER_FETCH,
        keyword: Optional[str] = None,
    ) -> list[PaperMetadata]:
        """
        搜索指定领域的论文。

        Args:
            domain: 领域名称（如 "大语言模型"）
            category: arXiv 分类（如 "cs.CL"）
            max_results: 最大结果数
            keyword: 可选的额外关键词过滤
        """
        query = f"cat:{category}"
        if keyword:
            query = f"{query} AND {keyword}"

        logger.info(f"搜索领域 [{domain}]，查询: {query}，上限: {max_results} 篇")

        papers = []
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        for result in self.client.results(search):
            paper = PaperMetadata(
                arxiv_id=result.get_short_id(),
                title=result.title.strip(),
                authors=[a.name for a in result.authors],
                summary=result.summary.strip().replace("\n", " "),
                published=result.published.replace(tzinfo=timezone.utc),
                categories=result.categories,
                pdf_url=result.pdf_url,
                domain=domain,
            )
            papers.append(paper)

        logger.info(f"领域 [{domain}] 搜索完成，获取 {len(papers)} 篇论文元数据")
        return papers

    def download_pdf(self, paper: PaperMetadata) -> Optional[Path]:
        """下载论文 PDF 到本地"""
        domain_dir = self.papers_dir / paper.domain.replace("/", "_")
        domain_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = domain_dir / f"{paper.arxiv_id}.pdf"

        if pdf_path.exists():
            logger.debug(f"PDF 已存在: {pdf_path}")
            paper.local_pdf_path = str(pdf_path)
            return pdf_path

        try:
            logger.info(f"下载 PDF: {paper.arxiv_id} - {paper.title[:60]}...")
            paper_pdf = paper.pdf_url if paper.pdf_url.endswith(".pdf") else paper.pdf_url + ".pdf"
            response = httpx.get(paper_pdf, timeout=60, follow_redirects=True)
            response.raise_for_status()

            pdf_path.write_bytes(response.content)
            paper.local_pdf_path = str(pdf_path)
            logger.info(f"PDF 已保存: {pdf_path}")
            return pdf_path

        except Exception as e:
            logger.error(f"下载失败 {paper.arxiv_id}: {e}")
            return None

    def crawl_domain(
        self,
        domain: str,
        category: str,
        download: bool = True,
        **kwargs,
    ) -> list[PaperMetadata]:
        """
        爬取单个领域的论文（搜索 + 下载）。

        Returns:
            成功获取的论文元数据列表
        """
        papers = self.search_papers(domain, category, **kwargs)
        if download:
            for paper in papers:
                self.download_pdf(paper)
        return papers

    def crawl_all_domains(
        self,
        domains: dict[str, str] = None,
        download: bool = True,
    ) -> dict[str, list[PaperMetadata]]:
        """
        批量爬取所有配置的领域。

        Returns:
            {domain_name: [papers]} 字典
        """
        if domains is None:
            domains = RESEARCH_DOMAINS

        results = {}
        for domain_name, category in domains.items():
            logger.info(f"=== 开始爬取领域: {domain_name} ({category}) ===")
            papers = self.crawl_domain(domain_name, category, download=download)
            results[domain_name] = papers
            time.sleep(ARXIV_REQUEST_DELAY)  # 领域之间间隔

        total = sum(len(v) for v in results.values())
        logger.info(f"全部领域爬取完成，共获取 {total} 篇论文")
        return results

    def get_stored_papers(self) -> dict[str, list[Path]]:
        """获取本地已存储的所有论文 PDF 路径"""
        stored = {}
        if not self.papers_dir.exists():
            return stored

        for domain_dir in self.papers_dir.iterdir():
            if domain_dir.is_dir():
                pdfs = list(domain_dir.glob("*.pdf"))
                if pdfs:
                    stored[domain_dir.name] = pdfs

        return stored
