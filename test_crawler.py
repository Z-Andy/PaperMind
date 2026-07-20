"""
快速验证脚本：测试爬虫模块是否正常工作。
无需 API Key 即可运行，仅测试论文元数据抓取（不下载 PDF）。
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from src.crawler import ArxivCrawler


def test_crawler():
    """测试爬虫基础功能"""
    print("=" * 50)
    print("  arXiv 爬虫模块测试")
    print("=" * 50)

    crawler = ArxivCrawler()

    # 测试单领域搜索（只抓元数据，不下载）
    print("\n[测试1] 搜索 cs.AI 领域论文（仅元数据）...")
    papers = crawler.search_papers(
        domain="人工智能",
        category="cs.AI",
        max_results=5,
    )

    print(f"\n获取到 {len(papers)} 篇论文:")
    for i, p in enumerate(papers[:3]):
        print(f"\n  [{i + 1}] {p.title}")
        print(f"      作者: {', '.join(p.authors[:3])}{'...' if len(p.authors) > 3 else ''}")
        print(f"      日期: {p.published.strftime('%Y-%m-%d')}")
        print(f"      摘要: {p.summary[:120]}...")
        print(f"      PDF: {p.pdf_url}")

    # 测试下载一篇
    if papers:
        print(f"\n[测试2] 下载论文 PDF: {papers[0].arxiv_id}...")
        pdf_path = crawler.download_pdf(papers[0])
        if pdf_path:
            print(f"  PDF 已保存: {pdf_path} ({pdf_path.stat().st_size} bytes)")
        else:
            print("  下载失败（可能是网络问题）")

    print("\n[测试3] 检查本地已存储论文:")
    stored = crawler.get_stored_papers()
    if stored:
        for domain, pdfs in stored.items():
            print(f"  {domain}: {len(pdfs)} 篇")
    else:
        print("  （尚未下载任何论文）")

    print("\n" + "=" * 50)
    print("  爬虫模块测试完成!")
    print("=" * 50)
    print("\n提示: 设置 OPENAI_API_KEY 后运行 start.bat 即可启动完整系统")


if __name__ == "__main__":
    test_crawler()
