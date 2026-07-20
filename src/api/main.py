"""
FastAPI 后端服务：提供 RESTful API 接口。
"""
import logging
import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import API_HOST, API_PORT
from src.agents.system import get_system, MultiAgentSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="多Agent协作研究平台",
    description="基于 RAG + Multi-Agent 的学术论文研究助手",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- 请求/响应模型 ----

class QueryRequest(BaseModel):
    question: str = Field(..., description="用户提问")
    enable_review: bool = Field(default=True, description="是否启用审查Agent复核")


class CrawlRequest(BaseModel):
    domain: Optional[str] = Field(default=None, description="指定领域（空=全部）")
    schedule: bool = Field(default=False, description="是否启用定时调度")


class QueryResponse(BaseModel):
    answer: str
    stats: Optional[dict] = None


class StatsResponse(BaseModel):
    status: str
    data: dict


class IngestResponse(BaseModel):
    message: str
    result: dict


# ---- 系统初始化 ----

system: MultiAgentSystem = None


@app.on_event("startup")
async def startup():
    global system
    logger.info("正在初始化多Agent协作系统...")
    system = get_system()
    logger.info("系统启动完成")


# ---- API 路由 ----

@app.get("/")
async def root():
    return {
        "service": "多Agent协作研究平台",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """用户提问接口：检索 → 分析 → 审查 → 综合回答"""
    if system is None:
        raise HTTPException(503, "系统尚未初始化")

    try:
        answer = await asyncio.to_thread(
            system.query, req.question, req.enable_review
        )
        return QueryResponse(
            answer=answer,
            stats=system.get_stats(),
        )
    except Exception as e:
        logger.error(f"查询失败: {e}", exc_info=True)
        raise HTTPException(500, f"查询处理失败: {str(e)}")


@app.post("/crawl")
async def trigger_crawl(req: CrawlRequest):
    """手动触发论文爬取"""
    if system is None:
        raise HTTPException(503, "系统尚未初始化")

    try:
        if req.domain:
            result = await asyncio.to_thread(system.crawl_domain, req.domain)
        else:
            result = await asyncio.to_thread(system.update_knowledge_base)

        return {"message": "爬取完成", "result": result}
    except Exception as e:
        logger.error(f"爬取失败: {e}", exc_info=True)
        raise HTTPException(500, f"爬取失败: {str(e)}")


@app.post("/ingest")
async def ingest_papers():
    """将本地已下载的论文入链"""
    if system is None:
        raise HTTPException(503, "系统尚未初始化")

    try:
        result = await asyncio.to_thread(system.rag_pipeline.ingest_papers)
        return IngestResponse(
            message="入链完成",
            result=result,
        )
    except Exception as e:
        logger.error(f"入链失败: {e}", exc_info=True)
        raise HTTPException(500, f"入链失败: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """获取系统状态"""
    if system is None:
        raise HTTPException(503, "系统尚未初始化")

    return StatsResponse(
        status="running",
        data=system.get_stats(),
    )


@app.post("/scheduler/start")
async def start_scheduler():
    """启动定时爬虫"""
    if system is None:
        raise HTTPException(503, "系统尚未初始化")

    system.start_scheduler()
    return {"message": "定时爬虫已启动",
            "interval_hours": system.scheduler.interval_hours}


@app.post("/scheduler/stop")
async def stop_scheduler():
    """停止定时爬虫"""
    if system is None:
        raise HTTPException(503, "系统尚未初始化")

    system.stop_scheduler()
    return {"message": "定时爬虫已停止"}


# ---- 启动入口 ----

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
