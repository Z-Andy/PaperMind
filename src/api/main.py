"""
FastAPI 后端服务：提供 RESTful API 接口。
"""
import json
import logging
import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import API_HOST, API_PORT
from src.agents.system import get_system, MultiAgentSystem
from src.metrics import get_metrics

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
    conversation_id: Optional[str] = Field(
        default=None, description="多轮对话ID，同一会话使用相同ID以保持上下文"
    )


class CrawlRequest(BaseModel):
    domain: Optional[str] = Field(default=None, description="指定领域（空=全部）")
    max_results: int = Field(default=20, ge=1, le=200, description="每个领域最多爬取篇数")
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


@app.get("/health")
async def health():
    """健康检查：验证 LLM API 和知识库状态"""
    if system is None:
        return {"status": "initializing", "llm": None, "kb_size": 0}
    result = system.check_health()
    result["status"] = "healthy" if not result["issues"] else "degraded"
    return result


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """用户提问接口：检索 → 分析 → 审查 → 综合回答"""
    if system is None:
        raise HTTPException(503, "系统尚未初始化")

    health = system.check_health()
    if health["kb_size"] == 0:
        raise HTTPException(400, "知识库为空，请先更新知识库后再提问")

    try:
        answer = await asyncio.to_thread(
            system.query, req.question, req.enable_review, req.conversation_id
        )
        return QueryResponse(
            answer=answer,
            stats=system.get_stats(),
        )
    except Exception as e:
        logger.error(f"查询失败: {e}", exc_info=True)
        raise HTTPException(500, f"查询处理失败: {str(e)}")


@app.post("/crawl/cancel")
async def cancel_crawl():
    """取消正在进行的论文爬取"""
    if system is None:
        raise HTTPException(503, "系统尚未初始化")
    system.crawler.cancel()
    return {"message": "已发送取消信号，爬取将在当前下载完成后停止"}


@app.post("/crawl")
async def trigger_crawl(req: CrawlRequest):
    """手动触发论文爬取"""
    if system is None:
        raise HTTPException(503, "系统尚未初始化")

    try:
        if req.domain:
            result = await asyncio.to_thread(
                system.crawl_domain, req.domain, req.max_results
            )
        else:
            result = await asyncio.to_thread(
                system.update_knowledge_base, None, req.max_results
            )

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


@app.get("/metrics")
async def get_metrics_endpoint():
    """获取性能指标：响应时间、缓存命中率、各步骤耗时分布"""
    metrics = get_metrics()
    return metrics.get_summary()


@app.get("/metrics/recent")
async def get_recent_queries(limit: int = 20):
    """获取最近 N 条查询的明细日志"""
    metrics = get_metrics()
    return {"queries": metrics.get_recent_queries(limit)}


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


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """流式查询接口 (SSE)：逐 Agent 返回进度和最终结果"""
    if system is None:
        raise HTTPException(503, "系统尚未初始化")

    async def event_generator():
        try:
            async for event in system.query_stream(
                req.question, req.enable_review, req.conversation_id
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"流式查询失败: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---- 启动入口 ----

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
