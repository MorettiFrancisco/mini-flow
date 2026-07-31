"""API HTTP. Servicio de larga vida: varios traces concurrentes, cada uno con su
user_id y session_id, que es el caso realista que muestra la UI de Langfuse.
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field

from app import graph, obs, settings, vectorstore

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Ollama en CPU serializa de verdad: sin este tope, 10 requests HTTP encolan
# decenas de inferencias y todas dan timeout.
_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENCY)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info(
        "listo | modelo=%s embeddings=%s chroma=%s:%s langfuse=%s",
        settings.CHAT_MODEL,
        settings.EMBED_MODEL,
        settings.CHROMA_HOST,
        settings.CHROMA_PORT,
        "on" if obs.ENABLED else "off",
    )
    yield
    # Único flush del proceso. Por request no: mataría el batching del SDK.
    obs.shutdown()


app = FastAPI(title="mini-flow", version="0.2.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    user_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    judgement: dict | None = None
    rewrites: int = 0
    retrieved: list[dict] = []
    graded: list[dict] = []
    trace_id: str | None = None
    session_id: str


@app.get("/health")
async def health() -> dict:
    """Chequea las dos dependencias duras. Lo usa el healthcheck del compose."""
    try:
        docs = await asyncio.to_thread(vectorstore.count)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"chroma no responde: {exc}") from exc
    if docs == 0:
        raise HTTPException(503, "la colección de Chroma está vacía: corré el bootstrap")
    return {"status": "ok", "documents": docs, "tracing": obs.ENABLED}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    session_id = req.session_id or f"s-{uuid.uuid4().hex[:8]}"
    async with _semaphore:
        try:
            result = await graph.run_query(
                req.question, user_id=req.user_id, session_id=session_id
            )
        except GraphRecursionError as exc:
            raise HTTPException(500, "el grafo alcanzó el recursion_limit") from exc
    return AskResponse(session_id=session_id, **result)
