"""Nodos del grafo. Todos async: es lo que mantiene el contexto de OpenTelemetry
(y por lo tanto el anidado de spans) intacto en el fan-out del grader.

Cada nodo devuelve su payload de debug en el estado. El CallbackHandler serializa
lo devuelto como output del span del nodo, así que los documentos recuperados y
descartados quedan en el trace sin llamadas extra al SDK.
"""

import asyncio
import logging

from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from app import obs, prompts, settings, vectorstore
from app.state import RAGState

log = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = (
    "No encontré información sobre eso en los documentos indexados. "
    "Probá reformular la pregunta o revisá que el corpus en data/ cubra el tema."
)


class Grade(BaseModel):
    """Schema chato y diminuto: es lo que hace confiable la salida estructurada
    en un modelo de 3B."""

    keep: bool = Field(description="true si el documento ayuda a responder la pregunta")
    why: str = Field(description="justificación de 15 palabras o menos")


def _label(doc, i: int) -> str:
    return f"{doc.metadata.get('source', '?')}#{doc.metadata.get('chunk', i)}"


async def retrieve(state: RAGState) -> dict:
    query = state.get("query") or state["question"]
    hits = await vectorstore.store().asimilarity_search_with_relevance_scores(
        query, k=settings.TOP_K
    )
    return {
        "query": query,
        "docs": [doc for doc, _ in hits],
        "n_retrieved": len(hits),
        "retrieval_debug": [
            {
                "source": doc.metadata.get("source"),
                "chunk": doc.metadata.get("chunk"),
                "score": round(score, 4),
                "preview": doc.page_content[:120],
            }
            for doc, score in hits
        ],
    }


async def grade_documents(state: RAGState) -> dict:
    """Un LLM decide por documento. Fan-out con asyncio.gather y un span por doc.

    asyncio.gather y no un ThreadPoolExecutor: los contextvars se copian por task,
    así que cada span queda colgado del padre correcto. Con threads pelados el
    contexto de OTel no cruza el borde y los spans se irían a un trace huérfano.
    """
    docs = state.get("docs", [])
    if not docs:
        return {"kept": [], "grade_debug": [], "n_kept": 0, "n_dropped": 0}

    chain = prompts.template("rag/grade-document") | settings.chat_llm().with_structured_output(
        Grade
    ).with_retry(stop_after_attempt=2)
    question = state["question"]

    async def grade_one(i: int, doc):
        label = _label(doc, i)
        with obs.node_span(
            f"grade:{label}",
            input={"question": question, "preview": doc.page_content[:200]},
        ) as span:
            try:
                verdict = await chain.ainvoke(
                    {"question": question, "document": doc.page_content}
                )
                keep, why = verdict.keep, verdict.why
            except Exception as exc:  # noqa: BLE001
                # Fail-open: descartar un doc bueno cuesta la respuesta entera,
                # conservar uno malo solo cuesta un poco de calidad.
                log.warning("el grader falló en %s: %s", label, exc)
                keep, why = True, f"grader falló ({type(exc).__name__}), se conserva"
            span.update(output={"keep": keep, "why": why})
            return keep, {
                "source": doc.metadata.get("source"),
                "chunk": doc.metadata.get("chunk"),
                "keep": keep,
                "why": why,
            }

    results = await asyncio.gather(*(grade_one(i, d) for i, d in enumerate(docs)))
    kept = [doc for doc, (keep, _) in zip(docs, results) if keep]
    return {
        "kept": kept,
        "grade_debug": [dbg for _, dbg in results],
        "n_kept": len(kept),
        "n_dropped": len(docs) - len(kept),
    }


async def rewrite_query(state: RAGState) -> dict:
    chain = prompts.template("rag/rewrite-query") | settings.chat_llm(0.3) | StrOutputParser()
    rewrites = state.get("rewrites", 0) + 1
    try:
        new_query = (await chain.ainvoke({"question": state["question"]})).strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("falló la reescritura: %s", exc)
        new_query = state.get("query") or state["question"]
    return {"query": new_query or state["question"], "rewrites": rewrites}


async def generate(state: RAGState) -> dict:
    kept = state.get("kept", [])
    if not kept:
        # Sin contexto aprobado no se llama al LLM: garantiza cero alucinación y
        # ahorra la generación más cara del pipeline.
        return {"answer": NO_CONTEXT_ANSWER}

    context = "\n\n".join(
        f"[{i + 1}] ({doc.metadata.get('source')}) {doc.page_content}"
        for i, doc in enumerate(kept)
    )
    chain = prompts.template("rag/generate") | settings.chat_llm() | StrOutputParser()
    answer = await chain.ainvoke({"context": context, "question": state["question"]})
    return {"answer": answer.strip()}
