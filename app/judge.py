"""LLM-as-a-judge y emisión de scores.

El nodo solo juzga y deja el veredicto en el estado; los scores los emite
`emit_scores`, que llama el caller (api.py / main.py) desde adentro del
`trace_scope`. Es a propósito: `score_current_trace` depende del contexto de
OpenTelemetry, y el caller es el único lugar donde ese contexto está garantizado
—si algún nodo pasara a ser sync, LangGraph lo correría en un threadpool y los
scores se perderían en silencio.
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from app import obs, prompts, settings
from app.state import RAGState

log = logging.getLogger(__name__)


class Judgement(BaseModel):
    """Enteros 0-5, no floats 0-1: un modelo chico acierta mucho más con "4" que
    con "0.73". Se dividen por 5 antes de mandarlos a Langfuse."""

    groundedness: int = Field(ge=0, le=5)
    completeness: int = Field(ge=0, le=5)
    usefulness: int = Field(ge=0, le=5)
    verdict: Literal["pass", "partial", "fail"]
    rationale: str


async def judge(state: RAGState) -> dict:
    kept = state.get("kept", [])
    context = "\n\n".join(
        f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(kept)
    ) or "(sin contexto aprobado)"

    chain = prompts.template("rag/judge") | settings.chat_llm().with_structured_output(
        Judgement
    ).with_retry(stop_after_attempt=2)

    try:
        verdict = await chain.ainvoke(
            {
                "question": state["question"],
                "context": context,
                "answer": state.get("answer", ""),
            }
        )
        return {"judgement": {**verdict.model_dump(), "parse_ok": True}}
    except Exception as exc:  # noqa: BLE001
        # Fail visible, no silencioso: el ratio de fallo de salida estructurada en
        # un modelo chico es de lo más interesante que muestra este demo.
        log.warning("el judge no devolvió una estructura válida: %s", exc)
        return {"judgement": {"parse_ok": False, "error": type(exc).__name__}}


def emit_scores(judgement: dict | None) -> None:
    """Tiene que llamarse dentro de un `obs.trace_scope` abierto."""
    if not judgement:
        return

    if not judgement.get("parse_ok"):
        obs.score(
            "judge_parse_ok",
            0,
            data_type="BOOLEAN",
            comment=judgement.get("error", "salida estructurada inválida"),
        )
        return

    rationale = judgement.get("rationale", "")
    for dimension in ("groundedness", "completeness", "usefulness"):
        obs.score(
            dimension,
            judgement[dimension] / 5,
            data_type="NUMERIC",
            comment=rationale,
        )
    obs.score("verdict", judgement["verdict"], data_type="CATEGORICAL")
    obs.score("judge_parse_ok", 1, data_type="BOOLEAN")
