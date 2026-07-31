"""Armado del grafo, los routers y el runner instrumentado.

Los routers son funciones puras fuera de `build_graph()` — así se testean sin
Ollama, sin Chroma y sin Langfuse (ver tests/test_routing.py).
"""

import functools
import logging

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from app import judge as judge_mod
from app import nodes, obs, settings
from app.state import RAGState

log = logging.getLogger(__name__)


def route_after_grade(
    state: RAGState,
    *,
    min_relevant: int | None = None,
    max_rewrites: int | None = None,
) -> str:
    """Decide entre reescribir la consulta y generar la respuesta.

    Los límites son parámetros con default para que los tests sean deterministas
    sin tocar variables de entorno; LangGraph llama la función con solo el estado.
    """
    min_relevant = settings.MIN_RELEVANT if min_relevant is None else min_relevant
    max_rewrites = settings.MAX_REWRITES if max_rewrites is None else max_rewrites

    if len(state.get("kept", [])) >= min_relevant:
        return "generate"
    # Guarda contra el loop infinito: agotados los reintentos se genera igual, y
    # `generate` sabe responder "no lo encontré" con cero documentos.
    if state.get("rewrites", 0) >= max_rewrites:
        return "generate"
    return "rewrite_query"


def initial_state(question: str) -> RAGState:
    return {
        "question": question,
        "query": question,
        "docs": [],
        "kept": [],
        "answer": "",
        "rewrites": 0,
        "retrieval_debug": [],
        "grade_debug": [],
        "n_retrieved": 0,
        "n_kept": 0,
        "n_dropped": 0,
        "judgement": None,
    }


@functools.lru_cache(maxsize=1)
def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade_documents", nodes.grade_documents)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("generate", nodes.generate)
    graph.add_node("judge", judge_mod.judge)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents", route_after_grade, ["rewrite_query", "generate"]
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "judge")
    graph.add_edge("judge", END)

    return graph.compile()


async def run_query(
    question: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
) -> dict:
    """Corre el pipeline dentro de un trace de Langfuse.

    Único lugar donde se abre el trace, para que api.py y main.py compartan
    exactamente la misma instrumentación.
    """
    with obs.trace_scope(question, user_id=user_id, session_id=session_id) as root:
        config = {
            "callbacks": obs.callbacks(),
            "recursion_limit": settings.RECURSION_LIMIT,
        }
        try:
            result = await build_graph().ainvoke(initial_state(question), config=config)
        except GraphRecursionError:
            # Que el trace quede cerrado y visible igual: un run que se pasó del
            # límite es justo lo que uno quiere poder encontrar en la UI.
            log.exception("se alcanzó el recursion_limit")
            root.update(output={"error": "recursion_limit"})
            raise

        # Los scores se emiten acá, no dentro del nodo: es el único punto donde el
        # contexto de OTel está garantizado (ver docstring de app/judge.py).
        judge_mod.emit_scores(result.get("judgement"))

        root.update(
            output={
                "answer": result.get("answer"),
                "judgement": result.get("judgement"),
            },
            metadata={
                "rewrites": result.get("rewrites", 0),
                "n_retrieved": result.get("n_retrieved", 0),
                "n_kept": result.get("n_kept", 0),
                "n_dropped": result.get("n_dropped", 0),
                "query_final": result.get("query"),
            },
        )
        return {
            "answer": result.get("answer", ""),
            "judgement": result.get("judgement"),
            "rewrites": result.get("rewrites", 0),
            "retrieved": result.get("retrieval_debug", []),
            "graded": result.get("grade_debug", []),
            "trace_id": obs.current_trace_id(),
        }
