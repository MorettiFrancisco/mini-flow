"""Chequeo del ruteo del ciclo de autocorrección. No necesita Ollama, Chroma ni
Langfuse levantados."""

from langchain_core.documents import Document

from app.graph import route_after_grade

LIMITS = {"min_relevant": 2, "max_rewrites": 2}


def _state(n_kept: int, rewrites: int) -> dict:
    return {"kept": [Document(page_content=f"d{i}") for i in range(n_kept)], "rewrites": rewrites}


def test_suficientes_docs_va_a_generate():
    assert route_after_grade(_state(2, 0), **LIMITS) == "generate"
    assert route_after_grade(_state(5, 1), **LIMITS) == "generate"


def test_pocos_docs_con_reintentos_disponibles_reescribe():
    assert route_after_grade(_state(1, 0), **LIMITS) == "rewrite_query"
    assert route_after_grade(_state(0, 1), **LIMITS) == "rewrite_query"


def test_reintentos_agotados_genera_igual():
    """Guarda contra el loop infinito: sin esto el ciclo retrieve->grade->rewrite
    no termina nunca cuando el corpus no cubre la pregunta."""
    assert route_after_grade(_state(0, 2), **LIMITS) == "generate"
    assert route_after_grade(_state(1, 3), **LIMITS) == "generate"


def test_estado_vacio_no_explota():
    assert route_after_grade({}, **LIMITS) == "rewrite_query"
