"""Estado del grafo.

Sin `Annotated`/reducers: cada nodo sobreescribe sus claves. `question` y `query`
están separadas a propósito — es lo que permite ver en el trace la pregunta
original al lado de la query reescrita.
"""

from typing import Any, TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    question: str  # pregunta original del usuario, nunca se modifica
    query: str  # query efectiva; rewrite_query la cambia

    docs: list[Document]  # lo que devolvió el retriever
    kept: list[Document]  # los que el grader aprobó

    answer: str
    rewrites: int

    # Payload de debug. Va en el estado devuelto a propósito: el CallbackHandler
    # serializa lo que devuelve el nodo como output de su span, así que esto
    # aparece en Langfuse sin una sola llamada extra al SDK.
    retrieval_debug: list[dict[str, Any]]
    grade_debug: list[dict[str, Any]]

    n_retrieved: int
    n_kept: int
    n_dropped: int

    judgement: dict[str, Any] | None
