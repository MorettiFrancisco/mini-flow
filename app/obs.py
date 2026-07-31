"""La única puerta al SDK de Langfuse.

Todo el resto de la app habla con Langfuse a través de acá. El SDK rompió
compatibilidad entre v3 y v4 (`update_current_trace` -> `propagate_attributes`,
`start_as_current_span` -> `start_as_current_observation`), así que el próximo
salto de versión toca este archivo y ninguno más.

Requiere SDK v4 (`langfuse>=4,<5`) contra un server Langfuse v4.
"""

import contextlib
import logging
import os

from app import settings

log = logging.getLogger(__name__)

# Kill switch propio: la app tiene que funcionar con el observador apagado o caído.
ENABLED = os.getenv("LANGFUSE_ENABLED", "true").strip().lower() not in (
    "false",
    "0",
    "no",
)


class _NullSpan:
    """Stub con la misma superficie que una observación, para cuando ENABLED=False."""

    def update(self, **_kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def client():
    from langfuse import get_client

    return get_client()


def _scalars(**kwargs) -> dict[str, str]:
    """Normaliza metadata para `propagate_attributes`.

    Restricciones del SDK v4: metadata es dict[str, str], las claves deben ser
    alfanuméricas y los valores de más de 200 caracteres se DESCARTAN en silencio.
    De ahí que las claves sean camelCase sin guiones bajos y que todo se recorte.
    """
    return {k: str(v)[:200] for k, v in kwargs.items() if v is not None}


@contextlib.contextmanager
def trace_scope(
    question: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    trace_name: str = "rag-query",
    root_name: str = "rag-pipeline",
    extra_tags: tuple[str, ...] = (),
):
    """Atributos de trace + observación raíz.

    Devuelve la observación raíz: usá `root.update(output=..., metadata=...)` al
    terminar para colgarle los agregados (iteraciones, cantidad de docs, etc.).
    """
    if not ENABLED:
        yield _NullSpan()
        return

    from langfuse import propagate_attributes

    lf = client()
    tags = ["rag", "ollama", settings.CHAT_MODEL, settings.ENVIRONMENT, *extra_tags]

    with propagate_attributes(
        trace_name=trace_name,
        user_id=user_id,
        session_id=session_id,
        tags=tags,
        metadata=_scalars(
            chatModel=settings.CHAT_MODEL,
            embedModel=settings.EMBED_MODEL,
            topK=settings.TOP_K,
            minRelevant=settings.MIN_RELEVANT,
            maxRewrites=settings.MAX_REWRITES,
        ),
    ):
        with lf.start_as_current_observation(
            name=root_name, as_type="span", input={"question": question}
        ) as root:
            yield root


def node_span(name: str, *, input=None, as_type: str = "span"):
    """Observación manual, para lo que no es un nodo del grafo (ej: fan-out del grader)."""
    if not ENABLED:
        return _NullSpan()
    return client().start_as_current_observation(name=name, as_type=as_type, input=input)


def callbacks() -> list:
    """Callbacks para pasarle a LangGraph. Una instancia nueva por request:
    el handler mantiene un mapa interno de runs."""
    if not ENABLED:
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def current_trace_id() -> str | None:
    if not ENABLED:
        return None
    try:
        return client().get_current_trace_id()
    except Exception:  # noqa: BLE001 - nunca romper el request por el tracing
        log.warning("no se pudo obtener el trace_id actual", exc_info=True)
        return None


def score(name: str, value, *, data_type: str = "NUMERIC", comment: str | None = None):
    """Score sobre el trace actual. Tiene que llamarse dentro de un `trace_scope`."""
    if not ENABLED:
        return
    try:
        client().score_current_trace(
            name=name,
            value=value,
            data_type=data_type,
            comment=comment[:500] if comment else None,
        )
    except Exception:  # noqa: BLE001
        log.warning("falló el score %r", name, exc_info=True)


def shutdown() -> None:
    """Flush + cierre. Obligatorio al terminar el proceso: el SDK batchea sobre
    OTel y si no, se pierde la cola de los traces."""
    if not ENABLED:
        return
    try:
        client().shutdown()
    except Exception:  # noqa: BLE001
        log.warning("falló el shutdown de langfuse", exc_info=True)
