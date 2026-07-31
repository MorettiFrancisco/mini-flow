"""Prompts del pipeline.

`PROMPTS` cumple doble función:
  1. es la fuente que `scripts/bootstrap.py` siembra en Langfuse (`create_prompt`),
  2. es el `fallback=` de `get_prompt`, para que el pipeline siga andando si
     Langfuse no responde o las keys están mal.

Sintaxis mustache `{{var}}` porque es la de Langfuse; `get_langchain_prompt()`
la convierte a `{var}` para LangChain. Ojo: **nada de llaves literales** en los
cuerpos, LangChain las interpreta como variables. La forma de la salida
estructurada la impone `with_structured_output`, no el texto del prompt.
"""

from app import settings
from app.obs import ENABLED, client

GRADE_DOCUMENT = """Sos un evaluador de relevancia en un sistema de búsqueda.

Pregunta del usuario:
{{question}}

Documento recuperado:
{{document}}

¿Este documento contiene información que ayude a responder la pregunta?
Respondé keep=true solo si aporta algo concreto a la respuesta. Si apenas
comparte el tema pero no responde nada, respondé keep=false.
En "why" justificá en 15 palabras o menos."""

REWRITE_QUERY = """Reescribí esta consulta para que funcione mejor en una búsqueda
semántica sobre documentación técnica.

Consulta original:
{{question}}

La búsqueda anterior no trajo documentos relevantes. Reformulá usando términos
más explícitos y sinónimos que probablemente aparezcan en la documentación.
Devolvé únicamente la consulta reescrita, sin explicaciones ni comillas."""

GENERATE = """Respondé la pregunta del usuario usando exclusivamente el contexto
que sigue. No uses conocimiento previo.

Contexto:
{{context}}

Pregunta:
{{question}}

Si el contexto no alcanza para responder, decílo explícitamente en vez de
inventar. Sé conciso: 2 a 4 oraciones. Respondé en español."""

JUDGE = """Sos un evaluador de respuestas de un sistema RAG. Puntuá de 0 a 5.

Pregunta:
{{question}}

Contexto que se le dio al modelo:
{{context}}

Respuesta generada:
{{answer}}

Criterios:
- groundedness: cuánto de la respuesta está respaldado por el contexto. 5 = todo
  afirmado sale del contexto; 0 = inventado.
- completeness: cuánto de la pregunta queda respondida. 5 = completa; 0 = no responde.
- usefulness: qué tan útil le resulta al usuario. 5 = accionable y clara; 0 = inútil.

verdict: "pass" si las tres son 4 o 5, "fail" si alguna es 0 o 1, "partial" en el resto.
rationale: una oración, 25 palabras o menos."""

PROMPTS: dict[str, str] = {
    "rag/grade-document": GRADE_DOCUMENT,
    "rag/rewrite-query": REWRITE_QUERY,
    "rag/generate": GENERATE,
    "rag/judge": JUDGE,
}


def get(name: str):
    """Trae el prompt de Langfuse con el label configurado.

    `fallback=` no es opcional: es lo que hace que un Langfuse caído degrade la
    observabilidad en vez de tumbar el pipeline.
    """
    if not ENABLED:
        return _LocalPrompt(name)
    return client().get_prompt(
        name,
        label=settings.PROMPT_LABEL,
        cache_ttl_seconds=settings.PROMPT_CACHE_TTL,
        fallback=PROMPTS[name],
    )


def template(name: str):
    """ChatPromptTemplate ligado a la versión del prompt en Langfuse.

    El `metadata={"langfuse_prompt": p}` es lo que hace que el CallbackHandler
    linkee la generación con la versión del prompt en la UI.
    """
    from langchain_core.prompts import ChatPromptTemplate

    p = get(name)
    return ChatPromptTemplate.from_template(
        p.get_langchain_prompt(), metadata={"langfuse_prompt": p}
    )


class _LocalPrompt:
    """Stand-in para cuando el tracing está apagado: misma superficie mínima."""

    def __init__(self, name: str):
        self.name = name
        self.prompt = PROMPTS[name]

    def get_langchain_prompt(self) -> str:
        return self.prompt.replace("{{", "{").replace("}}", "}")

    def compile(self, **kwargs) -> str:
        out = self.prompt
        for key, value in kwargs.items():
            out = out.replace("{{" + key + "}}", str(value))
        return out
