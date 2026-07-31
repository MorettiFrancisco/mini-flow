"""Evaluación offline sobre un dataset de Langfuse.

Corre las preguntas del dataset por el pipeline completo y sube las tres
dimensiones del juez como evaluaciones del experimento. Es lo que permite
comparar dos versiones de prompt (o dos modelos) con números en vez de
impresiones: cambiá el label `production` de un prompt, volvé a correr esto, y
compará los dos runs en la UI.

    uv run scripts/eval.py
    uv run scripts/eval.py mi-nombre-de-run
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import graph, obs, settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("eval")

DATASET = "mini-flow-rag"

# Las tres primeras están cubiertas por data/; las dos últimas no, y son las que
# ejercitan el ciclo de reescritura y la respuesta "no lo encontré".
ITEMS = [
    {
        "input": "¿Cuál es la diferencia entre un span y una generation en Langfuse?",
        "expected_output": "Un span es un paso de trabajo cualquiera; una generation es específicamente una llamada a un LLM y además registra modelo, parámetros y tokens.",
    },
    {
        "input": "¿Por qué conviene setear num_ctx explícitamente en Ollama?",
        "expected_output": "Porque el contexto por defecto es chico y cuando el prompt lo excede se trunca en silencio, sin error, degradando la calidad.",
    },
    {
        "input": "¿Qué hago si el grader de documentos falla o no devuelve una estructura válida?",
        "expected_output": "Fail-open: conservar el documento, porque descartar uno bueno arruina la respuesta entera mientras que conservar uno malo solo cuesta algo de calidad.",
    },
    {
        "input": "¿Cómo se testea un router de LangGraph sin levantar el modelo?",
        "expected_output": "Definiéndolo como función pura afuera de la fábrica del grafo, y llamándolo con un diccionario de estado armado a mano.",
    },
    {
        "input": "¿Cuánto cuesta una licencia empresarial de Oracle Database?",
        "expected_output": "No está en el corpus: el sistema debe admitir que no encontró la información.",
    },
]


def seed_dataset() -> None:
    client = obs.client()
    try:
        client.create_dataset(name=DATASET, description="Preguntas de humo sobre el corpus de data/")
    except Exception:  # noqa: BLE001 - ya existe
        log.info("dataset: %s ya existe", DATASET)
    for i, item in enumerate(ITEMS):
        client.create_dataset_item(
            dataset_name=DATASET,
            # Id determinístico: re-correr esto actualiza los items, no los duplica.
            id=f"{DATASET}-{i:02d}",
            input=item["input"],
            expected_output=item["expected_output"],
        )
    log.info("dataset: %d items", len(ITEMS))


def _task(*, item, **_kwargs):
    result = asyncio.run(
        graph.run_query(item.input, user_id="eval", session_id=f"eval-{DATASET}")
    )
    return {"answer": result["answer"], "judgement": result["judgement"]}


def _judge_evaluators():
    """Una evaluación por dimensión del juez, más el flag de parseo."""
    from langfuse import Evaluation

    def make(dimension: str):
        def evaluator(*, output, **_kwargs):
            judgement = (output or {}).get("judgement") or {}
            if not judgement.get("parse_ok"):
                return Evaluation(name=dimension, value=0, comment="juez sin salida válida")
            return Evaluation(
                name=dimension,
                value=judgement[dimension] / 5,
                comment=judgement.get("rationale", ""),
            )

        evaluator.__name__ = f"eval_{dimension}"
        return evaluator

    def parse_ok(*, output, **_kwargs):
        judgement = (output or {}).get("judgement") or {}
        return Evaluation(name="judge_parse_ok", value=1 if judgement.get("parse_ok") else 0)

    return [make("groundedness"), make("completeness"), make("usefulness"), parse_ok]


def main() -> None:
    if not obs.ENABLED:
        raise SystemExit("este script necesita Langfuse habilitado (LANGFUSE_ENABLED=true)")

    run_name = sys.argv[1] if len(sys.argv) > 1 else f"{settings.CHAT_MODEL}-{settings.PROMPT_LABEL}"
    try:
        seed_dataset()
        dataset = obs.client().get_dataset(DATASET)
        log.info("corriendo experimento %r sobre %d items...", run_name, len(ITEMS))
        result = dataset.run_experiment(
            name=run_name, task=_task, evaluators=_judge_evaluators()
        )
        log.info("listo. mirá el dataset run %r en la UI de Langfuse", run_name)
        log.info("%s", result)
    finally:
        obs.shutdown()


if __name__ == "__main__":
    main()
