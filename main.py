"""CLI fina para debuggear el grafo sin levantar HTTP.

    uv run main.py "tu pregunta"

La API de larga vida está en app/api.py; esto usa exactamente el mismo runner
instrumentado (`graph.run_query`).
"""

import asyncio
import logging
import sys

from app import graph, obs


async def _run(question: str) -> None:
    result = await graph.run_query(question, user_id="cli", session_id="cli")

    print(f"\n--- Respuesta ---\n{result['answer']}")

    if result["graded"]:
        print("\n--- Documentos ---")
        for doc in result["graded"]:
            mark = "OK  " if doc["keep"] else "DROP"
            print(f"  [{mark}] {doc['source']}#{doc['chunk']}: {doc['why']}")

    judgement = result["judgement"] or {}
    if judgement.get("parse_ok"):
        print(
            f"\n--- Juez --- {judgement['verdict']} | "
            f"groundedness={judgement['groundedness']}/5 "
            f"completeness={judgement['completeness']}/5 "
            f"usefulness={judgement['usefulness']}/5"
        )
        print(f"  {judgement['rationale']}")
    elif judgement:
        print(f"\n--- Juez --- salida estructurada inválida ({judgement.get('error')})")

    print(f"\nreescrituras: {result['rewrites']} | trace: {result['trace_id']}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    question = " ".join(sys.argv[1:]) or input("Pregunta: ")
    try:
        asyncio.run(_run(question))
    finally:
        # Sin esto se pierde la cola del trace: el SDK batchea sobre OTel.
        obs.shutdown()


if __name__ == "__main__":
    main()
