"""Preparación del entorno antes del primer request. One-shot e idempotente.

Dos trabajos:
  1. sembrar los prompts en Langfuse (si no existen ya),
  2. ingestar data/ en Chroma (si la colección está vacía).

    uv run scripts/bootstrap.py
    INGEST_FORCE=1 uv run scripts/bootstrap.py    # re-ingesta forzada
"""

import hashlib
import logging
import os
import sys
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import obs, prompts, settings, vectorstore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("bootstrap")

EXTENSIONS = ("*.md", "*.txt")


def wait_for_chroma(timeout: int = 90) -> None:
    """Espera a que Chroma acepte conexiones.

    El compose usa `service_started` para chroma en vez de `service_healthy`: la
    imagen no garantiza traer curl ni wget, y un healthcheck que apunta a un
    binario ausente cuelga toda la cadena de depends_on. Reintentar desde acá es
    inmune a lo que haya adentro de esa imagen.
    """
    deadline = time.monotonic() + timeout
    attempt = 0
    while True:
        attempt += 1
        try:
            vectorstore.count()
            log.info("chroma: disponible (intento %d)", attempt)
            return
        except Exception as exc:  # noqa: BLE001
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"chroma no respondió en {timeout}s "
                    f"({settings.CHROMA_HOST}:{settings.CHROMA_PORT}): {exc}"
                ) from exc
            time.sleep(2)


def seed_prompts() -> None:
    """Siembra los 4 prompts con label `production`.

    El guard es obligatorio: `create_prompt` **siempre** crea una versión nueva,
    así que sin él cada `docker compose up` infla el contador de versiones.
    """
    if not obs.ENABLED:
        log.info("prompts: tracing apagado, se saltea el seed")
        return

    client = obs.client()
    for name, body in prompts.PROMPTS.items():
        try:
            client.get_prompt(name, label=settings.PROMPT_LABEL, cache_ttl_seconds=0)
            log.info("prompts: %s ya existe, se deja como está", name)
            continue
        except Exception:  # noqa: BLE001 - la clase exacta de "not found" varía según versión del SDK
            pass
        try:
            client.create_prompt(
                name=name,
                type="text",
                prompt=body,
                labels=[settings.PROMPT_LABEL],
                config={"model": settings.CHAT_MODEL, "temperature": 0},
            )
            log.info("prompts: %s sembrado (v1, label=%s)", name, settings.PROMPT_LABEL)
        except Exception:  # noqa: BLE001
            log.warning("prompts: no se pudo sembrar %s", name, exc_info=True)


def load_documents() -> list[Document]:
    data_dir = Path(settings.DATA_DIR)
    if not data_dir.is_dir():
        log.warning("ingesta: no existe el directorio %s", data_dir.resolve())
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )
    documents: list[Document] = []
    files = sorted(p for pattern in EXTENSIONS for p in data_dir.rglob(pattern))
    for path in files:
        text = path.read_text(encoding="utf-8")
        source = path.relative_to(data_dir).as_posix()
        for i, chunk in enumerate(splitter.split_text(text)):
            documents.append(
                Document(page_content=chunk, metadata={"source": source, "chunk": i})
            )
        log.info("ingesta: %s", source)
    return documents


def ingest() -> None:
    force = os.getenv("INGEST_FORCE", "").strip().lower() in ("1", "true", "yes")
    existing = vectorstore.count()
    if existing and not force:
        log.info(
            "ingesta: la colección %s ya tiene %d chunks, se saltea (INGEST_FORCE=1 para forzar)",
            settings.CHROMA_COLLECTION,
            existing,
        )
        return

    documents = load_documents()
    if not documents:
        log.warning("ingesta: no hay documentos en %s", settings.DATA_DIR)
        return

    # Ids determinísticos: re-ingestar hace upsert en vez de duplicar.
    ids = [
        hashlib.sha256(
            f"{d.metadata['source']}:{d.metadata['chunk']}".encode()
        ).hexdigest()
        for d in documents
    ]
    vectorstore.store().add_documents(documents, ids=ids)
    log.info(
        "ingesta: %d chunks en la colección %s (total %d)",
        len(documents),
        settings.CHROMA_COLLECTION,
        vectorstore.count(),
    )


def main() -> None:
    log.info(
        "bootstrap | chat=%s embed=%s chroma=%s:%s langfuse=%s",
        settings.CHAT_MODEL,
        settings.EMBED_MODEL,
        settings.CHROMA_HOST,
        settings.CHROMA_PORT,
        "on" if obs.ENABLED else "off",
    )
    try:
        seed_prompts()
        wait_for_chroma()
        ingest()
    finally:
        obs.shutdown()


if __name__ == "__main__":
    main()
