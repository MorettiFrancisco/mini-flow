"""Acceso a Chroma.

Siempre se pasa `embedding_function` explícito, en ingesta y en query. Si se deja
el default de Chroma (all-MiniLM-L6-v2, 384-d) contra una colección construida
con nomic-embed-text (768-d), el resultado es error o —peor— resultados basura
sin error.
"""

import functools

from app import settings


@functools.lru_cache(maxsize=1)
def store():
    """Vector store cacheado. El cliente HTTP de Chroma es reusable entre requests."""
    import chromadb
    from langchain_chroma import Chroma

    client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
    return Chroma(
        client=client,
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=settings.embeddings(),
    )


def count() -> int:
    return store()._collection.count()
