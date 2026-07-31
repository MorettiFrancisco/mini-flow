"""Toda la configuración por env vive acá. Nada de os.environ desparramado."""

import os

from dotenv import load_dotenv

# Antes de cualquier import de langfuse: el cliente singleton lee credenciales
# del env al construirse.
load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
# La dimensión va en el nombre: cambiar de modelo de embeddings crea una
# colección nueva en vez de corromper la vieja (nomic-embed-text es 768-d).
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "docs_nomic_768")

TOP_K = int(os.getenv("TOP_K", "5"))
MIN_RELEVANT = int(os.getenv("MIN_RELEVANT", "2"))
MAX_REWRITES = int(os.getenv("MAX_REWRITES", "2"))
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "15"))

# El default de contexto de Ollama es chico y trunca en silencio. Explícito.
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
# Recargar un modelo en CPU cuesta segundos; que se quede caliente entre requests.
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "3"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

ENVIRONMENT = os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "local")
PROMPT_LABEL = os.getenv("PROMPT_LABEL", "production")
PROMPT_CACHE_TTL = int(os.getenv("PROMPT_CACHE_TTL", "300"))

DATA_DIR = os.getenv("DATA_DIR", "data")


def chat_llm(temperature: float = 0.0):
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_ctx=NUM_CTX,
        keep_alive=KEEP_ALIVE,
    )


def embeddings():
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
