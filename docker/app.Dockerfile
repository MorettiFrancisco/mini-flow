# Imagen compartida por los servicios `bootstrap` y `api`.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Prompts y respuestas en español: sin esto, mojibake al salir por consola.
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8

# Capa de dependencias separada de la de código: editar un nodo no re-resuelve
# el árbol de dependencias.
#
# `uv.lock*` con glob: si el lock está commiteado se usa (build reproducible), y
# si no, `uv sync` resuelve en el build y la imagen se arma igual. Los pines de
# pyproject.toml están todos acotados por major, así que un resolve fresco no
# puede saltar de versión mayor. Ver el README para fijarlo.
#
# Con dev deps (pytest son unos pocos MB): es la misma imagen la que corre los
# tests, así el comando del README funciona sin un segundo build.
COPY pyproject.toml uv.lock* ./
RUN uv sync

COPY . .

EXPOSE 8080
CMD ["uv", "run", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]
