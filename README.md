# mini-flow

Agente de consola en LangGraph con un flujo simple **research → summarize →
critique** (con loop de revisión hasta aprobar o llegar a 3 iteraciones),
corriendo contra un modelo local de Ollama y trackeado con Langfuse.

## Setup

1. Levantar Ollama y pullear el modelo (`llama3.2`):

   ```
   docker compose up -d
   ```

   Esperá a que el servicio `ollama-pull` termine (mirá los logs con
   `docker compose logs -f ollama-pull`).

2. Copiar `.env.example` a `.env` y completar las keys de Langfuse (apuntando
   al contenedor de Langfuse que ya tengas corriendo local):

   ```
   cp .env.example .env
   ```

3. Instalar dependencias (con [uv](https://docs.astral.sh/uv/)):

   ```
   uv sync
   ```

## Uso

```
uv run main.py "tu tema acá"
```

o sin argumento, te pide el topic por consola. Al final imprime el resumen y
cuántas iteraciones de crítica tardó en aprobarse. Cada corrida queda
registrada como un trace en Langfuse.

## Test

```
uv run test_graph.py
```

Chequea la lógica de ruteo del loop (aprobar vs. reintentar) sin necesitar
Ollama ni Langfuse levantados.