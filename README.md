# mini-flow

RAG autocorrectivo en LangGraph sobre un modelo local de Ollama, instrumentado de
punta a punta con **Langfuse**: traces, spans anidados, metadata, tags, prompts
versionados y scores de un LLM-as-a-judge.

El corpus de `data/` documenta justamente los conceptos que usa el pipeline
(Langfuse, LangGraph, RAG autocorrectivo, Ollama), así que se puede preguntar por
el propio sistema.

## El flujo

```
POST /ask ──► retrieve ──► grade_documents ──┬──► rewrite_query ──┐
              (Chroma)     (1 LLM por doc,   │   (hasta 2 veces)  │
                            en paralelo)     │                    └──► vuelve a retrieve
                                             └──► generate ──► judge ──► 5 scores
```

- **grade_documents** descarta los documentos irrelevantes, uno por uno, y deja
  registrado el porqué de cada descarte.
- Si quedaron menos de `MIN_RELEVANT` documentos, **rewrite_query** reformula la
  consulta y vuelve a recuperar, hasta `MAX_REWRITES` veces.
- **generate** responde usando exclusivamente el contexto aprobado. Con cero
  documentos aprobados no llama al modelo: devuelve "no lo encontré".
- **judge** puntúa groundedness, completitud y utilidad, y el resultado se sube
  como scores del trace.

## Requisitos previos

**Langfuse tiene que estar corriendo aparte** — este compose no lo incluye. Se
asume el stack oficial (`langfuse/langfuse`, imágenes `:4`) publicando `:3000` en
el host. Falta solo crear un proyecto y sacar las API keys.

Después:

```bash
cp .env.example .env
```

y completar `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`.

## Levantar todo

```bash
docker compose up -d
```

Eso hace, en orden: arranca Ollama, pullea los modelos (solo la primera vez),
arranca Chroma, siembra los 4 prompts en Langfuse, ingesta `data/` en Chroma y
deja la API en `:8080`.

El primer arranque baja ~2,3 GB de modelos. Seguirlo con:

```bash
docker compose logs -f bootstrap
```

## Preguntar

```bash
curl -s -X POST http://localhost:8080/ask -H "Content-Type: application/json" -d "{\"question\":\"que diferencia hay entre un span y una generation\",\"session_id\":\"s-1\",\"user_id\":\"dev\"}"
```

La respuesta incluye el `trace_id` para abrir el trace directo en Langfuse.

**Latencia esperada en CPU** (sin GPU, modelo de 3B): retrieve ~0,2 s, grading de
5 docs en paralelo ~4-8 s, generate ~10-20 s, judge ~5-10 s → **20-35 s por
pregunta**. No está colgado.

También hay un CLI para debuggear sin HTTP:

```bash
docker compose run --rm bootstrap uv run main.py "tu pregunta"
```

## Qué mirar en Langfuse

Un trace `rag-query` por request, con `user_id`, `session_id` y tags. Adentro:

| Qué | Dónde |
|---|---|
| Latencia por nodo | Calculada por el servidor, sin timers en el código |
| Documentos recuperados con score y source | Output del span `retrieve` |
| Qué documentos se descartaron y por qué | Output de `grade_documents` + un span por documento |
| Iteraciones del ciclo, cantidad de docs | Metadata del span raíz `rag-pipeline` |
| Versión de prompt usada en cada llamada | Generations linkeadas al prompt |
| groundedness / completeness / usefulness / verdict / judge_parse_ok | Scores del trace |

**La prueba interesante**: editá el prompt `rag/generate` en la UI de Langfuse,
movéle el label `production` a la versión nueva, y volvé a hacer el `curl` **sin
reiniciar nada**. La respuesta cambia (hasta 300 s de demora por el caché de
prompts, configurable con `PROMPT_CACHE_TTL`).

Y para ver varios traces concurrentes agrupados por sesión, tirar tres requests a
la vez con el mismo `session_id`.

## Evaluación offline

```bash
docker compose run --rm bootstrap uv run scripts/eval.py
```

Crea un dataset de 5 preguntas (3 cubiertas por el corpus, 2 que no) y corre un
experimento con las tres dimensiones del juez como evaluaciones. Sirve para
comparar dos versiones de prompt o dos modelos con números: cambiá el label
`production`, volvé a correrlo, y compará los dos runs en la UI.

## Tests

```bash
docker compose run --rm --no-deps bootstrap uv run pytest
```

`--no-deps` porque `docker compose run` arranca las dependencias del servicio por
defecto, y estos tests no necesitan ninguna.

Chequea el ruteo del ciclo de autocorrección (suficientes docs, pocos docs con
reintentos disponibles, reintentos agotados). No necesita Ollama, Chroma ni
Langfuse levantados.

## Configuración

Todo por env var, ver [.env.example](.env.example). Los que más mueven la aguja:

| Variable | Default | Para qué |
|---|---|---|
| `CHAT_MODEL` | `qwen2.5:3b` | Subir a `qwen2.5:7b-instruct` mejora el juez y el grader, a costa de 2-3× la latencia en CPU |
| `NUM_CTX` | `8192` | El default de Ollama es chico y **trunca el prompt en silencio** |
| `TOP_K` / `MIN_RELEVANT` | `5` / `2` | Cuántos docs se recuperan y cuántos relevantes hacen falta para no reescribir |
| `MAX_CONCURRENCY` | `3` | Tope de requests en vuelo; Ollama en CPU serializa |
| `LANGFUSE_ENABLED` | `true` | En `false` apaga la instrumentación y usa los prompts hardcodeados |

Agregar documentos es tirar `.md` o `.txt` en `data/` y re-ingestar:

```bash
docker compose run --rm -e INGEST_FORCE=1 bootstrap
```

## Estructura

```
app/
  settings.py     env + factories de ChatOllama/OllamaEmbeddings
  obs.py          única puerta al SDK de Langfuse
  prompts.py      cuerpos de prompt (fallback) + wrapper de get_prompt
  state.py        RAGState
  vectorstore.py  cliente de Chroma
  nodes.py        retrieve, grade_documents, rewrite_query, generate
  judge.py        Judgement + emisión de scores
  graph.py        StateGraph, routers puros y el runner instrumentado
  api.py          FastAPI
data/             corpus (bind mount read-only)
scripts/
  bootstrap.py    seed de prompts + ingesta (one-shot, idempotente)
  eval.py         dataset + experimento
docker/app.Dockerfile
main.py           CLI para debug
```

Toda la API de Langfuse está encapsulada en `app/obs.py`. El SDK rompió
compatibilidad entre v3 y v4 (`update_current_trace` → `propagate_attributes`),
así que el próximo salto de versión toca un solo archivo.

## Notas para Windows

- **Puerto 11434**: si tenés el "Ollama for Windows" nativo en la bandeja, ya lo
  tiene tomado y `docker compose up` falla con un bind error. Salidas: cerrar la
  app de la bandeja, o borrar los servicios `ollama`/`ollama-pull` del compose y
  poner `OLLAMA_BASE_URL=http://host.docker.internal:11434`.
- **Memoria de WSL2**: con el stack de Langfuse (ClickHouse incluido) corriendo,
  16 GB queda ajustado. En `%USERPROFILE%\.wslconfig`:

  ```ini
  [wsl2]
  memory=12GB
  processors=6
  ```

  y después `wsl --shutdown`.
- **Acentos en la consola**: `chcp 65001` antes de correr el CLI.

## Fijar las versiones (recomendado)

El repo no trae `uv.lock` commiteado, así que el build resuelve las dependencias
en el momento. Los pines de `pyproject.toml` están acotados por major
(`langfuse>=4,<5`, `langgraph>=1,<2`), pero para que dos builds den exactamente
lo mismo conviene generar el lock una vez y commitearlo:

```bash
docker run --rm -v "${PWD}:/w" -w /w ghcr.io/astral-sh/uv:latest uv lock
```

No hace falta tener `uv` instalado, y `${PWD}` funciona igual en bash y en
PowerShell. Con el lock presente el `COPY uv.lock*` del
Dockerfile lo levanta solo.

## Reset

```bash
docker compose down -v
```

Borra los volúmenes de Ollama y Chroma (se vuelven a bajar los modelos). No toca
nada de tu stack de Langfuse.
