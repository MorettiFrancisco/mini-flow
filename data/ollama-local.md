# Modelos locales con Ollama

## Servidor y modelos

Ollama expone una API HTTP en el puerto 11434 y guarda los modelos descargados en
un directorio propio. En Docker ese directorio es `/root/.ollama`; montarlo como
volumen es lo que evita volver a bajar varios gigabytes en cada recreación del
contenedor.

`ollama pull` sobre un modelo que ya está descargado igual consulta el registry,
así que falla sin red. Para que un arranque sea idempotente y funcione offline
conviene chequear la lista local antes de pullear.

## Contexto: num_ctx

El tamaño de contexto por defecto de Ollama es chico. Cuando el prompt lo excede,
el contenido se **trunca en silencio**: no hay error, solo respuestas
inexplicablemente malas. En un pipeline de RAG, donde el prompt lleva varios
chunks de documentación, setear `num_ctx` explícitamente es casi obligatorio. Un
valor de 8192 es razonable para empezar.

Cada slot de ejecución paralela reserva su propia caché KV, así que el contexto
grande multiplicado por muchos slots consume memoria rápido.

## keep_alive

Después de un período de inactividad Ollama descarga el modelo de memoria. En
inferencia por CPU, volver a cargarlo cuesta varios segundos que se le suman al
próximo request. Subir `keep_alive` mantiene el modelo caliente entre pedidos.

## Paralelismo

`OLLAMA_NUM_PARALLEL` controla cuántos requests atiende simultáneamente un mismo
modelo, y `OLLAMA_MAX_LOADED_MODELS` cuántos modelos distintos mantiene en
memoria a la vez. Correr un modelo de chat y uno de embeddings al mismo tiempo
requiere que ese segundo valor sea al menos dos.

En CPU el paralelismo tiene rendimientos decrecientes: los slots compiten por los
mismos núcleos. Sigue conviniendo para tapar latencia de arranque, pero no
multiplica el throughput.

## Salida estructurada

Ollama soporta decodificación restringida por un JSON Schema. Eso hace que la
salida sea válida contra el schema casi siempre; lo que no garantiza es que sea
sensata.

Con modelos chicos las reglas que funcionan son: schemas chatos y diminutos, sin
anidamiento ni campos opcionales; temperatura en cero; y un reintento. Además,
una estrategia de fallback explícita para cuando la validación falla igual.

## Embeddings

`nomic-embed-text` es un modelo de embeddings de 768 dimensiones que corre en el
mismo servidor Ollama, así que no agrega dependencias de Python al proyecto.

El detalle que rompe cosas en silencio es la dimensión: si una colección se
construye con un modelo de 768 dimensiones y después se consulta con otro de 384,
el resultado es un error o —peor— resultados sin sentido sin ningún error. Poner
el modelo o la dimensión en el nombre de la colección hace que cambiar de modelo
cree una colección nueva en lugar de corromper la existente.
