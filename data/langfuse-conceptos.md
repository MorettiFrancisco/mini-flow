# Conceptos de Langfuse

## Trace

Un **trace** representa una ejecución completa de la aplicación: un request, una
corrida del agente, una pregunta del usuario. Es el contenedor de todo lo demás.
Un trace tiene nombre, input, output, y opcionalmente `user_id`, `session_id`,
tags y metadata. La metadata sirve para filtrar y agrupar traces en la UI.

## Observations: spans y generations

Dentro de un trace viven las **observations**, anidadas en árbol. Hay dos tipos
que importan:

- **span**: un paso de trabajo cualquiera. Recuperar documentos, validar una
  entrada, llamar a una API.
- **generation**: específicamente una llamada a un modelo de lenguaje. Además de
  input y output registra el modelo usado, los parámetros y el consumo de tokens.

La latencia de cada observation la calcula el servidor a partir de los timestamps
de inicio y fin. No hace falta cronometrar nada a mano en la aplicación.

## Sessions y users

Varios traces se agrupan en una **session** compartiendo el mismo `session_id`.
Es la forma de ver una conversación completa en vez de mensajes aislados. El
`user_id` permite filtrar por usuario y ver costo y volumen por persona.

## Scores

Un **score** es una evaluación adosada a un trace, a una observation o a una
sesión. Los tipos de dato son cuatro: `NUMERIC`, `CATEGORICAL`, `BOOLEAN` y
`TEXT`. Los scores pueden venir de tres lugares: anotación manual en la UI,
feedback del usuario final, o un evaluador automático como un LLM-as-a-judge.

Un score numérico admite un comentario, que es donde suele ir la justificación
que devolvió el juez.

## Prompt management

Langfuse guarda los prompts fuera del código, versionados. Cada vez que se
guarda un prompt con el mismo nombre se crea una versión nueva. Los **labels**
—típicamente `production`— apuntan a la versión activa.

La aplicación pide el prompt por nombre y label en tiempo de ejecución, con un
TTL de caché para no pegarle a la API en cada llamada. Mover el label
`production` a otra versión cambia el comportamiento de la aplicación sin
redeploy.

Pedir un prompt siempre con un valor de fallback es lo que evita que una caída
de Langfuse tumbe la aplicación: si la API no responde, se usa el prompt
hardcodeado y el pipeline sigue.

Cuando una generation se asocia a una versión de prompt, Langfuse puede comparar
métricas de calidad y latencia entre versiones.

## Datasets y experimentos

Un **dataset** es un conjunto de items entrada/salida esperada. Correr un
experimento sobre un dataset ejecuta la aplicación contra cada item y agrupa los
traces y scores resultantes bajo un dataset run. Es la forma de comparar dos
versiones de prompt o dos modelos con números en vez de impresiones.
