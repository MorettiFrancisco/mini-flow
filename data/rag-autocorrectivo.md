# RAG autocorrectivo

El RAG clásico recupera documentos y genera la respuesta. Si la recuperación
trajo basura, el modelo genera basura con tono seguro. El RAG autocorrectivo
agrega pasos de control entre la recuperación y la generación.

## Retrieve

Se embebe la consulta y se buscan los `k` chunks más cercanos en la base
vectorial. Un `k` típico es entre 4 y 8. La búsqueda devuelve, además del texto,
un score de similitud que conviene registrar: es la primera pista cuando las
respuestas salen mal.

## Grading de documentos

Un LLM evalúa cada documento recuperado contra la pregunta y decide si es
relevante. Los irrelevantes se descartan antes de armar el contexto. La decisión
es binaria y conviene pedirle también una justificación corta, para poder
auditar después por qué se descartó algo.

El grading cuesta una llamada al modelo por documento. Con `k=5` son cinco
inferencias extra. Ejecutarlas en paralelo esconde buena parte de esa latencia.

Una decisión importante es qué hacer cuando el grader falla o no devuelve una
estructura válida. Conviene **fail-open**: conservar el documento. Descartar un
documento bueno arruina la respuesta entera, mientras que conservar uno malo solo
le cuesta un poco de calidad.

## Rewrite query

Si después del grading no quedaron documentos relevantes —o quedaron menos que un
mínimo—, en vez de responder mal se reescribe la consulta y se vuelve a
recuperar. La reescritura busca términos más explícitos y sinónimos que
probablemente aparezcan en la documentación.

Este ciclo necesita un tope duro de reintentos. Sin un contador de reescrituras
en el estado, una pregunta que el corpus simplemente no cubre hace girar el ciclo
hasta el límite de recursión.

## Generate

La respuesta se genera usando exclusivamente el contexto aprobado. El prompt debe
prohibir explícitamente el conocimiento previo del modelo y pedirle que admita
cuando el contexto no alcanza.

El caso de cero documentos aprobados merece atención: si se lo delega al modelo,
a veces responde igual de memoria. Cortocircuitarlo en código y devolver una
respuesta fija de "no encontré esto en los documentos" garantiza cero
alucinación y además ahorra la generación más cara del pipeline.

## LLM-as-a-judge

Un segundo modelo evalúa la respuesta producida. Tres dimensiones habituales:

- **groundedness**: cuánto de lo afirmado está respaldado por el contexto. Es la
  medida directa de alucinación.
- **completitud**: cuánto de la pregunta quedó efectivamente respondido.
- **utilidad**: qué tan accionable y clara resulta la respuesta.

El juez tiene que devolver una estructura parseable. Con modelos chicos conviene
pedir enteros en una escala corta —0 a 5— en lugar de decimales entre 0 y 1: un
modelo de tres mil millones de parámetros acierta mucho más con "4" que con
"0.73". La conversión a la escala final se hace en código.

Cuando el juez no devuelve una estructura válida, registrar el fallo como un
score booleano es mejor que saltearlo en silencio: el ratio de fallos queda
visible y medible.
