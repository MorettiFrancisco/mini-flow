# LangGraph: lo básico

## StateGraph

LangGraph modela un agente como un grafo dirigido con estado. Se declara un
`StateGraph` sobre un tipo de estado —normalmente un `TypedDict`—, se agregan
nodos y se conectan con aristas. `compile()` devuelve algo invocable con
`invoke` o `ainvoke`.

## Estado y reducers

Cada nodo recibe el estado completo y devuelve un diccionario parcial con las
claves que modificó. Por defecto cada clave devuelta **sobreescribe** la
anterior. Si se necesita acumular en vez de reemplazar, se anota la clave con
`Annotated` y una función reductora; el caso típico es una lista de mensajes que
va creciendo.

Devolver un diccionario parcial en vez del estado completo no es solo prolijidad:
lo devuelto es lo que las integraciones de observabilidad registran como output
del nodo. Poner ahí el payload de diagnóstico es la forma más económica de que
aparezca en el trace.

## Aristas condicionales

`add_edge` conecta dos nodos siempre. `add_conditional_edges` recibe una función
router que mira el estado y devuelve el nombre del próximo nodo, o `END`. Es lo
que permite ramificar y ciclar.

Conviene que el router sea una **función pura** definida afuera de la fábrica del
grafo: así se testea con un diccionario armado a mano, sin levantar el modelo ni
las dependencias.

## Ciclos y recursion limit

Una arista que vuelve a un nodo anterior crea un ciclo. LangGraph corta a los 25
pasos por defecto, configurable con `recursion_limit`, y lanza
`GraphRecursionError`. Confiar solo en ese límite es mala idea: conviene además
llevar un contador de iteraciones en el estado y que el router salga del ciclo al
alcanzar un máximo explícito.

## Sync vs async

Los nodos pueden ser funciones sync o async. La diferencia importa para la
observabilidad: un nodo async corre en el mismo event loop y hereda el contexto
de la ejecución, mientras que un nodo sync se ejecuta en un pool de threads. Si
el contexto de trazado vive en `contextvars` —como pasa con OpenTelemetry—, ese
salto de thread puede dejar spans huérfanos.

Por eso, cuando hay paralelismo dentro de un nodo, `asyncio.gather` es preferible
a un `ThreadPoolExecutor` armado a mano: las tasks de asyncio copian el contexto,
los threads pelados no.
