# observador-lab

## ¿Qué es esto?

`observador-lab` es un proyecto de práctica e investigación personal sobre
**interpretabilidad mecanicista**: la disciplina que trata de entender qué pasa
"adentro" de un modelo de lenguaje mientras procesa una frase, en vez de solo mirar
qué responde. El nombre describe la función del sistema — observar el mecanismo
interno de un modelo — y no es una marca ni un producto comercial.

La pregunta que este proyecto intenta explorar es la siguiente: cuando un modelo de
lenguaje explica en palabras por qué resolvió algo de cierta manera, ¿esa
explicación refleja de verdad lo que pasó dentro del modelo (sus pesos de atención,
sus activaciones), o es una narrativa que el modelo genera por separado, desconectada
de su propio mecanismo interno?

Para explorar esto se usa una tarea simple y verificable: la resolución de
**correferencia ambigua** (frases tipo "the trophy doesn't fit in the suitcase
because it is too big" — ¿a qué se refiere "it"?). Se compara, para una misma
frase, el mapa de atención real del modelo con la explicación en palabras que el
mismo modelo da sobre su propia respuesta.

Importante: esto **no** es una librería ni un producto para instalar y usar en
otro proyecto. Es un laboratorio de experimentos chicos, pensado para correr en
una máquina local, sin GPU, y para que cualquier persona interesada en
interpretabilidad pueda levantarlo y explorar sus propias frases.

**Los resultados de los primeros experimentos (Fases 1 y 2) ya están documentados
en [`RESULTADOS.md`](./RESULTADOS.md)**, incluyendo un hallazgo inesperado: la
atención promedio no resuelve esta tarea de forma confiable en ninguno de los tres
modelos, y el mecanismo real (cuando se pudo verificar causalmente, vía ablation)
resultó ser un sesgo posicional hacia el sujeto de la oración, no una resolución
semántica del significado. Recomendamos leer ese documento antes de explorar la
interfaz, para tener el contexto de qué tan lejos llega (y dónde no llega) lo que
la herramienta puede mostrar.

## Instalación paso a paso

Estas instrucciones asumen que no tenés experiencia previa con `transformer_lens`
ni con interpretabilidad mecanicista — solo que tenés Python instalado.

### 1. Requisitos previos

- Python 3.10 o superior instalado en tu máquina.
- Conexión a internet (solo se necesita la primera vez, para descargar los modelos).
- No hace falta GPU: todo corre en CPU.

### 2. Clonar el repositorio

```bash
git clone https://github.com/Natvoc/observador-lab.git
cd observador-lab
```

### 3. Crear un entorno virtual

Un entorno virtual mantiene las dependencias de este proyecto separadas del resto
de tu sistema.

En Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\activate
```

En Linux/Mac:

```bash
python -m venv venv
source venv/bin/activate
```

Vas a saber que el entorno está activado porque el prompt de tu terminal empieza
con `(venv)`.

### 4. Instalar las dependencias

```bash
pip install -r requirements.txt
```

Esto instala `torch` (en su versión de CPU), `transformer_lens`, `gradio`,
`plotly` y `circuitsvis`. La primera instalación puede tardar varios minutos.

### 5. Correr la aplicación

```bash
python app.py
```

La primera vez que elijas un modelo, se va a descargar automáticamente desde
HuggingFace (esto puede tardar unos minutos según tu conexión). Las veces
siguientes ya queda en caché local y carga rápido. Una vez que arranca, Gradio
te va a dar una dirección local (algo como `http://127.0.0.1:7860`) para abrir en
el navegador. No hace falta conexión a internet salvo para la descarga inicial de
cada modelo, y no se necesita ningún servicio externo ni hosting.

## Qué modelos usa, y por qué son chicos

El proyecto usa exclusivamente modelos **pequeños y abiertos**:

- **GPT-2 small** (~124M parámetros)
- **Pythia 70M**
- **Pythia 160M**

Esto es una **decisión deliberada, no una limitación de recursos**. Los modelos
grandes actuales (GPT-4, Claude, etc.) son en la práctica cajas negras: tienen
miles de millones de parámetros y no es viable inspeccionar sus mecanismos
internos de forma completa con las herramientas disponibles hoy, ni correrlos
localmente en una laptop. Los modelos chicos, en cambio, permiten observar
literalmente cada peso y cada activación con herramientas como `transformer_lens`,
en una computadora normal, sin GPU. La ventaja no es la potencia del modelo, sino
la **transparencia**: se puede mirar todo lo que pasa adentro.

Como contrapartida, estos modelos son bastante más limitados que los modelos
grandes que la mayoría de la gente usa a diario (ChatGPT, Claude, etc.). Es
esperable que a veces den respuestas incoherentes o fallen directamente en la
tarea — y eso, en este proyecto, también es un resultado válido y interesante de
mostrar, no un error del sistema.

## Limitaciones del sistema

- Solo funciona de forma confiable con **frases cortas en inglés**. Estos modelos
  fueron entrenados mayoritariamente en inglés; en español su comportamiento sería
  mucho más pobre o incoherente, así que los experimentos son en inglés.
- Solo tiene sentido interpretativo para el tipo de tarea para el que fue diseñado:
  **correferencia ambigua** (frases donde un pronombre puede referirse a más de un
  sustantivo). Se puede probar con otro tipo de frases (razonamiento, opinión,
  matemáticas) y la interfaz va a mostrar igual el mapa de atención, pero la
  comparación "auto-reporte vs. mecanismo" pierde su marco de referencia.
- Al ser modelos de 70 a 160 millones de parámetros, pueden **fallar o dar
  explicaciones incoherentes** ante frases largas o complejas. Eso no es un bug:
  es, en sí mismo, un resultado que vale la pena observar (el modelo ni siquiera
  resolvió la tarea).
- **El mapa de atención que muestra la interfaz es un indicio observacional, no
  una prueba de mecanismo.** Un patrón de atención que "coincide" con la respuesta
  correcta puede ser un artefacto (attention sink al primer token, cabeza de
  sintaxis/función) y no la causa real de esa respuesta. Solo se verificó
  causalmente, vía ablation, para tres oraciones de ejemplo específicas (ver
  `RESULTADOS.md`). Para cualquier otra frase que escribas, tratá el mapa de
  atención como un punto de partida para explorar, no como una prueba definitiva
  de qué "decidió" el modelo.

## Alcance de los resultados

Cualquier hallazgo de este proyecto se reporta **a la escala real de los modelos
usados (70–160 millones de parámetros)**. En ningún caso se generaliza ni se
extrapola a modelos de lenguaje grandes (GPT-4, Claude, u otros). Lo que se observe
acá es válido para estos modelos chicos, y punto — no es una afirmación sobre cómo
funcionan "los LLMs" en general.

## Qué no es este proyecto

- No es una librería ni un producto con nombre comercial.
- No hace fine-tuning ni entrena modelos propios: todos los modelos se usan tal
  cual fueron publicados.
- No acumula datos a gran escala: el foco está en pocos casos, bien entendidos y
  bien visualizados, no en miles de filas de resultados.
- No simula un "flujo de pensamiento" inventado: toda visualización corresponde a
  pesos y activaciones reales del modelo.
