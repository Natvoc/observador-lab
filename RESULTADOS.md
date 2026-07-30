# Resultados — observador-lab

Este documento consolida los experimentos de las Fases 1, 2 y 6: interpretabilidad
mecanicista sobre GPT-2 small, Pythia 70M y Pythia 160M, aplicada a una tarea de
correferencia ambigua tipo Winograd. Todo lo que sigue vale **únicamente para estos
modelos, en este rango de 70 a 160 millones de parámetros**. No se generaliza a
modelos de lenguaje grandes (GPT-4, Claude u otros), ni se afirma nada sobre "los
LLMs" en general.

*(La Fase 6 se adelantó antes que la Fase 5 porque extiende directamente el
hallazgo de ablation de la Fase 1, en vez de abrir una línea nueva — ver
`TAREAS.md`.)*

## 1. La tarea semilla

Se usó la resolución de correferencia ambigua tipo Winograd: oraciones donde un
pronombre ("it") puede referirse a dos sustantivos distintos, y una sola palabra
(un adjetivo) determina cuál es el correcto. El ejemplo canónico:

> "The trophy doesn't fit in the suitcase because it is too **big**." → "it" = trophy
> "The trophy doesn't fit in the suitcase because it is too **small**." → "it" = suitcase

Se eligió esta tarea por tres razones: tiene una respuesta objetivamente
verificable, está bien documentada en la literatura de interpretabilidad
mecanicista, y el patrón de atención esperado es relativamente fácil de
interpretar a simple vista.

## 2. Atención promediada: ningún modelo resuelve la tarea de forma confiable

Se armó una muestra de 8 oraciones tipo Winograd (no citas textuales del dataset
oficial; oraciones propias del mismo estilo), cada una con dos variantes, corridas
en los 3 modelos: 48 casos en total.

Midiendo la atención promedio (todas las capas y cabezas) desde la palabra que
desambigua hacia los dos sustantivos candidatos:

| Métrica | Aciertos | % |
|---|---|---|
| Atención (coincide con el sustantivo esperado) | 23/48 | 48% |
| Comportamiento (logit de continuación coincide) | 25/48 | 52% |

Ambos números son estadísticamente indistinguibles del azar (50%) con esta muestra.
Pero el hallazgo más informativo no es el porcentaje agregado, sino esto: **23 de
24 pares de oración-modelo (atención) y 21 de 24 (comportamiento) dan la misma
respuesta en las dos variantes**, sin importar cuál sea la palabra que desambigua.
Es decir, en la gran mayoría de los casos el modelo no está reaccionando al
contexto que cambia el significado — tiene una preferencia fija por oración, y esa
preferencia coincide con la respuesta correcta en algunas variantes y no en otras,
lo cual infla el promedio a algo cercano al 50% sin que haya resolución real de la
correferencia.

*(Nota metodológica: la atención se mide desde la palabra que desambigua, no desde
"it". Estos modelos son causales/autoregresivos: en la posición de "it" el modelo
todavía no vio la palabra que aparece después en la oración, así que su atención
ahí no puede depender de ella.)*

## 3. El mecanismo causal identificado en Pythia 70M (capa 0, cabeza 2)

Buscando cabeza por cabeza (no el promedio) sobre las 8 oraciones, aparecieron 4
cabezas candidatas que "resolvían" 3 o más de las 8 oraciones en al menos una
variante consistente. Inspeccionando a mano el patrón de atención completo (no
solo el par de candidatos, sino dónde iba *todo* el peso), 3 de esas 4 resultaron
ser artefactos:

- **GPT-2 small, capa 8/cabeza 3**: dominada por un "attention sink" al primer
  token (65–91% del peso).
- **Pythia 160M, capa 1/cabeza 6**: dominada por palabras funcionales/determinantes
  (80–98% del peso); el acierto entre candidatos era por márgenes de centésimas.
- **Pythia 70M, capa 3/cabeza 5**: 99.6–99.9% de la atención va siempre al primer
  token de la oración, sin importar el contenido — el "acierto" era ruido de punto
  flotante sobre un residuo insignificante.

Solo **Pythia 70M, capa 0/cabeza 2** mostró un patrón genuinamente interpretable:
en varias oraciones, los dos sustantivos candidatos eran literalmente los tokens
con más atención de toda la oración, por encima de cualquier palabra funcional, y
en el orden correcto.

Atención correcta no prueba causalidad. Se hizo zero-ablation (anular la salida de
esa cabeza en todas las posiciones, vía hook de `transformer_lens` sobre
`hook_z`) y se comparó el margen de logit (log-prob del sustantivo correcto menos
el incorrecto) antes y después, en 3 oraciones (trophy_suitcase, car_truck,
man_couch, 6 casos):

| Caso | Sujeto de la oración | Esperado | Margen sin ablation | Margen con ablation | Cambio |
|---|---|---|---|---|---|
| trophy_suitcase/big | trophy | trophy (=sujeto) | +2.330 | -3.496 | -5.826 |
| trophy_suitcase/small | trophy | suitcase (≠sujeto) | -1.126 | +5.188 | +6.314 |
| car_truck/slow | car | car (=sujeto) | -0.128 | -1.501 | -1.373 |
| car_truck/wide | car | truck (≠sujeto) | +0.732 | +2.140 | +1.407 |
| man_couch/weak | man | man (=sujeto) | +0.911 | +0.599 | -0.312 |
| man_couch/heavy | man | couch (≠sujeto) | -0.106 | +0.192 | +0.298 |

**Los 6 de 6 casos son consistentes con una sola regla**: apagar esta cabeza
siempre reduce el apoyo al sustantivo mencionado *primero* en la oración (el
sujeto), sin importar el adjetivo. Cuando el sujeto es la respuesta correcta, la
ablation empeora o invierte el acierto; cuando el sujeto es la respuesta
incorrecta, la ablation lo mejora o lo corrige. Esto es un mecanismo causal real,
localizado, y consistente — pero implementa un **sesgo posicional hacia el
sujeto**, no una resolución semántica del adjetivo.

## 4. ¿Un mecanismo comparable en GPT-2 small y Pythia 160M?

Se repitió el mismo método (búsqueda por cabeza sobre las mismas 3 oraciones,
inspección del patrón, zero-ablation de la mejor candidata) en los otros dos
modelos.

- **GPT-2 small (capa 0, cabeza 2)**: replica el mismo patrón exacto que Pythia
  70M — los 6/6 casos son consistentes con sesgo hacia el sujeto. La magnitud del
  efecto es más chica (cambios de 0.17 a 0.92 nats, contra hasta 6.3 nats en
  Pythia 70M), pero la firma causal es idéntica.
- **Pythia 160M (capa 0, cabeza 1)**: acá el resultado es ambiguo. Hay efecto
  causal real (los márgenes cambian de forma no trivial), pero la dirección es
  **inconsistente**: 3/6 casos coinciden con el sesgo hacia el sujeto, 2/6 van en
  la dirección opuesta (car_truck, ambas variantes), y 1/6 es despreciable. No se
  identifica esta cabeza como portadora de un mecanismo limpio. Esta cabeza empató
  en cantidad de oraciones resueltas con otras tres (capa 0, cabezas 2, 7 y 9) que
  no se llegaron a inspeccionar — no se descarta que alguna de esas muestre un
  patrón más consistente, pero queda fuera del alcance de esta ronda.

## 5. Los 18 auto-reportes (6 casos × 3 modelos)

Para cada modelo, se generó el auto-reporte (sin fine-tuning, solo prompting:
"...It refers to the [elección del modelo] because...") sobre los mismos 6 casos,
usando la elección real del modelo (no la respuesta forzada como correcta).

| Modelo | Patrones observados | Ejemplo |
|---|---|---|
| Pythia 70M | Circular / autocontradictorio | "The trophy is not a trophy because it is too big." |
| Pythia 70M | Diálogo fabricado no relacionado | "'I'm not going to get a ride on the road,' he said." |
| GPT-2 small | Glitch de tokenización | "It refers to the trophy **becauserophy** because..." |
| GPT-2 small | Eco literal del input | Repite la oración original completa, palabra por palabra |
| GPT-2 small | Narrativa fabricada, fluida pero irrelevante | "The car was traveling at a speed of about 100 mph..." |
| Pythia 160M | Mayormente fluida, ocasionalmente coherente | Ver caso único abajo |
| Pythia 160M | Error semántico pese a fluidez | "It refers to the man because he was too heavy" (heavy describía al sofá, no al hombre) |

**Hallazgo central: en ninguno de los 18 auto-reportes (ni en las versiones
generadas con la cabeza causal apagada) el modelo insinúa el mecanismo posicional
real.** Ni una sola vez menciona algo parecido a "porque se mencionó primero" o
expresa incertidumbre sobre su propia respuesta. Tampoco ajusta el contenido al
adjetivo real: en varios casos (trophy_suitcase, man_couch en Pythia 70M) el texto
generado es **prácticamente idéntico** entre las dos variantes, pese a que el
adjetivo cambió. En GPT-2 small y Pythia 160M, además, el modelo a veces **se
equivoca sobre su propia premisa** al restablecerla (dice "too small" cuando la
oración real decía "big", "too high" cuando decía "wide").

Extra — con la cabeza de Pythia 70M apagada: cuando la ablation cambia el output,
el auto-reporte cambia de superficie (menciona el nuevo sustantivo elegido) pero
no se vuelve sistemáticamente más ni menos coherente.

## 6. Gradiente de coherencia lingüística

Ordenando los 18 auto-reportes por qué tan "roto" suena el texto, aparece un
gradiente progresivo, no un salto categórico:

**Pythia 70M** (más roto: circularidad, autocontradicción) **< GPT-2 small**
(menos circular, pero con glitches de tokenización, ecos literales y tangentes
fabricadas) **< Pythia 160M** (más fluido gramaticalmente).

El único de los 18 auto-reportes que se lee como una explicación genuinamente
coherente es **Pythia 160M, man_couch/variante "weak"**:

> "It refers to the man because **he was too weak to lift the couch**."

Esto conecta la premisa con el pronombre de forma sensata y gramaticalmente
completa. Pero es importante no sobre-interpretarlo: es básicamente una
**reformulación de la premisa** (la oración ya decía "the man couldn't lift the
couch... too weak"), no evidencia de un razonamiento nuevo sobre por qué "it" se
refiere al hombre y no al sofá. Ni siquiera este caso, el más coherente de los 18,
revela nada sobre el mecanismo posicional real.

## 7. Conclusión honesta, acotada a esta escala (70–160M parámetros)

El auto-reporte y el mecanismo interno están desconectados en estos modelos, pero
**no por confabulación convincente** (una narrativa falsa pero creíble que oculte
el verdadero motivo) — la desconexión se explica, en su mayor parte, por una
**falta de capacidad para articular contenido real en absoluto**, cierto o falso.
La incoherencia lingüística (circularidad, glitches, tangentes fabricadas) es en sí
misma la principal razón por la que el auto-reporte no puede reflejar el mecanismo:
no hay suficiente "ahí" en el texto generado como para que revele o esconda nada de
forma consistente.

Ese déficit de coherencia mejora de forma gradual con más parámetros dentro del
rango estudiado (70M → 124M [GPT-2 small] → 160M), pero **no cierra la brecha con
el mecanismo**:
ni siquiera Pythia 160M, el modelo más grande de esta muestra y el único que
produjo un auto-reporte genuinamente coherente, mostró jamás indicio del sesgo
posicional que la ablation causal confirmó como mecanismo real (al menos en Pythia
70M y GPT-2 small).

## 9. Fase 6: activation patching cruzado — dos mecanismos causales desenredados

La ablation de la Fase 1 mostró *que* la cabeza (0,2) tiene efecto causal, pero no
si el adjetivo mismo llega a usarse en algún punto de la red. El activation
patching cruzado (transplantar activaciones entre las dos variantes de una
oración) responde esa pregunta, y además permite separar dos mecanismos que
conviven en la misma capa.

### 9.1 El adjetivo sí tiene efecto causal — pero no el que se esperaría

Se transplantó el residual stream completo (`resid_post`) en la posición del
adjetivo desambiguador, de una variante hacia la otra, capa por capa, en
trophy_suitcase, car_truck y man_couch, en GPT-2 small y Pythia 70M.

El efecto es real y grande, concentrado en las primeras capas de cada red y cae a
cero hacia la mitad (capa 4 de 6 en Pythia 70M; capa 9–11 de 12 en GPT-2 small).
Esto ya refuta que el adjetivo "nunca se use en absoluto". Pero el patrón exacto
es revelador. Ejemplo (GPT-2 small, trophy_suitcase, capa 0):

| Corrida | Margen natural (propio) |
|---|---|
| "big" sola | +2.68 (favorece trophy) |
| "small" sola | +3.41 (relativo a trophy — también favorece trophy, aunque ahí sea incorrecto) |
| Parche "big" → "small" | **+2.66** |
| Parche "small" → "big" | **+3.42** |

El valor después de parchar no converge a "la respuesta correcta": converge casi
exactamente al **valor natural propio del donante** (+2.66 ≈ 2.68; +3.42 ≈ 3.41).
Este patrón se repitió con precisión similar en las otras 2 oraciones y en Pythia
70M. Conclusión: el adjetivo sí se usa causalmente, pero lo que transporta no es
una comparación semántica correcta de tamaño/peso — es una **preferencia fija
propia de cada palabra adjetivo**, que en la mayoría de los casos empuja en la
misma dirección tanto para "big" como para "small" (o "slow"/"wide",
"weak"/"heavy"). Esto explica mecánicamente por qué el acierto natural (Fase 1)
queda cerca del azar: el adjetivo influye, pero no discrimina bien entre las dos
variantes. *(Gráfico completo por capa: `outputs/fase6/patching_por_capa.html`,
no incluido en el repo.)*

### 9.2 Cabeza (0,2) vs. resto de la capa 0

Como el efecto se concentra justo en la capa 0 — la misma de la cabeza con sesgo
posicional —, se probó si es el mismo mecanismo. Se parchó SOLO la salida
(`hook_z`) de la cabeza (0,2), dejando el resto de la capa intacta (las demás
cabezas y el MLP siguen viendo la mezcla real y recalculan con normalidad), y se
comparó contra el patching del residual completo:

| Modelo | Fracción del efecto de capa 0 explicada por la cabeza (0,2) |
|---|---|
| GPT-2 small | ~0% |
| Pythia 70M | ~9% |

La cabeza (0,2) no explica el efecto del adjetivo — es un mecanismo distinto del
sesgo posicional hallado por ablation.

### 9.3 Barrido completo: el MLP de la capa 0 domina, pero no siempre limpiamente

Se barrieron una por una las 12 cabezas (GPT-2) / 8 cabezas (Pythia 70M) más el
MLP de la capa 0, primero en las 3 oraciones ya estudiadas, y después en las 5
oraciones restantes de la muestra de 8 (para chequear si generaliza):

| Modelo | Muestra | MLP capa 0 | Mejor cabeza individual |
|---|---|---|---|
| GPT-2 small | 3 oraciones (6 casos) | 87% | cabeza 3: 9% |
| GPT-2 small | 5 oraciones (10 casos) | 99% | cabeza 8: 24% |
| Pythia 70M | 3 oraciones (6 casos) | 67% | cabeza 3: 11% |
| Pythia 70M | 5 oraciones (10 casos) | 35% (empatada con cabeza 3: 35%) | cabeza 3: 35% |

**En GPT-2 small el hallazgo generaliza de forma robusta**: el MLP de la capa 0
domina ampliamente en ambas muestras (87% y 99%); ninguna cabeza individual supera
el 24%.

**En Pythia 70M el hallazgo es menos limpio de lo que parecía con solo 3
oraciones.** En la muestra original el MLP dominaba (67%); al extender a las 5
oraciones restantes, una cabeza distinta — capa 0, cabeza 3 (no la cabeza 2 de la
ablation) — empata con el MLP (35% cada una). Combinando las 8 oraciones
completas, el MLP explica en promedio ~47% y la cabeza 3 ~26%: el MLP sigue siendo
el mayor contribuyente individual, pero la dominancia es bastante menos clara que
en GPT-2 small, y sugiere que en Pythia 70M el mecanismo puede estar más repartido
entre el MLP y al menos una cabeza adicional, en vez de concentrado en un solo
componente.

### 9.4 Conclusión de la Fase 6

Quedan identificados y desenredados dos mecanismos causales distintos,
coexistiendo en la capa 0, ninguno de los dos implementando una resolución
semántica real de la correferencia:

1. **Cabeza (0,2)** (GPT-2 small y Pythia 70M, hallada por ablation en la Fase 1):
   sesgo posicional hacia el sujeto de la oración, independiente del adjetivo.
2. **MLP de la capa 0** (y, en Pythia 70M, también la cabeza (0,3) en un grado
   comparable): carga la identidad léxica del adjetivo, pero como una preferencia
   fija por palabra que no discrimina correctamente entre las dos variantes — no
   una comparación semántica de tamaño/peso.

Esto refuerza, con evidencia causal más fina, la misma conclusión honesta de las
Fases 1–2: la desconexión entre auto-reporte y mecanismo no es por falta de
mecanismo — hay al menos dos, verificados causalmente — sino porque ninguno de
los mecanismos reales implementa lo que una persona describiría como "resolver la
referencia por significado", y el auto-reporte tampoco los describe.

## 10. Limitaciones

- **Tamaño de muestra chico**: 8 oraciones para el análisis de atención agregada y
  para el barrido de componentes de la Fase 6; solo 3 (trophy_suitcase, car_truck,
  man_couch) para la ablation original, el patching por capa, y los
  auto-reportes. Son pocos casos bien mirados, no una muestra estadísticamente
  representativa — el propio hallazgo de la Fase 6 (sección 9.3) muestra que
  ampliar de 3 a 8 oraciones cambió la conclusión para Pythia 70M.
- **Un solo tipo de tarea**: correferencia ambigua tipo Winograd, con una
  estructura sintáctica similar en las 8 oraciones (sujeto — verbo — objeto, con
  "the" + sustantivo). No se probó si el sesgo hacia el sujeto o el efecto del MLP
  se sostienen con otras estructuras sintácticas o tipos de ambigüedad.
- **El patching y el desenredado de componentes no se probaron en Pythia 160M**:
  se limitaron a GPT-2 small y Pythia 70M, los dos modelos donde la ablation de la
  Fase 1 encontró un mecanismo limpio que extender.
- **El desenredado cabeza-vs-MLP se hizo solo en la capa 0**: el efecto del
  residual completo seguía siendo apreciable en las capas 1–3 (sobre todo en
  Pythia 70M); no se repitió el barrido de componentes en esas capas.
- **Un solo tipo de ablation/patching**: zero-ablation y patching directo
  (transplante 1:1 de activaciones). No se probó mean-ablation ni combinaciones de
  varios componentes a la vez.
- **Búsqueda de cabezas no exhaustiva en el cierre de la Fase 1**: en Pythia 160M,
  la cabeza elegida empató con otras tres candidatas que no se llegaron a
  inspeccionar.
- **Nada de esto se generaliza más allá de 70–160M parámetros.** No se afirma
  nada sobre cómo funcionan modelos grandes (GPT-4, Claude, etc.), ni sobre "los
  LLMs" en general. Los hallazgos son válidos únicamente para los tres modelos y
  las oraciones puntuales usadas en este documento.
