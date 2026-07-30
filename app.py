"""observador-lab — interfaz Gradio.

Compara, para una oracion tipo Winograd, el mapa de atencion (un indicio
observacional) con el auto-reporte que el propio modelo genera sobre su
eleccion. Corre 100% local, sin conexion a internet salvo la descarga inicial
de cada modelo.

Distincion clave (ver PROYECTO.md, "Actualizacion de metodologia", y
RESULTADOS.md): la atencion, incluso por cabeza individual, resulto ser
correlacional. Solo se verifico causalmente (via ablation) que capa 0/cabeza 2
de GPT-2 small y Pythia 70M implementa un sesgo posicional hacia el sujeto de
la oracion -- no una resolucion semantica del adjetivo. Esta interfaz separa
claramente ese hallazgo verificado de la exploracion libre para cualquier
otra capa/cabeza u oracion.
"""

import re

import gradio as gr

from src.attention import get_attention_patterns
from src.behavior import compare_candidates
from src.models import CONFIGURACION_MODELO, MODELOS_DISPONIBLES, load_model
from src.self_report import generate_continuation, generate_self_report
from src.visualization import plot_attention_heatmap
from src.winograd_sentences import ORACIONES

# Modelos donde la Fase 1 confirmo causalmente (via ablation) un mecanismo
# limpio en capa 0/cabeza 2. Pythia 160M queda afuera: el resultado ahi fue
# ambiguo (efecto causal real, pero de direccion inconsistente).
MODELOS_CON_MECANISMO_VERIFICADO = {"gpt2", "pythia-70m"}
CAPA_VERIFICADA = 0
CABEZA_VERIFICADA = 2

# Las 3 oraciones (6 variantes) usadas en la Fase 2 para comparar auto-reporte
# contra el mecanismo causal verificado (ver RESULTADOS.md, seccion 5).
NOMBRES_ORACIONES_ESTUDIADAS = {"trophy_suitcase", "car_truck", "man_couch"}


def normalizar_oracion(texto: str) -> str:
    """trim + minusculas + espacios multiples colapsados, para comparar oraciones."""
    return re.sub(r"\s+", " ", texto.strip().lower())


CASOS_VERIFICADOS: dict[str, dict] = {}
EJEMPLOS: list[list] = []
for _oracion in ORACIONES:
    if _oracion["nombre"] not in NOMBRES_ORACIONES_ESTUDIADAS:
        continue
    for _disambiguador, _esperado in _oracion["variantes"].items():
        _frase = _oracion["plantilla"].format(d=_disambiguador)
        CASOS_VERIFICADOS[normalizar_oracion(_frase)] = {
            "candidatos": _oracion["candidatos"],
            "esperado": _esperado,
            "sujeto": _oracion["candidatos"][0],
        }
        EJEMPLOS.append([_frase, "Pythia 70M", CAPA_VERIFICADA, CABEZA_VERIFICADA])


TEXTO_LIMITACIONES = """
### Limitaciones (leer antes de explorar)

- Solo funciona de forma confiable con **frases cortas en inglés**. Estos modelos
  fueron entrenados mayoritariamente en inglés.
- Solo tiene sentido interpretativo para el tipo de tarea para el que fue diseñado:
  **correferencia ambigua** (un pronombre que puede referirse a más de un
  sustantivo). Otro tipo de frases (razonamiento, opinión, matemáticas) van a
  mostrar igual el mapa de atención, pero la comparación pierde su marco de
  referencia.
- Estos modelos (70–160M parámetros) pueden fallar o dar auto-reportes
  incoherentes en frases largas o complejas — eso es, en sí mismo, un resultado
  válido y observable, no un error del sistema.
- **El mapa de atención es un indicio observacional, no una prueba de mecanismo
  causal.** Solo se verificó causalmente (vía ablation) para capa 0/cabeza 2 de
  GPT-2 small y Pythia 70M, sobre 3 oraciones de prueba — no para Pythia 160M, ni
  para ninguna otra capa/cabeza, ni como garantía sobre cualquier oración nueva.
  Ver [`RESULTADOS.md`](./RESULTADOS.md) para el detalle completo.
"""


def actualizar_selector_capas_cabezas(modelo_display: str):
    nombre_tl = MODELOS_DISPONIBLES[modelo_display]
    cfg = CONFIGURACION_MODELO[nombre_tl]
    return (
        gr.Dropdown(choices=list(range(cfg["n_layers"])), value=0),
        gr.Dropdown(choices=list(range(cfg["n_heads"])), value=0),
    )


def analizar(frase: str, modelo_display: str, capa: int, cabeza: int):
    if not frase or not frase.strip():
        vacio = "_(escribí una oración primero)_"
        return None, vacio, vacio, vacio

    nombre_tl = MODELOS_DISPONIBLES[modelo_display]
    modelo = load_model(nombre_tl)
    cfg = CONFIGURACION_MODELO[nombre_tl]
    capa = min(int(capa), cfg["n_layers"] - 1)
    cabeza = min(int(cabeza), cfg["n_heads"] - 1)

    # --- Panel de atencion ---
    resultado = get_attention_patterns(modelo, frase)
    patron_cabeza = resultado.patterns[capa][cabeza]
    figura = plot_attention_heatmap(resultado.tokens, patron_cabeza)

    es_cabeza_verificada = (
        nombre_tl in MODELOS_CON_MECANISMO_VERIFICADO
        and capa == CAPA_VERIFICADA
        and cabeza == CABEZA_VERIFICADA
    )
    if es_cabeza_verificada:
        nota_atencion = (
            f"**Cabeza con mecanismo verificado por ablation causal.** En "
            f"{modelo_display}, capa {CAPA_VERIFICADA}/cabeza {CABEZA_VERIFICADA} "
            "se confirmó causalmente (ver `RESULTADOS.md`, secciones 3–4) como "
            "responsable de un **sesgo posicional hacia el sujeto de la "
            "oración** — no una resolución semántica del adjetivo. Esto se "
            "verificó sobre 3 oraciones de prueba: es evidencia sobre esta "
            "cabeza del modelo, no una garantía sobre la oración que escribiste."
        )
    else:
        nota_atencion = (
            "**Exploración libre — sin verificación causal.** Este patrón es "
            "un indicio observacional. Puede ser un artefacto (attention sink "
            "al primer token, cabeza de sintaxis/función) y no necesariamente "
            "el mecanismo real detrás de la respuesta del modelo, aunque la "
            "oración sea una de las ya estudiadas: la verificación causal fue "
            "específica de la cabeza (0, 2), no de la oración en general."
        )

    # --- Panel de auto-reporte ---
    caso = CASOS_VERIFICADOS.get(normalizar_oracion(frase))
    frase_limpia = frase.strip()

    if caso is not None and nombre_tl in MODELOS_CON_MECANISMO_VERIFICADO:
        candidatos = caso["candidatos"]
        prefijo = frase_limpia + " It refers to the"
        log_probs = compare_candidates(modelo, prefijo, candidatos)
        eleccion = max(log_probs, key=log_probs.get)
        continuacion = generate_self_report(modelo, frase_limpia, eleccion)
        texto_autoreporte = f'"It refers to the {eleccion} because{continuacion}"'

        nota_autoreporte = (
            "**Esta oración es una de las 3 estudiadas en la Fase 2** (ver "
            f"`RESULTADOS.md`, sección 5). Sujeto de la oración: "
            f"'{caso['sujeto']}' · respuesta esperada por sentido común: "
            f"'{caso['esperado']}'. El auto-reporte de arriba **no menciona ni "
            "insinúa** el sesgo posicional hacia el sujeto — ninguno de los 18 "
            "auto-reportes generados en este proyecto lo hizo."
        )
    else:
        prefijo = frase_limpia + " It refers to"
        continuacion = generate_continuation(modelo, prefijo)
        texto_autoreporte = f'"It refers to{continuacion}"'

        nota_autoreporte = (
            "**Auto-reporte exploratorio, sin comparación validada.** Esta "
            "oración no es una de las 3 estudiadas causalmente en este "
            "proyecto (o el modelo elegido no tiene mecanismo verificado), así "
            "que no hay nada confirmado contra qué compararlo. Es solo lo que "
            'el modelo genera al pedirle que continúe explicando a qué se '
            'refiere "it".'
        )

    return figura, nota_atencion, texto_autoreporte, nota_autoreporte


with gr.Blocks(title="observador-lab") as demo:
    gr.Markdown(
        """
        # observador-lab

        Laboratorio de interpretabilidad mecanicista: compara el mapa de atención
        de un modelo de lenguaje chico con el auto-reporte que el propio modelo da
        sobre su elección, para oraciones con un pronombre ambiguo (tipo Winograd).
        Ver [`RESULTADOS.md`](./RESULTADOS.md) para los hallazgos completos.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            oracion = gr.Textbox(
                label="Oración en inglés (con un pronombre ambiguo)",
                placeholder="The trophy doesn't fit in the suitcase because it is too big.",
                lines=2,
            )
        with gr.Column(scale=1):
            modelo_dd = gr.Dropdown(
                choices=list(MODELOS_DISPONIBLES.keys()),
                value="Pythia 70M",
                label="Modelo",
            )
            capa_dd = gr.Dropdown(
                choices=list(range(CONFIGURACION_MODELO["pythia-70m"]["n_layers"])),
                value=0,
                label="Capa",
            )
            cabeza_dd = gr.Dropdown(
                choices=list(range(CONFIGURACION_MODELO["pythia-70m"]["n_heads"])),
                value=0,
                label="Cabeza",
            )

    gr.Examples(
        examples=EJEMPLOS,
        inputs=[oracion, modelo_dd, capa_dd, cabeza_dd],
        label="Ejemplos con mecanismo causal verificado (capa 0 / cabeza 2)",
    )

    boton = gr.Button("Analizar", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Mapa de atención")
            plot = gr.Plot()
            nota_atencion_md = gr.Markdown()
        with gr.Column():
            gr.Markdown("### Auto-reporte del modelo")
            autoreporte_md = gr.Markdown()
            nota_autoreporte_md = gr.Markdown()

    gr.Markdown(TEXTO_LIMITACIONES)

    modelo_dd.change(
        actualizar_selector_capas_cabezas,
        inputs=modelo_dd,
        outputs=[capa_dd, cabeza_dd],
    )
    boton.click(
        analizar,
        inputs=[oracion, modelo_dd, capa_dd, cabeza_dd],
        outputs=[plot, nota_atencion_md, autoreporte_md, nota_autoreporte_md],
    )


if __name__ == "__main__":
    demo.launch()
