"""Fase 6 - activation patching cruzado sobre el residual stream.

Extiende el hallazgo de ablation (Fase 1): en vez de apagar una cabeza,
transplanta el residual stream completo en la posicion del adjetivo
desambiguador, de una variante a la otra, capa por capa. Pregunta: ¿en
alguna capa la informacion del adjetivo SI tiene efecto causal sobre la
prediccion, antes de perderse frente al sesgo posicional (capa 0/cabeza 2)
identificado en la Fase 1? O ¿ninguna capa muestra efecto, reforzando que la
informacion semantica del adjetivo nunca se usa en absoluto?

Mismos 3 pares de oraciones ya verificados (trophy_suitcase, car_truck,
man_couch), en GPT-2 small y Pythia 70M. No se amplia la muestra en esta
primera pasada.
"""

import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.behavior import continuation_log_prob
from src.models import load_model
from src.patching import patch_resid_por_capa
from src.winograd_sentences import ORACIONES

MODELOS = ["gpt2", "pythia-70m"]
ORACIONES_A_PROBAR = ["trophy_suitcase", "car_truck", "man_couch"]
DIRECTORIO_SALIDA = Path(__file__).resolve().parent.parent / "outputs" / "fase6"


def oracion_por_nombre(nombre: str) -> dict:
    for o in ORACIONES:
        if o["nombre"] == nombre:
            return o
    raise KeyError(nombre)


def margen_base(modelo, frase_objetivo: str, candidatos: list[str], esperado_objetivo: str) -> float:
    otro = candidatos[0] if esperado_objetivo == candidatos[1] else candidatos[1]
    prefijo = frase_objetivo + " It refers to the"
    lp_correcto = continuation_log_prob(modelo, prefijo, " " + esperado_objetivo)
    lp_incorrecto = continuation_log_prob(modelo, prefijo, " " + otro)
    return lp_correcto - lp_incorrecto


def main() -> None:
    resumen_por_modelo: dict[str, dict[int, list[float]]] = {}

    for nombre_tl in MODELOS:
        print(f"\n{'#' * 70}\n# {nombre_tl}\n{'#' * 70}")
        modelo = load_model(nombre_tl)
        n_layers = modelo.cfg.n_layers
        acumulado_por_capa: dict[int, list[float]] = {capa: [] for capa in range(n_layers)}

        for nombre in ORACIONES_A_PROBAR:
            oracion = oracion_por_nombre(nombre)
            candidatos = oracion["candidatos"]
            variantes = list(oracion["variantes"].items())  # [(d1, e1), (d2, e2)]

            direcciones = [
                (variantes[0], variantes[1]),
                (variantes[1], variantes[0]),
            ]
            for (d_donante, _e_donante), (d_objetivo, e_objetivo) in direcciones:
                frase_donante = oracion["plantilla"].format(d=d_donante)
                frase_objetivo = oracion["plantilla"].format(d=d_objetivo)

                base = margen_base(modelo, frase_objetivo, candidatos, e_objetivo)
                margenes_por_capa = patch_resid_por_capa(
                    modelo, frase_donante, d_donante, frase_objetivo, d_objetivo,
                    candidatos, e_objetivo,
                )

                print(f"\n  {nombre}: {d_donante} -> {d_objetivo} (objetivo espera '{e_objetivo}')")
                print(f"    margen base (sin parche): {base:+.3f}")
                for capa, margen_parchado in margenes_por_capa.items():
                    cambio = margen_parchado - base
                    acumulado_por_capa[capa].append(cambio)
                    print(
                        f"    capa {capa}: margen parchado={margen_parchado:+.3f}  "
                        f"cambio={cambio:+.3f}"
                    )

        print(
            f"\n  === Resumen por capa ({nombre_tl}), 6 mediciones por capa "
            f"(3 oraciones x 2 direcciones) ==="
        )
        for capa in range(n_layers):
            valores = acumulado_por_capa[capa]
            promedio = sum(valores) / len(valores)
            print(
                f"    capa {capa}: cambio promedio={promedio:+.3f}  "
                f"(min={min(valores):+.3f}, max={max(valores):+.3f})"
            )

        resumen_por_modelo[nombre_tl] = acumulado_por_capa

    # --- Grafico simple: capa vs cambio promedio, una linea por modelo ---
    DIRECTORIO_SALIDA.mkdir(parents=True, exist_ok=True)
    figura = go.Figure()
    for nombre_tl, acumulado_por_capa in resumen_por_modelo.items():
        capas = sorted(acumulado_por_capa.keys())
        promedios = [sum(acumulado_por_capa[c]) / len(acumulado_por_capa[c]) for c in capas]
        figura.add_trace(go.Scatter(x=capas, y=promedios, mode="lines+markers", name=nombre_tl))
    figura.update_layout(
        title="Activation patching cruzado: cambio promedio en el margen de logit por capa",
        xaxis_title="Capa (resid_post)",
        yaxis_title="Cambio promedio en el margen (parchado - base)",
        yaxis_zeroline=True,
    )
    archivo = DIRECTORIO_SALIDA / "patching_por_capa.html"
    figura.write_html(archivo)
    print(f"\nGrafico guardado en: {archivo}")


if __name__ == "__main__":
    main()
