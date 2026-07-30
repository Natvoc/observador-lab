"""Fase 1 - prueba causal (zero-ablation) sobre la unica cabeza con patron interpretable.

De las 4 cabezas candidatas inspeccionadas a mano, solo Pythia 70M (capa 0,
cabeza 2) mostro un patron de atencion interpretable: atiende directamente a
los dos sustantivos candidatos (por encima de palabras funcionales) y los
ordena correctamente en varias oraciones. Pero atencion correcta no prueba
causalidad -- la cabeza podria estar "mirando" al sustantivo correcto sin ser
lo que efectivamente determina la prediccion final del modelo.

Este script apaga esa cabeza especifica (zero-ablation: se reemplaza por
cero su salida `z`, en TODAS las posiciones de la secuencia, via un hook de
transformer_lens sobre `blocks.{capa}.attn.hook_z`, antes de la proyeccion de
salida W_O) y vuelve a correr el mismo sondeo comportamental de
`src/behavior.py` (log-probabilidad de cada sustantivo candidato como
continuacion de "... It refers to the ___"). Se compara el margen
(log-prob del correcto menos el incorrecto) antes y despues de apagar la
cabeza:

- Si el margen se derrumba o se invierte al apagarla: evidencia de que la
  cabeza es causalmente responsable de la prediccion.
- Si el margen practicamente no cambia: el patron de atencion que vimos es
  correlacional, no causal -- otra parte del modelo determina el output.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.behavior import continuation_log_prob
from src.models import load_model
from src.winograd_sentences import ORACIONES

NOMBRE_MODELO = "pythia-70m"
CAPA = 0
CABEZA = 2
ORACIONES_A_PROBAR = ["trophy_suitcase", "car_truck", "man_couch"]
SONDEO = " It refers to the"


def oracion_por_nombre(nombre: str) -> dict:
    for o in ORACIONES:
        if o["nombre"] == nombre:
            return o
    raise KeyError(nombre)


def hook_zero_ablation(z, hook):
    z[:, :, CABEZA, :] = 0.0
    return z


def main() -> None:
    modelo = load_model(NOMBRE_MODELO)
    nombre_hook = f"blocks.{CAPA}.attn.hook_z"
    fwd_hooks = [(nombre_hook, hook_zero_ablation)]

    for nombre in ORACIONES_A_PROBAR:
        oracion = oracion_por_nombre(nombre)
        print(f"\n=== {nombre} ===")

        for disambiguador, esperado in oracion["variantes"].items():
            candidatos = oracion["candidatos"]
            otro = candidatos[0] if esperado == candidatos[1] else candidatos[1]
            frase = oracion["plantilla"].format(d=disambiguador)
            prefijo = frase + SONDEO

            lp_correcto_base = continuation_log_prob(modelo, prefijo, " " + esperado)
            lp_incorrecto_base = continuation_log_prob(modelo, prefijo, " " + otro)
            margen_base = lp_correcto_base - lp_incorrecto_base

            lp_correcto_abl = continuation_log_prob(
                modelo, prefijo, " " + esperado, fwd_hooks=fwd_hooks
            )
            lp_incorrecto_abl = continuation_log_prob(
                modelo, prefijo, " " + otro, fwd_hooks=fwd_hooks
            )
            margen_abl = lp_correcto_abl - lp_incorrecto_abl

            print(f"  Variante '{disambiguador}' (esperado: {esperado}, incorrecto: {otro}):")
            print(
                f"    Sin ablation:  logprob({esperado})={lp_correcto_base:7.3f}  "
                f"logprob({otro})={lp_incorrecto_base:7.3f}  margen={margen_base:+.3f}"
            )
            print(
                f"    Con ablation:  logprob({esperado})={lp_correcto_abl:7.3f}  "
                f"logprob({otro})={lp_incorrecto_abl:7.3f}  margen={margen_abl:+.3f}"
            )
            print(f"    Cambio en el margen (ablation - base): {margen_abl - margen_base:+.3f}")


if __name__ == "__main__":
    main()
