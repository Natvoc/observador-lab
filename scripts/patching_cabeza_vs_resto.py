"""Fase 6 (continuacion) - desenredar cabeza (0,2) vs. resto de la capa 0.

El patching cruzado del residual stream completo (`patching_causal.py`) mostro
un efecto causal fuerte en la capa 0, que decae a cero hacia la mitad de la
red. Como `resid_post` de la capa 0 incluye la contribucion de TODAS las
cabezas mas el MLP, no se podia saber si ese efecto es la cabeza (0,2) que ya
identificamos por ablation, o si hay un mecanismo adicional en el resto de la
capa.

Este script parcha SOLO la salida (`hook_z`) de la cabeza (0,2) -- dejando el
resto de la capa 0 (las otras cabezas y el MLP) sin tocar, viendo la mezcla
real de la oracion objetivo -- y compara el efecto resultante contra el
patching del residual completo de esa misma capa, para los mismos 3 pares de
oraciones en GPT-2 small y Pythia 70M.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import load_model
from src.patching import margen_sin_parche, patch_head_output, patch_resid_por_capa
from src.winograd_sentences import ORACIONES

MODELOS = ["gpt2", "pythia-70m"]
ORACIONES_A_PROBAR = ["trophy_suitcase", "car_truck", "man_couch"]
CAPA = 0
CABEZA = 2


def oracion_por_nombre(nombre: str) -> dict:
    for o in ORACIONES:
        if o["nombre"] == nombre:
            return o
    raise KeyError(nombre)


def main() -> None:
    for nombre_tl in MODELOS:
        print(f"\n{'#' * 70}\n# {nombre_tl}\n{'#' * 70}")
        modelo = load_model(nombre_tl)

        fracciones: list[float] = []

        for nombre in ORACIONES_A_PROBAR:
            oracion = oracion_por_nombre(nombre)
            candidatos = oracion["candidatos"]
            variantes = list(oracion["variantes"].items())

            direcciones = [
                (variantes[0], variantes[1]),
                (variantes[1], variantes[0]),
            ]
            for (d_donante, _e_donante), (d_objetivo, e_objetivo) in direcciones:
                frase_donante = oracion["plantilla"].format(d=d_donante)
                frase_objetivo = oracion["plantilla"].format(d=d_objetivo)

                margen_base = margen_sin_parche(modelo, frase_objetivo, candidatos, e_objetivo)

                margenes_resid = patch_resid_por_capa(
                    modelo, frase_donante, d_donante, frase_objetivo, d_objetivo,
                    candidatos, e_objetivo,
                )
                margen_resid_completo = margenes_resid[CAPA]

                margen_solo_cabeza = patch_head_output(
                    modelo, frase_donante, d_donante, frase_objetivo, d_objetivo,
                    candidatos, e_objetivo, CAPA, CABEZA,
                )

                cambio_resid_completo = margen_resid_completo - margen_base
                cambio_solo_cabeza = margen_solo_cabeza - margen_base

                if abs(cambio_resid_completo) > 1e-6:
                    fraccion = cambio_solo_cabeza / cambio_resid_completo
                    fracciones.append(fraccion)
                    fraccion_str = f"{fraccion:.2f}"
                else:
                    fraccion_str = "n/a (cambio_resid_completo ~ 0)"

                print(f"\n  {nombre}: {d_donante} -> {d_objetivo} (objetivo espera '{e_objetivo}')")
                print(f"    margen base:                         {margen_base:+.3f}")
                print(
                    f"    margen parchando resid completo capa0: {margen_resid_completo:+.3f}  "
                    f"(cambio {cambio_resid_completo:+.3f})"
                )
                print(
                    f"    margen parchando SOLO cabeza (0,{CABEZA}):  {margen_solo_cabeza:+.3f}  "
                    f"(cambio {cambio_solo_cabeza:+.3f})"
                )
                print(f"    fraccion del efecto explicada por la cabeza: {fraccion_str}")

        if fracciones:
            promedio = sum(fracciones) / len(fracciones)
            print(
                f"\n  === {nombre_tl}: fraccion promedio del efecto de capa 0 explicada "
                f"por la cabeza (0,{CABEZA}): {promedio:.2f} (n={len(fracciones)}) ==="
            )


if __name__ == "__main__":
    main()
