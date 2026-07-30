"""Fase 2 - mismo auto-reporte que en Pythia 70M, pero en GPT-2 small y Pythia 160M.

Aisla una variable: ¿la incoherencia del auto-reporte que vimos en Pythia 70M
es un limite de la tarea (ningun modelo de esta escala puede articular el
mecanismo) o un limite de capacidad linguistica particular de ese modelo (no
sabe articular ninguna explicacion, cierta o falsa)? No se repite la
ablation causal aca -- solo se compara el TEXTO generado entre modelos, para
los mismos 6 casos (trophy_suitcase, car_truck, man_couch x 2 variantes).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.behavior import compare_candidates
from src.models import load_model
from src.self_report import generate_self_report
from src.winograd_sentences import ORACIONES

MODELOS_A_PROBAR = ["gpt2", "pythia-160m"]
ORACIONES_A_PROBAR = ["trophy_suitcase", "car_truck", "man_couch"]


def oracion_por_nombre(nombre: str) -> dict:
    for o in ORACIONES:
        if o["nombre"] == nombre:
            return o
    raise KeyError(nombre)


def main() -> None:
    for nombre_tl in MODELOS_A_PROBAR:
        print(f"\n{'#' * 70}\n# {nombre_tl}\n{'#' * 70}")
        modelo = load_model(nombre_tl)

        for nombre in ORACIONES_A_PROBAR:
            oracion = oracion_por_nombre(nombre)
            candidatos = oracion["candidatos"]
            sujeto = candidatos[0]
            print(f"\n=== {nombre} ===")

            for disambiguador, esperado in oracion["variantes"].items():
                frase = oracion["plantilla"].format(d=disambiguador)
                prefijo = frase + " It refers to the"

                log_probs = compare_candidates(modelo, prefijo, candidatos)
                eleccion = max(log_probs, key=log_probs.get)
                auto_reporte = generate_self_report(modelo, frase, eleccion)

                print(f"\n  --- Variante '{disambiguador}' (esperado: {esperado}) ---")
                print(f"  Frase: {frase}")
                print(
                    f"  Eleccion: '{eleccion}' | sujeto: '{sujeto}' | acerto: {eleccion == esperado} "
                    f"| eligio el sujeto: {eleccion == sujeto}"
                )
                print(f"  Auto-reporte: \"It refers to the {eleccion} because{auto_reporte}\"")


if __name__ == "__main__":
    main()
