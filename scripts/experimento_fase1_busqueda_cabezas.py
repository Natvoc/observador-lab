"""Fase 1 - busqueda de cabezas individuales que distingan el contexto.

El promedio sobre todas las cabezas y capas (`experimento_fase1_atencion.py`)
no mostro senal real: en 23 de 24 pares oracion-modelo, la respuesta con mas
atencion no cambiaba segun la palabra desambiguadora. Este script pregunta si,
aun asi, hay UNA cabeza especifica (de hasta 144 por modelo) que si distinga
correctamente entre las dos variantes -- algo que el promedio podria estar
diluyendo.

Definicion de "una cabeza resuelve un par de variantes": en AMBAS variantes,
esa cabeza le da mas atencion (desde la palabra desambiguadora) al sustantivo
correcto que al incorrecto. Por construccion esto es mas exigente que "le
achunto en una variante": una cabeza con una preferencia fija (que no
reacciona al contexto) nunca puede resolver las dos variantes a la vez,
porque cada variante espera un sustantivo distinto. Por eso encontrar una
cabeza que resuelva ambas variantes es evidencia real de sensibilidad al
contexto, no de un sesgo fijo.

Ademas de contar cuantas cabezas resuelven cada oracion, se busca si alguna
cabeza puntual (capa, cabeza) resuelve MUCHAS de las 8 oraciones -- una
cabeza que solo acierta por casualidad en una oracion no deberia repetirse en
otras; una que sea una "cabeza de correferencia" real si.
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attention import cabezas_que_resuelven
from src.models import MODELOS_DISPONIBLES, load_model
from src.winograd_sentences import ORACIONES


def main() -> None:
    for nombre_display, nombre_tl in MODELOS_DISPONIBLES.items():
        print(f"\n=== {nombre_display} ===")
        modelo = load_model(nombre_tl)
        total_cabezas = modelo.cfg.n_layers * modelo.cfg.n_heads

        conteo_por_cabeza: dict[tuple[int, int], int] = defaultdict(int)
        oraciones_sin_senal = []

        for oracion in ORACIONES:
            resolviendo = cabezas_que_resuelven(modelo, oracion)
            for ch in resolviendo:
                conteo_por_cabeza[ch] += 1

            if not resolviendo:
                oraciones_sin_senal.append(oracion["nombre"])
                print(f"  {oracion['nombre']:16s}: 0/{total_cabezas} cabezas resuelven ambas variantes")
            else:
                print(
                    f"  {oracion['nombre']:16s}: {len(resolviendo)}/{total_cabezas} cabezas "
                    f"resuelven ambas variantes -> {resolviendo}"
                )

        print(f"\n  Oraciones sin ninguna cabeza que resuelva: {oraciones_sin_senal}")

        # Cabezas "campeonas": las que resuelven mas de una oracion (posible
        # senal real y consistente, no una coincidencia puntual).
        campeonas = sorted(
            ((ch, n) for ch, n in conteo_por_cabeza.items() if n >= 2),
            key=lambda x: -x[1],
        )
        if campeonas:
            print("  Cabezas que resuelven 2 o mas de las 8 oraciones (posible senal real):")
            for (capa, cabeza), n in campeonas:
                print(f"    capa {capa}, cabeza {cabeza}: resuelve {n}/8 oraciones")
        else:
            print("  Ninguna cabeza resuelve mas de 1 de las 8 oraciones (nada consistente).")


if __name__ == "__main__":
    main()
