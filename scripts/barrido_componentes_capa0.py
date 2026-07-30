"""Fase 6 (cierre) - identificar que componente de la capa 0 carga el efecto.

El patching cruzado del residual completo (`patching_causal.py`) encontro un
efecto causal fuerte en la capa 0. Aislar la cabeza (0,2) (`patching_cabeza_vs_
resto.py`) mostro que esa cabeza especifica explica casi nada de ese efecto
(~0% en GPT-2 small, ~9% en Pythia 70M) -- la mayor parte viene de otro lado.

Este script barre, uno por uno, el resto de las cabezas de la capa 0 y el MLP
de esa misma capa (mismo metodo: parchar solo la salida de ese componente,
dejando el resto intacto), sobre los mismos 3 pares de oraciones, para
identificar cual es el que realmente carga el efecto.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import load_model
from src.patching import (
    margen_sin_parche,
    patch_head_output,
    patch_mlp_output,
    patch_resid_por_capa,
)
from src.winograd_sentences import ORACIONES

MODELOS = ["gpt2", "pythia-70m"]
ORACIONES_A_PROBAR = ["box_drawer", "bottle_cup", "knife_bread", "plane_runway", "backpack_locker"]
CAPA = 0


def oracion_por_nombre(nombre: str) -> dict:
    for o in ORACIONES:
        if o["nombre"] == nombre:
            return o
    raise KeyError(nombre)


def casos(oracion: dict):
    """Genera (frase_donante, d_donante, frase_objetivo, d_objetivo, e_objetivo) por direccion."""
    variantes = list(oracion["variantes"].items())
    for (d_donante, _e_donante), (d_objetivo, e_objetivo) in [
        (variantes[0], variantes[1]),
        (variantes[1], variantes[0]),
    ]:
        frase_donante = oracion["plantilla"].format(d=d_donante)
        frase_objetivo = oracion["plantilla"].format(d=d_objetivo)
        yield frase_donante, d_donante, frase_objetivo, d_objetivo, e_objetivo


def main() -> None:
    for nombre_tl in MODELOS:
        print(f"\n{'#' * 70}\n# {nombre_tl}\n{'#' * 70}")
        modelo = load_model(nombre_tl)
        n_heads = modelo.cfg.n_heads

        # Referencia: cambio total al parchar todo el residual de la capa 0,
        # por caso (ya calculado en la pasada anterior, se recalcula aca para
        # que este script sea autocontenido).
        referencia = []  # (oracion, direccion_str, candidatos, e_objetivo, cambio_resid_completo)
        for nombre in ORACIONES_A_PROBAR:
            oracion = oracion_por_nombre(nombre)
            for frase_donante, d_donante, frase_objetivo, d_objetivo, e_objetivo in casos(oracion):
                base = margen_sin_parche(modelo, frase_objetivo, oracion["candidatos"], e_objetivo)
                resid_completo = patch_resid_por_capa(
                    modelo, frase_donante, d_donante, frase_objetivo, d_objetivo,
                    oracion["candidatos"], e_objetivo,
                )[CAPA]
                referencia.append(
                    (
                        nombre, f"{d_donante}->{d_objetivo}", oracion["candidatos"], e_objetivo,
                        frase_donante, d_donante, frase_objetivo, d_objetivo,
                        base, resid_completo - base,
                    )
                )

        # Barrido: cada cabeza de la capa 0 y el MLP.
        componentes = [("cabeza", h) for h in range(n_heads)] + [("mlp", None)]
        promedio_fraccion_por_componente: dict[str, list[float]] = {}

        for tipo, indice in componentes:
            etiqueta = f"cabeza_{indice}" if tipo == "cabeza" else "MLP"
            fracciones = []
            for (
                nombre, direccion, candidatos, e_objetivo, frase_donante, d_donante,
                frase_objetivo, d_objetivo, base, cambio_resid_completo,
            ) in referencia:
                if tipo == "cabeza":
                    margen_componente = patch_head_output(
                        modelo, frase_donante, d_donante, frase_objetivo, d_objetivo,
                        candidatos, e_objetivo, CAPA, indice,
                    )
                else:
                    margen_componente = patch_mlp_output(
                        modelo, frase_donante, d_donante, frase_objetivo, d_objetivo,
                        candidatos, e_objetivo, CAPA,
                    )
                cambio_componente = margen_componente - base
                if abs(cambio_resid_completo) > 1e-6:
                    fracciones.append(cambio_componente / cambio_resid_completo)

            promedio_fraccion_por_componente[etiqueta] = fracciones
            if fracciones:
                promedio = sum(fracciones) / len(fracciones)
                print(f"  {etiqueta:10s}: fraccion promedio explicada = {promedio:+.3f}  (n={len(fracciones)})")

        print(f"\n  === {nombre_tl}: ranking de componentes por |fraccion promedio| ===")
        ranking = sorted(
            promedio_fraccion_por_componente.items(),
            key=lambda kv: -abs(sum(kv[1]) / len(kv[1])) if kv[1] else 0,
        )
        for etiqueta, fracciones in ranking:
            if not fracciones:
                continue
            promedio = sum(fracciones) / len(fracciones)
            print(f"    {etiqueta:10s}: {promedio:+.3f}")


if __name__ == "__main__":
    main()
