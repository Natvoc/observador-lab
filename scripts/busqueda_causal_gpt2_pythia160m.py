"""Fase 1 (cierre) - buscar y probar causalmente una cabeza analoga, en GPT-2 small y Pythia 160M.

Mismo metodo ya aplicado a Pythia 70M (capa 0, cabeza 2): busqueda por cabeza
individual sobre las mismas 3 oraciones usadas en esa ablation (trophy_suitcase,
car_truck, man_couch), eleccion de la cabeza que resuelve mas oraciones,
inspeccion manual de su patron de atencion completo, y zero-ablation causal
sobre esa cabeza -- para saber si cada modelo tiene un mecanismo causal
comparable (sesgo hacia el sujeto) o distinto (resolucion real, o nada claro).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attention import cabezas_que_resuelven, find_token_span, get_attention_patterns
from src.behavior import continuation_log_prob
from src.models import load_model
from src.winograd_sentences import ORACIONES

MODELOS = ["gpt2", "pythia-160m"]
ORACIONES_A_PROBAR = ["trophy_suitcase", "car_truck", "man_couch"]
SONDEO = " It refers to the"


def oracion_por_nombre(nombre: str) -> dict:
    for o in ORACIONES:
        if o["nombre"] == nombre:
            return o
    raise KeyError(nombre)


def buscar_candidatas(modelo) -> dict[tuple[int, int], int]:
    conteo: dict[tuple[int, int], int] = {}
    for nombre in ORACIONES_A_PROBAR:
        oracion = oracion_por_nombre(nombre)
        for ch in cabezas_que_resuelven(modelo, oracion):
            conteo[ch] = conteo.get(ch, 0) + 1
    return conteo


def mostrar_top_atencion(modelo, frase: str, capa: int, cabeza: int, desde_palabra: str, candidatos: list[str]) -> None:
    resultado = get_attention_patterns(modelo, frase)
    tokens = resultado.tokens
    idx_desde = find_token_span(tokens, desde_palabra)[1] - 1
    fila = resultado.patterns[capa][cabeza, idx_desde, :]

    marcas = {}
    for candidato in candidatos:
        span_c = find_token_span(tokens, candidato)
        if span_c is not None:
            for i in range(span_c[0], span_c[1]):
                marcas[i] = candidato

    pares = [(peso, i, tok) for i, (tok, peso) in enumerate(zip(tokens, fila.tolist()))]
    for peso, i, tok in sorted(pares, key=lambda x: -x[0])[:6]:
        etiqueta = f" <-- {marcas[i]}" if i in marcas else ""
        print(f"      [{i:2d}] {tok!r:12s} peso={peso:.4f}{etiqueta}")


def hook_zero_ablation_factory(cabeza: int):
    def hook_fn(z, hook):
        z[:, :, cabeza, :] = 0.0
        return z

    return hook_fn


def correr_ablation(modelo, capa: int, cabeza: int) -> None:
    fwd_hooks = [(f"blocks.{capa}.attn.hook_z", hook_zero_ablation_factory(cabeza))]

    for nombre in ORACIONES_A_PROBAR:
        oracion = oracion_por_nombre(nombre)
        print(f"\n  === {nombre} ===")
        for disambiguador, esperado in oracion["variantes"].items():
            candidatos = oracion["candidatos"]
            otro = candidatos[0] if esperado == candidatos[1] else candidatos[1]
            frase = oracion["plantilla"].format(d=disambiguador)
            prefijo = frase + SONDEO

            lp_c_base = continuation_log_prob(modelo, prefijo, " " + esperado)
            lp_i_base = continuation_log_prob(modelo, prefijo, " " + otro)
            margen_base = lp_c_base - lp_i_base

            lp_c_abl = continuation_log_prob(modelo, prefijo, " " + esperado, fwd_hooks=fwd_hooks)
            lp_i_abl = continuation_log_prob(modelo, prefijo, " " + otro, fwd_hooks=fwd_hooks)
            margen_abl = lp_c_abl - lp_i_abl

            print(f"    Variante '{disambiguador}' (esperado: {esperado}):")
            print(f"      Sin ablation: margen={margen_base:+.3f}")
            print(f"      Con ablation: margen={margen_abl:+.3f}")
            print(f"      Cambio: {margen_abl - margen_base:+.3f}")


def main() -> None:
    for nombre_tl in MODELOS:
        print(f"\n{'#' * 70}\n# {nombre_tl}\n{'#' * 70}")
        modelo = load_model(nombre_tl)

        conteo = buscar_candidatas(modelo)
        if not conteo:
            print("  Ninguna cabeza resuelve ninguna de las 3 oraciones. Sin candidata causal que probar.")
            continue

        ordenadas = sorted(conteo.items(), key=lambda x: -x[1])
        print(f"  Cabezas que resuelven al menos 1 de las 3 oraciones: {len(ordenadas)}")
        print("  Top 5 por cantidad de oraciones resueltas:")
        for (capa, cabeza), n in ordenadas[:5]:
            print(f"    capa {capa}, cabeza {cabeza}: {n}/3")

        (mejor_capa, mejor_cabeza), mejor_n = ordenadas[0]
        print(f"\n  Candidata elegida: capa {mejor_capa}, cabeza {mejor_cabeza} ({mejor_n}/3 oraciones)")

        print("\n  Patron de atencion (6 tokens con mas peso, por caso):")
        for nombre in ORACIONES_A_PROBAR:
            oracion = oracion_por_nombre(nombre)
            for disambiguador, esperado in oracion["variantes"].items():
                frase = oracion["plantilla"].format(d=disambiguador)
                print(f"\n    {nombre} / '{disambiguador}' (esperado: {esperado}):")
                mostrar_top_atencion(
                    modelo, frase, mejor_capa, mejor_cabeza, disambiguador, oracion["candidatos"]
                )

        print(f"\n  Ablation causal sobre capa {mejor_capa}, cabeza {mejor_cabeza}:")
        correr_ablation(modelo, mejor_capa, mejor_cabeza)


if __name__ == "__main__":
    main()
