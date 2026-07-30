"""Inspeccion manual de las 4 cabezas candidatas de experimento_fase1_busqueda_cabezas.py.

No agrega oraciones ni modelos nuevos: usa las mismas 8 oraciones de
`src/winograd_sentences.py` y las mismas funciones de `src/attention.py`. Lo
unico que hace distinto es imprimir la distribucion COMPLETA de atencion
(desde la palabra desambiguadora hacia TODOS los tokens, no solo los dos
candidatos), para poder juzgar a mano si el patron de cada cabeza candidata
tiene sentido semantico o si el acierto binario anterior escondia un patron
sin interpretacion clara.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attention import find_token_span, get_attention_patterns
from src.models import load_model
from src.winograd_sentences import ORACIONES

# (nombre_transformer_lens, capa, cabeza, oraciones donde esa cabeza "resolvio"
# ambas variantes segun experimento_fase1_busqueda_cabezas.py)
CANDIDATAS = [
    ("gpt2", 8, 3, ["box_drawer", "knife_bread", "plane_runway", "backpack_locker"]),
    ("pythia-160m", 1, 6, ["trophy_suitcase", "bottle_cup", "plane_runway", "backpack_locker"]),
    ("pythia-70m", 0, 2, ["trophy_suitcase", "man_couch", "car_truck"]),
    ("pythia-70m", 3, 5, ["trophy_suitcase", "box_drawer", "man_couch"]),
]


def oracion_por_nombre(nombre: str) -> dict:
    for o in ORACIONES:
        if o["nombre"] == nombre:
            return o
    raise KeyError(nombre)


def mostrar_fila_atencion(modelo, frase: str, capa: int, cabeza: int, desde_palabra: str, candidatos: list[str]) -> None:
    resultado = get_attention_patterns(modelo, frase)
    tokens = resultado.tokens
    span = find_token_span(tokens, desde_palabra)
    idx_desde = span[1] - 1
    fila = resultado.patterns[capa][cabeza, idx_desde, :]  # [seq]

    marcas = {}
    for candidato in candidatos:
        span_c = find_token_span(tokens, candidato)
        if span_c is not None:
            for i in range(span_c[0], span_c[1]):
                marcas[i] = candidato

    print(f"    Frase: {frase}")
    print(f"    Desde token [{idx_desde}]='{tokens[idx_desde]}':")
    pares = []
    for i, (tok, peso) in enumerate(zip(tokens, fila.tolist())):
        etiqueta = f" <-- {marcas[i]}" if i in marcas else ""
        pares.append((peso, f"      [{i:2d}] {tok!r:12s} peso={peso:.4f}{etiqueta}"))
    for _, linea in sorted(pares, key=lambda x: -x[0]):
        print(linea)


def main() -> None:
    for nombre_tl, capa, cabeza, nombres_oraciones in CANDIDATAS:
        print(f"\n{'=' * 70}")
        print(f"=== {nombre_tl} - capa {capa}, cabeza {cabeza} ===")
        modelo = load_model(nombre_tl)

        for nombre_oracion in nombres_oraciones:
            oracion = oracion_por_nombre(nombre_oracion)
            print(f"\n  --- {nombre_oracion} ---")
            for disambiguador, esperado in oracion["variantes"].items():
                frase = oracion["plantilla"].format(d=disambiguador)
                print(f"\n  Variante '{disambiguador}' (esperado: {esperado}):")
                mostrar_fila_atencion(
                    modelo, frase, capa, cabeza, disambiguador, oracion["candidatos"]
                )


if __name__ == "__main__":
    main()
