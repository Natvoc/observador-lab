"""Fase 1 - atencion + verificacion comportamental sobre varias oraciones tipo Winograd.

Este script hace dos cosas, deliberadamente separadas:

1. Genera el mapa de atencion navegable por capa/cabeza (circuitsvis) para el
   ejemplo insignia (trofeo/maleta), en sus dos variantes, en los tres
   modelos. Este es el criterio de exito original de la Fase 1.

2. Corre una verificacion comportamental (independiente de la atencion) y una
   medicion de atencion cuantitativa sobre 8 oraciones tipo Winograd (no solo
   trofeo/maleta), para poder decir con una muestra real si estos modelos
   resuelven la tarea en absoluto, y si la atencion coincide con esa
   resolucion. No se generan visualizaciones HTML para las 8 oraciones: cada
   una pesa varios MB y el objetivo aca es el dato cuantitativo, no acumular
   archivos.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attention import attention_to_candidates, get_attention_patterns
from src.behavior import compare_candidates
from src.models import MODELOS_DISPONIBLES, load_model
from src.visualization import build_attention_html
from src.winograd_sentences import ORACIONES

DIRECTORIO_SALIDA = Path(__file__).resolve().parent.parent / "outputs" / "fase1"

# Prompt insignia, con visualizacion HTML completa.
PROMPTS_INSIGNIA = {
    "big": "The trophy doesn't fit in the suitcase because it is too big.",
    "small": "The trophy doesn't fit in the suitcase because it is too small.",
}

# Prompt de sondeo para la verificacion comportamental: se le pide al modelo
# que continue con esto y se compara la probabilidad de cada candidato como
# siguiente palabra. Mismo formato que se usara para el auto-reporte en la
# Fase 2, pero aca solo se mide el logit, no se genera texto libre.
SONDEO_COMPORTAMENTAL = " It refers to the"


def guardar_html_por_capas(nombre_modelo: str, variante: str, tokens: list[str], patterns) -> Path:
    partes = [
        "<html><head><meta charset='utf-8'>"
        f"<title>Atencion - {nombre_modelo} - {variante}</title></head><body>"
        f"<h1>{nombre_modelo} - prompt '{variante}'</h1>"
    ]
    for capa, patron_capa in enumerate(patterns):
        partes.append(f"<h2>Capa {capa}</h2>")
        partes.append(build_attention_html(tokens, patron_capa))
    partes.append("</body></html>")

    DIRECTORIO_SALIDA.mkdir(parents=True, exist_ok=True)
    nombre_archivo = f"{nombre_modelo}_{variante}.html".replace(" ", "_")
    archivo = DIRECTORIO_SALIDA / nombre_archivo
    archivo.write_text("\n".join(partes), encoding="utf-8")
    return archivo


def correr_prompt_insignia(nombre_display: str, modelo) -> None:
    for variante, prompt in PROMPTS_INSIGNIA.items():
        resultado = get_attention_patterns(modelo, prompt)
        archivo = guardar_html_por_capas(
            nombre_display, variante, resultado.tokens, resultado.patterns
        )
        print(f"    Visualizacion '{variante}' guardada en: {archivo}")


def evaluar_oracion(modelo, oracion: dict) -> list[dict]:
    filas = []
    for disambiguador, esperado in oracion["variantes"].items():
        frase = oracion["plantilla"].format(d=disambiguador)

        resultado = get_attention_patterns(modelo, frase)
        atencion = attention_to_candidates(
            resultado, desde_palabra=disambiguador, candidatos=oracion["candidatos"]
        )
        promedio_atencion = {c: sum(v) / len(v) for c, v in atencion.items()}
        ganador_atencion = max(promedio_atencion, key=promedio_atencion.get)

        log_probs = compare_candidates(
            modelo, frase + SONDEO_COMPORTAMENTAL, oracion["candidatos"]
        )
        ganador_comportamiento = max(log_probs, key=log_probs.get)

        filas.append(
            {
                "oracion": oracion["nombre"],
                "disambiguador": disambiguador,
                "esperado": esperado,
                "atencion_coincide": ganador_atencion == esperado,
                "comportamiento_coincide": ganador_comportamiento == esperado,
            }
        )
    return filas


def main() -> None:
    todas_las_filas = []

    for nombre_display, nombre_tl in MODELOS_DISPONIBLES.items():
        print(f"\n=== {nombre_display} ===")
        modelo = load_model(nombre_tl)

        correr_prompt_insignia(nombre_display, modelo)

        n_ok_atencion_modelo = 0
        n_ok_comportamiento_modelo = 0
        n_casos_modelo = 0

        for oracion in ORACIONES:
            for fila in evaluar_oracion(modelo, oracion):
                fila["modelo"] = nombre_display
                todas_las_filas.append(fila)
                n_casos_modelo += 1
                n_ok_atencion_modelo += fila["atencion_coincide"]
                n_ok_comportamiento_modelo += fila["comportamiento_coincide"]
                print(
                    f"    [{fila['oracion']:16s} / {fila['disambiguador']:6s}] "
                    f"esperado={fila['esperado']:10s} "
                    f"atencion_ok={'SI' if fila['atencion_coincide'] else 'NO':2s} "
                    f"comportamiento_ok={'SI' if fila['comportamiento_coincide'] else 'NO'}"
                )

        print(
            f"  Subtotal {nombre_display}: atencion {n_ok_atencion_modelo}/{n_casos_modelo}, "
            f"comportamiento {n_ok_comportamiento_modelo}/{n_casos_modelo}"
        )

    n = len(todas_las_filas)
    n_atencion_ok = sum(f["atencion_coincide"] for f in todas_las_filas)
    n_comportamiento_ok = sum(f["comportamiento_coincide"] for f in todas_las_filas)

    print(
        f"\n=== Resumen total ({n} casos: {len(ORACIONES)} oraciones x 2 variantes x "
        f"{len(MODELOS_DISPONIBLES)} modelos) ==="
    )
    print(
        f"Atencion coincide con el sustantivo esperado: {n_atencion_ok}/{n} "
        f"({100 * n_atencion_ok / n:.0f}%)"
    )
    print(
        f"Comportamiento (logit) coincide con el sustantivo esperado: {n_comportamiento_ok}/{n} "
        f"({100 * n_comportamiento_ok / n:.0f}%)"
    )
    print("(nivel de azar para una eleccion binaria: 50%)")


if __name__ == "__main__":
    main()
