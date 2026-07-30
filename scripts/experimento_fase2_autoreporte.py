"""Fase 2 (rediseñada tras el hallazgo causal de la Fase 1) - auto-reporte de Pythia 70M.

La Fase 1 encontro que Pythia 70M, capa 0, cabeza 2, empuja causalmente la
prediccion hacia el sustantivo mencionado primero en la oracion (el sujeto),
sin importar el adjetivo desambiguador (ver ablation_cabeza_candidata.py).
Esta fase pregunta: cuando el modelo "acierta" por esta razon posicional (o
falla por la misma razon), ¿su auto-reporte en palabras lo admite de alguna
forma, o siempre inventa una justificacion semantica plausible sobre el
tamaño/peso/atributo del objeto, como si hubiera razonado sobre el
significado?

Se usan las mismas 3 oraciones x 2 variantes (6 casos) de la ablation:
trophy_suitcase, car_truck, man_couch. Para cada caso:

1. Se determina la eleccion real del modelo (sin ablation), via
   `behavior.compare_candidates` -- no se le fuerza el candidato correcto:
   se le pide que explique LO QUE ELIGIO, sea o no la respuesta esperada.
2. Se genera el auto-reporte continuando "... It refers to the {eleccion}
   because" (generacion determinista, sin fine-tuning).
3. Se documenta si la eleccion coincide con el sujeto/primer sustantivo de
   la oracion (consistente con el sesgo posicional hallado en la Fase 1), y
   si coincide con la respuesta esperada por un humano.
4. Extra: se repite la eleccion y el auto-reporte con la cabeza (capa 0,
   cabeza 2) apagada, para ver si el auto-reporte cambia de tono/contenido
   aunque el output en si cambie.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.behavior import compare_candidates
from src.models import load_model
from src.self_report import generate_self_report
from src.winograd_sentences import ORACIONES

NOMBRE_MODELO = "pythia-70m"
CAPA = 0
CABEZA = 2
ORACIONES_A_PROBAR = ["trophy_suitcase", "car_truck", "man_couch"]


def oracion_por_nombre(nombre: str) -> dict:
    for o in ORACIONES:
        if o["nombre"] == nombre:
            return o
    raise KeyError(nombre)


def hook_zero_ablation(z, hook):
    z[:, :, CABEZA, :] = 0.0
    return z


def analizar_caso(modelo, oracion: dict, disambiguador: str, esperado: str, fwd_hooks=None) -> dict:
    candidatos = oracion["candidatos"]
    sujeto = candidatos[0]  # primer sustantivo mencionado en la plantilla
    frase = oracion["plantilla"].format(d=disambiguador)
    prefijo = frase + " It refers to the"

    log_probs = compare_candidates(modelo, prefijo, candidatos, fwd_hooks=fwd_hooks)
    eleccion = max(log_probs, key=log_probs.get)

    auto_reporte = generate_self_report(modelo, frase, eleccion, fwd_hooks=fwd_hooks)

    return {
        "frase": frase,
        "esperado": esperado,
        "sujeto": sujeto,
        "eleccion": eleccion,
        "acerto": eleccion == esperado,
        "eligio_sujeto": eleccion == sujeto,
        "auto_reporte": auto_reporte,
    }


def etiqueta_sesgo(caso: dict) -> str:
    if caso["acerto"] and caso["eligio_sujeto"]:
        return "acerto por sesgo (sujeto == esperado)"
    if not caso["acerto"] and caso["eligio_sujeto"]:
        return "fallo por sesgo (eligio el sujeto, pero no era el esperado)"
    if caso["acerto"] and not caso["eligio_sujeto"]:
        return "acerto sin seguir el sesgo (eligio el objeto, y era el esperado)"
    return "fallo sin seguir el sesgo (eligio el objeto, y no era el esperado)"


def main() -> None:
    modelo = load_model(NOMBRE_MODELO)
    nombre_hook = f"blocks.{CAPA}.attn.hook_z"
    fwd_hooks = [(nombre_hook, hook_zero_ablation)]

    for nombre in ORACIONES_A_PROBAR:
        oracion = oracion_por_nombre(nombre)
        print(f"\n{'=' * 70}\n=== {nombre} ===")

        for disambiguador, esperado in oracion["variantes"].items():
            print(f"\n  --- Variante '{disambiguador}' (esperado: {esperado}) ---")

            caso = analizar_caso(modelo, oracion, disambiguador, esperado)
            print(f"  Frase: {caso['frase']}")
            print(
                f"  Eleccion del modelo: '{caso['eleccion']}' | sujeto de la oracion: '{caso['sujeto']}' "
                f"| acerto: {caso['acerto']} | eligio el sujeto: {caso['eligio_sujeto']}"
            )
            print(f"  -> {etiqueta_sesgo(caso)}")
            print(f"  Auto-reporte: \"It refers to the {caso['eleccion']} because{caso['auto_reporte']}\"")

            caso_abl = analizar_caso(modelo, oracion, disambiguador, esperado, fwd_hooks=fwd_hooks)
            print(f"\n  [Con la cabeza (0,2) apagada]")
            print(f"  Eleccion del modelo: '{caso_abl['eleccion']}' | acerto: {caso_abl['acerto']}")
            print(
                f"  Auto-reporte: \"It refers to the {caso_abl['eleccion']} because"
                f"{caso_abl['auto_reporte']}\""
            )


if __name__ == "__main__":
    main()
