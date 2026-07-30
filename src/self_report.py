"""Generacion del auto-reporte del modelo sobre su propia eleccion.

No es fine-tuning: es prompting simple. Se continua la oracion pidiendole al
modelo que explique por que "it" se refiere al sustantivo que el propio
modelo prefirio (no se le fuerza el candidato "correcto" segun el humano):
la pregunta de esta fase es que dice el modelo sobre SU eleccion, sea
correcta o no.
"""

from transformer_lens import HookedTransformer


def generate_self_report(
    modelo: HookedTransformer,
    frase: str,
    eleccion_del_modelo: str,
    max_tokens_nuevos: int = 25,
    fwd_hooks: list | None = None,
) -> str:
    """Continua `frase + ' It refers to the {eleccion} because'` y devuelve solo lo generado.

    Generacion determinista (`do_sample=False`) para que el resultado sea
    reproducible. `fwd_hooks` (opcional) se aplica durante toda la generacion
    (via `modelo.hooks`, no solo un forward pass), para poder comparar el
    auto-reporte con una cabeza especifica apagada.
    """
    prompt = f"{frase} It refers to the {eleccion_del_modelo} because"
    tokens = modelo.to_tokens(prompt)

    with modelo.hooks(fwd_hooks=fwd_hooks or []):
        salida = modelo.generate(
            tokens,
            max_new_tokens=max_tokens_nuevos,
            do_sample=False,
            stop_at_eos=True,
            verbose=False,
        )

    texto_completo = modelo.to_string(salida[0])
    return texto_completo[len(prompt):]
