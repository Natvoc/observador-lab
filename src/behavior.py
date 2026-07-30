"""Verificacion comportamental, independiente de la atencion.

El mapa de atencion muestra un mecanismo (a que atendio el modelo), pero no
dice si el modelo resolvio la tarea. Este modulo mide otra cosa: la
probabilidad que el modelo le asigna a cada sustantivo candidato como
continuacion del prompt (via logits, sin generar texto libre ni hacer
fine-tuning). Es la pregunta previa a "¿la atencion explica la respuesta?":
primero hay que saber si hay una respuesta correcta que el modelo prefiera.
"""

import torch
from transformer_lens import HookedTransformer


def continuation_log_prob(
    modelo: HookedTransformer,
    prefijo: str,
    continuacion: str,
    fwd_hooks: list | None = None,
) -> float:
    """Log-probabilidad (suma sobre tokens) de que `continuacion` siga a `prefijo`.

    Tokeniza `prefijo + continuacion` en una sola pasada (teacher forcing) y
    suma la log-probabilidad de cada token de la continuacion, condicionado en
    todo lo anterior. Funciona aunque `continuacion` se parta en mas de un
    token BPE. Asume que la tokenizacion de `prefijo` no cambia al concatenar
    la continuacion, lo cual vale cuando `prefijo` termina en un limite de
    palabra (como en este experimento, donde siempre termina en espacio o
    punto).

    `fwd_hooks` (opcional) se pasa tal cual a `HookedTransformer.run_with_hooks`,
    para poder medir esta misma probabilidad bajo una intervencion (ej. apagar
    una cabeza especifica) y compararla contra la version sin intervenir.
    """
    tokens_prefijo = modelo.to_tokens(prefijo)
    tokens_completos = modelo.to_tokens(prefijo + continuacion)
    n_prefijo = tokens_prefijo.shape[1]

    logits = modelo.run_with_hooks(
        tokens_completos, return_type="logits", fwd_hooks=fwd_hooks or []
    )[0]
    log_probs = torch.log_softmax(logits, dim=-1)

    total = 0.0
    for pos in range(n_prefijo - 1, tokens_completos.shape[1] - 1):
        id_siguiente = tokens_completos[0, pos + 1].item()
        total += log_probs[pos, id_siguiente].item()
    return total


def compare_candidates(
    modelo: HookedTransformer,
    prefijo: str,
    candidatos: list[str],
    fwd_hooks: list | None = None,
) -> dict[str, float]:
    """Log-probabilidad de continuacion para cada candidato de `candidatos`.

    Antepone un espacio a cada candidato (convencion BPE de GPT-2/Pythia para
    una palabra nueva a mitad de oracion). Ver `continuation_log_prob` para
    `fwd_hooks`.
    """
    return {
        c: continuation_log_prob(modelo, prefijo, " " + c, fwd_hooks=fwd_hooks)
        for c in candidatos
    }
