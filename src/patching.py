"""Activation patching cruzado sobre el residual stream.

Complementa la ablation de cabezas (Fase 1): en vez de apagar un componente,
transplanta la activacion completa del residual stream (`resid_post`) en la
posicion de la palabra desambiguadora, desde una variante hacia la otra,
capa por capa. Esto prueba si la informacion semantica del adjetivo llega a
tener efecto causal en alguna capa, antes de perderse frente al sesgo
posicional identificado por ablation.
"""

import torch
from transformer_lens import HookedTransformer

from src.attention import find_token_span
from src.behavior import continuation_log_prob


def _hook_de_parche(valor_donante: torch.Tensor, posicion: int):
    def hook_fn(resid, hook):
        resid[:, posicion, :] = valor_donante
        return resid

    return hook_fn


def _hook_de_parche_cabeza(valor_donante: torch.Tensor, posicion: int, cabeza: int):
    def hook_fn(z, hook):
        z[:, posicion, cabeza, :] = valor_donante
        return z

    return hook_fn


def _margen(
    modelo: HookedTransformer,
    frase_objetivo: str,
    candidatos: list[str],
    esperado_objetivo: str,
    fwd_hooks: list,
) -> float:
    otro = candidatos[0] if esperado_objetivo == candidatos[1] else candidatos[1]
    prefijo_objetivo = frase_objetivo + " It refers to the"
    lp_correcto = continuation_log_prob(
        modelo, prefijo_objetivo, " " + esperado_objetivo, fwd_hooks=fwd_hooks
    )
    lp_incorrecto = continuation_log_prob(modelo, prefijo_objetivo, " " + otro, fwd_hooks=fwd_hooks)
    return lp_correcto - lp_incorrecto


def margen_sin_parche(
    modelo: HookedTransformer, frase: str, candidatos: list[str], esperado: str
) -> float:
    """Margen de logit (esperado vs el otro candidato) sin ninguna intervencion."""
    return _margen(modelo, frase, candidatos, esperado, fwd_hooks=[])


def patch_resid_por_capa(
    modelo: HookedTransformer,
    frase_donante: str,
    palabra_donante: str,
    frase_objetivo: str,
    palabra_objetivo: str,
    candidatos: list[str],
    esperado_objetivo: str,
) -> dict[int, float]:
    """Transplanta `resid_post` en la posicion de la palabra desambiguadora,
    capa por capa, desde `frase_donante` hacia `frase_objetivo`.

    Para cada capa, corre `frase_objetivo` (mas el sondeo "It refers to the")
    reemplazando su residual stream en la posicion de `palabra_objetivo` por
    el de `frase_donante` en la posicion de `palabra_donante` (misma capa), y
    devuelve el margen de logit resultante: log-prob de `esperado_objetivo`
    menos log-prob del otro candidato.

    El residual stream en una posicion dada, en un modelo causal, no depende
    de los tokens que vienen despues -- por eso alcanza con correr
    `frase_donante` sola (sin el sondeo) para obtener su activacion donante.
    """
    tokens_donante = modelo.to_tokens(frase_donante)
    idx_donante = find_token_span(modelo.to_str_tokens(frase_donante), palabra_donante)[1] - 1
    _, cache_donante = modelo.run_with_cache(tokens_donante, return_type=None)

    idx_objetivo = find_token_span(modelo.to_str_tokens(frase_objetivo), palabra_objetivo)[1] - 1

    resultado: dict[int, float] = {}
    for capa in range(modelo.cfg.n_layers):
        valor_donante = cache_donante["resid_post", capa][0, idx_donante, :]
        fwd_hooks = [
            (f"blocks.{capa}.hook_resid_post", _hook_de_parche(valor_donante, idx_objetivo))
        ]
        resultado[capa] = _margen(modelo, frase_objetivo, candidatos, esperado_objetivo, fwd_hooks)

    return resultado


def patch_head_output(
    modelo: HookedTransformer,
    frase_donante: str,
    palabra_donante: str,
    frase_objetivo: str,
    palabra_objetivo: str,
    candidatos: list[str],
    esperado_objetivo: str,
    capa: int,
    cabeza: int,
) -> float:
    """Parcha SOLO la salida (`hook_z`) de una cabeza puntual, dejando intacto
    el resto de la capa (las demas cabezas y el MLP).

    A diferencia de `patch_resid_por_capa` (que reemplaza todo el residual
    stream en esa posicion), esto aisla cuanto del efecto de parchar la capa
    entera se explica por esta cabeza en particular: el resto de las cabezas
    y el MLP de esa capa siguen viendo la mezcla real de `frase_objetivo` (no
    la del donante) y recalculan su salida en base a eso con normalidad.
    """
    tokens_donante = modelo.to_tokens(frase_donante)
    idx_donante = find_token_span(modelo.to_str_tokens(frase_donante), palabra_donante)[1] - 1
    _, cache_donante = modelo.run_with_cache(tokens_donante, return_type=None)
    valor_donante = cache_donante["z", capa][0, idx_donante, cabeza, :]

    idx_objetivo = find_token_span(modelo.to_str_tokens(frase_objetivo), palabra_objetivo)[1] - 1

    fwd_hooks = [
        (
            f"blocks.{capa}.attn.hook_z",
            _hook_de_parche_cabeza(valor_donante, idx_objetivo, cabeza),
        )
    ]
    return _margen(modelo, frase_objetivo, candidatos, esperado_objetivo, fwd_hooks)


def patch_mlp_output(
    modelo: HookedTransformer,
    frase_donante: str,
    palabra_donante: str,
    frase_objetivo: str,
    palabra_objetivo: str,
    candidatos: list[str],
    esperado_objetivo: str,
    capa: int,
) -> float:
    """Parcha SOLO la salida del MLP (`hook_mlp_out`) de una capa, dejando
    intacto el resto (todas las cabezas de atencion de esa capa).

    Mismo criterio que `patch_head_output`, pero aislando el MLP en vez de
    una cabeza puntual.
    """
    tokens_donante = modelo.to_tokens(frase_donante)
    idx_donante = find_token_span(modelo.to_str_tokens(frase_donante), palabra_donante)[1] - 1
    _, cache_donante = modelo.run_with_cache(tokens_donante, return_type=None)
    valor_donante = cache_donante["mlp_out", capa][0, idx_donante, :]

    idx_objetivo = find_token_span(modelo.to_str_tokens(frase_objetivo), palabra_objetivo)[1] - 1

    fwd_hooks = [
        (f"blocks.{capa}.hook_mlp_out", _hook_de_parche(valor_donante, idx_objetivo))
    ]
    return _margen(modelo, frase_objetivo, candidatos, esperado_objetivo, fwd_hooks)
