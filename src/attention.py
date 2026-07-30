"""Extraccion de pesos de atencion via hooks de transformer_lens.

Todas las funciones son parametrizables por prompt: no hay ningun ejemplo
hardcodeado aca, para que puedan usarse con cualquier oracion en ingles.
"""

from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer


@dataclass
class AttentionResult:
    """Resultado de correr un prompt y capturar sus patrones de atencion."""

    tokens: list[str]
    # patterns[capa] tiene forma [n_heads, seq_len, seq_len]
    patterns: list[torch.Tensor]


def get_attention_patterns(modelo: HookedTransformer, prompt: str) -> AttentionResult:
    """Corre `prompt` sobre `modelo` y devuelve los patrones de atencion de todas las capas."""
    tokens = modelo.to_tokens(prompt)
    _, cache = modelo.run_with_cache(tokens, return_type=None)
    patterns = [cache["pattern", capa][0] for capa in range(modelo.cfg.n_layers)]
    str_tokens = modelo.to_str_tokens(prompt)
    return AttentionResult(tokens=str_tokens, patterns=patterns)


def find_token_span(tokens: list[str], palabra: str) -> tuple[int, int] | None:
    """Rango [inicio, fin) de tokens consecutivos cuyo texto concatenado coincide con `palabra`.

    Los tokenizers BPE de estos modelos a veces parten una palabra en mas de un
    token (ej. "councilmen" -> "council" + "men"), y los tokens de GPT-2/Pythia
    llevan un espacio inicial (" trophy"). Por eso se busca, letra a letra, la
    primera secuencia contigua de tokens (recortados) cuya concatenacion sea
    igual a `palabra`, en vez de exigir que sea un unico token exacto.

    Devuelve None si no hay ninguna coincidencia.
    """
    objetivo = palabra.replace(" ", "").lower()
    n = len(tokens)
    for inicio in range(n):
        acumulado = ""
        for fin in range(inicio, n):
            acumulado += tokens[fin].strip()
            if acumulado.lower() == objetivo:
                return (inicio, fin + 1)
            if len(acumulado) > len(objetivo):
                break
    return None


def _resolver_indices(
    resultado: AttentionResult, desde_palabra: str, candidatos: list[str]
) -> tuple[int, dict[str, tuple[int, int]]]:
    """Resuelve el indice del token `desde_palabra` y los spans de los candidatos.

    Lanza ValueError si `desde_palabra` o algun candidato no aparece en el
    prompt como secuencia de tokens.
    """
    span_desde = find_token_span(resultado.tokens, desde_palabra)
    if span_desde is None:
        raise ValueError(f"No se encontro '{desde_palabra}' en el prompt")
    idx_desde = span_desde[1] - 1  # ultimo token del span (la mascara es causal)

    spans_candidatos: dict[str, tuple[int, int]] = {}
    for candidato in candidatos:
        span = find_token_span(resultado.tokens, candidato)
        if span is None:
            raise ValueError(f"No se encontro '{candidato}' en el prompt")
        spans_candidatos[candidato] = span

    return idx_desde, spans_candidatos


def attention_to_candidates(
    resultado: AttentionResult, desde_palabra: str, candidatos: list[str]
) -> dict[str, list[float]]:
    """Cuanto atiende `desde_palabra` a cada uno de los `candidatos`, por capa.

    Para cada capa, promedia la atencion sobre todas las cabezas. Si un
    candidato ocupa varios tokens, se suma la atencion recibida por todos
    ellos. Ver `_resolver_indices` para como se ubican los tokens.
    """
    idx_desde, spans_candidatos = _resolver_indices(resultado, desde_palabra, candidatos)

    resumen: dict[str, list[float]] = {c: [] for c in candidatos}
    for patron_capa in resultado.patterns:
        atencion_promedio_cabezas = patron_capa[:, idx_desde, :].mean(dim=0)
        for candidato, (inicio, fin) in spans_candidatos.items():
            resumen[candidato].append(atencion_promedio_cabezas[inicio:fin].sum().item())
    return resumen


def attention_to_candidates_per_head(
    resultado: AttentionResult, desde_palabra: str, candidatos: list[str]
) -> dict[str, torch.Tensor]:
    """Igual que `attention_to_candidates`, pero sin promediar entre cabezas.

    Devuelve, para cada candidato, un tensor [n_layers, n_heads] con la
    atencion que le presta cada cabeza individual. Sirve para buscar si una
    cabeza puntual distingue el contexto aunque el promedio no lo muestre.
    """
    idx_desde, spans_candidatos = _resolver_indices(resultado, desde_palabra, candidatos)

    n_layers = len(resultado.patterns)
    n_heads = resultado.patterns[0].shape[0]
    resumen = {c: torch.zeros(n_layers, n_heads) for c in candidatos}
    for capa, patron_capa in enumerate(resultado.patterns):
        atencion_desde = patron_capa[:, idx_desde, :]  # [n_heads, seq]
        for candidato, (inicio, fin) in spans_candidatos.items():
            resumen[candidato][capa] = atencion_desde[:, inicio:fin].sum(dim=-1)
    return resumen


def cabezas_que_resuelven(modelo: HookedTransformer, oracion: dict) -> list[tuple[int, int]]:
    """Lista de (capa, cabeza) que distinguen correctamente las dos variantes de `oracion`.

    Una cabeza "resuelve" una oracion si, en AMBAS variantes, le da mas
    atencion (desde la palabra desambiguadora) al sustantivo correcto que al
    incorrecto. Una cabeza con un sesgo fijo (que no reacciona al contexto)
    nunca puede resolver ambas variantes a la vez, porque cada variante
    espera un sustantivo distinto -- por eso esto es mas exigente que "le
    achunto en una variante sola".

    `oracion` sigue el formato de `src/winograd_sentences.py`: un dict con
    `plantilla` (con un slot `{d}`), `candidatos` (lista de 2 sustantivos) y
    `variantes` (dict palabra desambiguadora -> sustantivo esperado).
    """
    candidatos = oracion["candidatos"]
    variantes = list(oracion["variantes"].items())

    por_variante = []
    for disambiguador, _ in variantes:
        frase = oracion["plantilla"].format(d=disambiguador)
        resultado = get_attention_patterns(modelo, frase)
        atencion = attention_to_candidates_per_head(
            resultado, desde_palabra=disambiguador, candidatos=candidatos
        )
        por_variante.append(atencion)

    n_layers, n_heads = por_variante[0][candidatos[0]].shape
    resolviendo = []
    for capa in range(n_layers):
        for cabeza in range(n_heads):
            ok_en_todas = True
            for (_, esperado), atencion_variante in zip(variantes, por_variante):
                otro = candidatos[0] if esperado == candidatos[1] else candidatos[1]
                if not (
                    atencion_variante[esperado][capa, cabeza]
                    > atencion_variante[otro][capa, cabeza]
                ):
                    ok_en_todas = False
                    break
            if ok_en_todas:
                resolviendo.append((capa, cabeza))
    return resolviendo
