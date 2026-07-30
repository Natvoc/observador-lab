"""Visualizacion de patrones de atencion con circuitsvis.

Se usa `local_src` (JS embebido dentro del propio paquete) en vez de `str(widget)`
(que apunta a un CDN), para que el HTML generado funcione sin conexion a
internet, en linea con el requisito de que observador-lab corra 100% local.
"""

import torch
from circuitsvis.attention import attention_heads


def build_attention_html(tokens: list[str], patron_capa: torch.Tensor) -> str:
    """HTML autocontenido con un selector de cabeza para una capa dada.

    `patron_capa` tiene forma [n_heads, seq_len, seq_len] (salida de
    `attention.get_attention_patterns(...).patterns[capa]`).
    """
    widget = attention_heads(attention=patron_capa, tokens=tokens)
    return widget.local_src
