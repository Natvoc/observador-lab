"""Visualizacion de patrones de atencion con circuitsvis.

Se usa `local_src` (JS embebido dentro del propio paquete) en vez de `str(widget)`
(que apunta a un CDN), para que el HTML generado funcione sin conexion a
internet, en linea con el requisito de que observador-lab corra 100% local.
"""

import plotly.graph_objects as go
import torch
from circuitsvis.attention import attention_heads


def build_attention_html(tokens: list[str], patron_capa: torch.Tensor) -> str:
    """HTML autocontenido con un selector de cabeza para una capa dada.

    `patron_capa` tiene forma [n_heads, seq_len, seq_len] (salida de
    `attention.get_attention_patterns(...).patterns[capa]`).
    """
    widget = attention_heads(attention=patron_capa, tokens=tokens)
    return widget.local_src


def plot_attention_heatmap(tokens: list[str], patron_cabeza: torch.Tensor) -> go.Figure:
    """Heatmap de una sola cabeza: `patron_cabeza` tiene forma [seq_len, seq_len].

    A diferencia de `build_attention_html` (que embebe un selector de cabeza
    dentro del HTML, sin que Python sepa cual esta eligiendo la persona
    usuaria), esta version renderiza una sola cabeza puntual elegida desde
    Gradio -- necesario para poder resaltar cuando esa cabeza especifica es
    la que tiene mecanismo verificado por ablation.
    """
    etiquetas = [f"{i}:{tok.strip() or tok!r}" for i, tok in enumerate(tokens)]
    figura = go.Figure(
        data=go.Heatmap(
            z=patron_cabeza.tolist(),
            x=etiquetas,
            y=etiquetas,
            colorscale="Blues",
            zmin=0,
            zmax=1,
        )
    )
    figura.update_layout(
        xaxis_title="Token de origen (atendido)",
        yaxis_title="Token de destino (que atiende)",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=450,
    )
    return figura
