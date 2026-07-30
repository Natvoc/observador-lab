"""Carga y cacheo de los modelos soportados por observador-lab."""

from functools import lru_cache

from transformer_lens import HookedTransformer

# Nombre para mostrar en la interfaz -> nombre que espera transformer_lens.
MODELOS_DISPONIBLES = {
    "GPT-2 small": "gpt2",
    "Pythia 70M": "pythia-70m",
    "Pythia 160M": "pythia-160m",
}

# Capas y cabezas de cada arquitectura (valores publicos y fijos de estos
# checkpoints). Se hardcodean para poder poblar los selectores de la interfaz
# sin tener que cargar el modelo entero de antemano.
CONFIGURACION_MODELO = {
    "gpt2": {"n_layers": 12, "n_heads": 12},
    "pythia-70m": {"n_layers": 6, "n_heads": 8},
    "pythia-160m": {"n_layers": 12, "n_heads": 12},
}


@lru_cache(maxsize=None)
def load_model(nombre_transformer_lens: str) -> HookedTransformer:
    """Carga (con cache) un modelo por su nombre de transformer_lens, en CPU."""
    modelo = HookedTransformer.from_pretrained(nombre_transformer_lens, device="cpu")
    modelo.eval()
    return modelo
