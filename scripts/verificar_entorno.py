"""
Verificacion de entorno (Fase 0) para observador-lab.

Carga cada modelo soportado con transformer_lens, corre un forward pass
simple sobre una oracion de prueba, captura activaciones via run_with_cache,
y reporta tiempos. No entrena ni hace fine-tuning: solo confirma que el
entorno esta listo para los experimentos de las fases siguientes.
"""

import time

import torch
from transformer_lens import HookedTransformer

MODELOS = ["gpt2", "pythia-70m", "pythia-160m"]

PROMPT_DE_PRUEBA = (
    "The trophy doesn't fit in the suitcase because it is too big."
)


def verificar_modelo(nombre_modelo: str) -> None:
    print(f"\n=== {nombre_modelo} ===")

    inicio_carga = time.time()
    modelo = HookedTransformer.from_pretrained(nombre_modelo, device="cpu")
    segundos_carga = time.time() - inicio_carga
    print(f"Carga: {segundos_carga:.1f}s")

    print(f"Capas: {modelo.cfg.n_layers} | Cabezas: {modelo.cfg.n_heads} | "
          f"d_model: {modelo.cfg.d_model}")

    inicio_forward = time.time()
    tokens = modelo.to_tokens(PROMPT_DE_PRUEBA)
    logits, cache = modelo.run_with_cache(tokens)
    segundos_forward = time.time() - inicio_forward
    print(f"Forward pass + cache: {segundos_forward:.2f}s")
    print(f"Tokens de entrada: {tokens.shape}")
    print(f"Logits de salida: {logits.shape}")

    patron_atencion = cache["pattern", 0]
    print(f"Patron de atencion (capa 0): {patron_atencion.shape}")

    del modelo, cache, logits
    print(f"OK: {nombre_modelo} carga y corre correctamente en CPU.")


def main() -> None:
    assert not torch.cuda.is_available() or True  # informativo, no bloqueante
    print(f"CUDA disponible: {torch.cuda.is_available()} (se fuerza CPU igual)")

    for nombre_modelo in MODELOS:
        verificar_modelo(nombre_modelo)

    print("\nTodos los modelos cargaron y corrieron sin errores.")


if __name__ == "__main__":
    main()
