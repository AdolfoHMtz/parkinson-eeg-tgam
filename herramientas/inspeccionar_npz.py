"""
Inspeccion rapida de archivos .npz del proyecto.

Uso:
    python inspeccionar_npz.py ruta/al/archivo.npz
"""

import sys
from pathlib import Path

import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Uso: python inspeccionar_npz.py ruta/al/archivo.npz")
        sys.exit(1)

    ruta = Path(sys.argv[1])
    if not ruta.exists():
        print(f"No existe: {ruta}")
        sys.exit(1)

    datos = np.load(ruta, allow_pickle=True)
    print(f"Archivo: {ruta}")
    print(f"Claves: {datos.files}\n")

    for clave in datos.files:
        valor = datos[clave]
        shape = getattr(valor, "shape", None)
        dtype = getattr(valor, "dtype", None)
        print(f"- {clave}")
        print(f"  shape: {shape}")
        print(f"  dtype: {dtype}")
        if isinstance(valor, np.ndarray) and valor.size > 0 and np.issubdtype(valor.dtype, np.number):
            print(f"  min: {valor.min():.6f}")
            print(f"  max: {valor.max():.6f}")
            print(f"  mean: {valor.mean():.6f}")
        print()


if __name__ == "__main__":
    main()
