"""
Visor de series temporales y GAF para archivos .npz del proyecto.

Casos soportados:
1) Grabacion publica procesada:
     datos/openneuro_procesado/sub-*.npz
   - extrae una ventana de raw_fp1
   - calcula y muestra su GAF

2) Dataset procesado Rama A (raw):
     datos/procesados/rama_a_*/datos.npz
   - toma una muestra X_{split}[index]
   - la muestra como serie temporal
   - calcula y muestra su GAF

3) Dataset procesado Rama B (GAF):
     datos/procesados/rama_b_*/datos.npz
   - muestra el GAF almacenado
   - opcionalmente, con --serie-npz, carga la serie correspondiente y la muestra al lado
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocesamiento.gaf import generar_gaf


def _cargar_npz(ruta: str):
    return np.load(ruta, allow_pickle=True)


def _label_to_text(y_val) -> str:
    return "parkinson" if int(y_val) == 1 else "sano"


def _inferir_desde_publico(datos, inicio_sec: float, duracion_sec: float, gaf_size: int, gaf_method: str):
    fs = int(datos["fs"])
    raw = datos["raw_fp1"].astype(np.float32)
    inicio = int(inicio_sec * fs)
    n = int(duracion_sec * fs)
    fin = min(len(raw), inicio + n)
    serie = raw[inicio:fin]
    if len(serie) < 2:
        raise ValueError("La ventana extraida de raw_fp1 es demasiado corta.")
    gaf = generar_gaf(serie, tamano=gaf_size, metodo=gaf_method)
    titulo = f"{str(datos['sujeto'])}_{str(datos['sesion'])} | {_label_to_text(datos['etiqueta'])}"
    return {
        "serie": serie,
        "gaf": gaf,
        "fs": fs,
        "titulo": titulo,
    }


def _inferir_desde_dataset_rama_a(datos, split: str, index: int, gaf_size: int, gaf_method: str):
    X = datos[f"X_{split}"]
    y = datos[f"y_{split}"]
    sujetos = datos[f"sujetos_{split}"]
    muestra = X[index]
    if muestra.ndim != 1:
        raise ValueError("El dataset seleccionado no contiene series 1D en esa clave.")
    gaf = generar_gaf(muestra, tamano=gaf_size, metodo=gaf_method)
    return {
        "serie": muestra.astype(np.float32),
        "gaf": gaf,
        "fs": None,
        "titulo": f"{sujetos[index]} | {_label_to_text(y[index])} | split={split} | idx={index}",
    }


def _inferir_desde_dataset_rama_b(
    datos,
    split: str,
    index: int,
    serie_npz: str | None,
):
    X = datos[f"X_{split}"]
    y = datos[f"y_{split}"]
    sujetos = datos[f"sujetos_{split}"]
    imagen = X[index].astype(np.float32)
    serie = None

    if serie_npz:
        datos_serie = _cargar_npz(serie_npz)
        serie = datos_serie[f"X_{split}"][index].astype(np.float32)

    return {
        "serie": serie,
        "gaf": imagen,
        "fs": None,
        "titulo": f"{sujetos[index]} | {_label_to_text(y[index])} | split={split} | idx={index}",
    }


def _detectar_y_preparar(args):
    datos = _cargar_npz(args.npz)
    claves = set(datos.files)

    if "raw_fp1" in claves:
        return _inferir_desde_publico(
            datos,
            inicio_sec=args.inicio_sec,
            duracion_sec=args.duracion_sec,
            gaf_size=args.gaf_size,
            gaf_method=args.gaf_method,
        )

    if f"X_{args.split}" in claves:
        muestra = datos[f"X_{args.split}"][args.index]
        if muestra.ndim == 1:
            return _inferir_desde_dataset_rama_a(
                datos,
                split=args.split,
                index=args.index,
                gaf_size=args.gaf_size,
                gaf_method=args.gaf_method,
            )
        if muestra.ndim == 2:
            return _inferir_desde_dataset_rama_b(
                datos,
                split=args.split,
                index=args.index,
                serie_npz=args.serie_npz,
            )

    raise ValueError("No se pudo inferir el tipo de .npz soportado.")


def _graficar(payload, guardar: str | None = None):
    tiene_serie = payload["serie"] is not None

    if tiene_serie:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        x = payload["serie"]
        axes[0].plot(x, linewidth=1.2, color="#1f77b4")
        axes[0].set_title("Serie temporal")
        axes[0].set_xlabel("Muestra")
        axes[0].set_ylabel("Amplitud")
        axes[0].grid(True, alpha=0.25)
        img_ax = axes[1]
    else:
        fig, img_ax = plt.subplots(1, 1, figsize=(6, 5))

    im = img_ax.imshow(payload["gaf"], cmap="RdBu_r", aspect="auto", origin="lower")
    img_ax.set_title("GAF")
    img_ax.set_xlabel("Tiempo j")
    img_ax.set_ylabel("Tiempo i")
    fig.colorbar(im, ax=img_ax, fraction=0.046, pad=0.04)

    fig.suptitle(payload["titulo"])
    fig.tight_layout()

    if guardar:
        ruta = Path(guardar)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=150, bbox_inches="tight")
        print(f"Figura guardada en: {ruta}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualizar serie temporal y GAF desde .npz")
    parser.add_argument("npz", help="Ruta al archivo .npz")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--index", type=int, default=0, help="Indice de muestra para datasets procesados")
    parser.add_argument("--serie-npz", default="", help="Ruta opcional a un datos.npz de Rama A para emparejar con Rama B")
    parser.add_argument("--inicio-sec", type=float, default=0.0, help="Inicio en segundos para grabaciones publicas")
    parser.add_argument("--duracion-sec", type=float, default=2.0, help="Duracion en segundos para grabaciones publicas")
    parser.add_argument("--gaf-size", type=int, default=128, help="Tamano del GAF si se calcula on-the-fly")
    parser.add_argument("--gaf-method", choices=["summation", "difference"], default="summation")
    parser.add_argument("--guardar", default="", help="Ruta opcional para guardar la figura")
    args = parser.parse_args()

    payload = _detectar_y_preparar(args)
    _graficar(payload, guardar=args.guardar or None)


if __name__ == "__main__":
    main()
