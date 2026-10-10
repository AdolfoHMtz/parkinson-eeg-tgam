"""
Etiquetado y division train/val/test por sujeto.

Todas las ventanas de un mismo sujeto van al mismo conjunto (train, val o test).
Esto es critico para evitar data leakage: si ventanas del mismo sujeto aparecen
en train y test, el modelo aprende patrones individuales del sujeto en lugar de
patrones de la enfermedad.

IMPORTANTE: el identificador de sujeto debe ser el de la PERSONA, no el de la
grabacion. En OpenNeuro ds002778 cada paciente con Parkinson tiene dos sesiones
(on/off medicacion); si se usa "sujeto_sesion" como grupo, la misma persona
puede quedar en train y en test. Ver `id_paciente()`.

Etiquetas: sano = 0, Parkinson = 1.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


SANO = 0
PARKINSON = 1

PROPORCION_VAL = 0.15
PROPORCION_TEST = 0.15

# IDs de OpenNeuro ds002778: "pd11", "hc4", y variantes con sesion ("pd11_off")
_PATRON_OPENNEURO = re.compile(r"^(?:sub-)?((?:hc|pd)\d+)")


def id_paciente(sujeto: str) -> str:
    """
    Devuelve el identificador de la persona a partir de un ID de grabacion.

    "pd11_off" -> "pd11", "sub-hc4_ses-hc" -> "hc4". IDs que no siguen el
    formato de OpenNeuro (por ejemplo sesiones locales) se regresan tal cual.
    """
    m = _PATRON_OPENNEURO.match(str(sujeto))
    return m.group(1) if m else str(sujeto)


def crear_etiquetas(n: int, etiqueta: int) -> np.ndarray:
    """Vector de `n` etiquetas iguales (una por ventana)."""
    return np.full(n, etiqueta, dtype=np.int64)


def crear_ids_sujeto(n: int, sujeto: str) -> np.ndarray:
    """Vector de `n` IDs de sujeto iguales (uno por ventana)."""
    return np.array([str(sujeto)] * n)


def _balance(y: np.ndarray) -> dict:
    n = len(y)
    n_pd = int(np.sum(y == PARKINSON))
    return {
        "sano": int(np.sum(y == SANO)),
        "parkinson": n_pd,
        "proporcion_parkinson": float(n_pd / n) if n else 0.0,
    }


def calcular_info_division(division: dict) -> dict:
    """Resumen de una division: tamanos, sujetos y balance por conjunto."""
    info = {}
    for split in ("train", "val", "test"):
        info[f"n_{split}"] = int(len(division[f"y_{split}"]))
    for split in ("train", "val", "test"):
        info[f"sujetos_{split}"] = sorted(str(s) for s in np.unique(division[f"sujetos_{split}"]))
    for split in ("train", "val", "test"):
        info[f"n_sujetos_{split}"] = len(info[f"sujetos_{split}"])
    for split in ("train", "val", "test"):
        info[f"balance_{split}"] = _balance(division[f"y_{split}"])
    return info


def dividir_por_sujeto(
    X: np.ndarray,
    y: np.ndarray,
    sujetos: np.ndarray,
    proporcion_val: float = PROPORCION_VAL,
    proporcion_test: float = PROPORCION_TEST,
    semilla: int = 42,
) -> dict:
    """
    Divide en train/val/test sin que un sujeto aparezca en dos conjuntos.

    Usa GroupShuffleSplit dos veces: primero separa test, luego separa val
    del resto. Las proporciones se aplican sobre el numero de sujetos.

    Returns:
        Dict con X_/y_/sujetos_ para train, val y test, y "info".
    """
    sujetos = np.asarray(sujetos).astype(str)

    gss_test = GroupShuffleSplit(n_splits=1, test_size=proporcion_test, random_state=semilla)
    idx_resto, idx_test = next(gss_test.split(X, y, groups=sujetos))

    # val se calcula sobre lo que queda despues de quitar test
    val_relativo = proporcion_val / (1.0 - proporcion_test)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relativo, random_state=semilla)
    idx_train_rel, idx_val_rel = next(
        gss_val.split(X[idx_resto], y[idx_resto], groups=sujetos[idx_resto])
    )
    idx_train = idx_resto[idx_train_rel]
    idx_val = idx_resto[idx_val_rel]

    division = {}
    for split, idx in (("train", idx_train), ("val", idx_val), ("test", idx_test)):
        division[f"X_{split}"] = X[idx]
        division[f"y_{split}"] = y[idx]
        division[f"sujetos_{split}"] = sujetos[idx]

    division["info"] = calcular_info_division(division)
    return division


def guardar_division(division: dict, ruta_npz: str):
    """Guarda la division en un .npz comprimido (sin la clave "info")."""
    arrays = {k: v for k, v in division.items() if k != "info"}
    np.savez_compressed(ruta_npz, **arrays)
