"""
Extraccion de features estadisticos por ventana temporal.

Para cada ventana de senal EEG, calcula:
  - Media
  - Varianza
  - Skewness (asimetria)
  - Kurtosis (apuntamiento)

Se aplica tanto a raw EEG (1 canal) como a bandas (8 canales),
generando un vector de features por ventana que alimenta los
modelos de series temporales (LSTM, GRU, Transformer).
"""

import numpy as np
from scipy import stats


FEATURE_NAMES = ["media", "varianza", "skewness", "kurtosis"]


def extraer_features_lote(ventanas: np.ndarray) -> np.ndarray:
    """
    Extrae features estadisticos para un lote de ventanas.

    Args:
        ventanas: Array 2D (n_ventanas, muestras) para 1D,
                  o 3D (n_ventanas, muestras, n_canales) para multicanal.

    Returns:
        Array 2D (n_ventanas, n_features):
          - 1D: (n_ventanas, 4)
          - 3D: (n_ventanas, n_canales * 4), orden [media_ch0, var_ch0, skew_ch0, kurt_ch0, ...]
    """
    if ventanas.ndim == 2:
        # (n_ventanas, muestras) -> vectorizado sobre axis=1
        return np.column_stack([
            np.mean(ventanas, axis=1),
            np.var(ventanas, axis=1),
            stats.skew(ventanas, axis=1),
            stats.kurtosis(ventanas, axis=1),
        ]).astype(np.float32)

    elif ventanas.ndim == 3:
        # (n_ventanas, muestras, n_canales) -> vectorizado por canal
        n_ventanas, _, n_canales = ventanas.shape
        # Transponer a (n_canales, n_ventanas, muestras) para procesar cada canal
        v = ventanas.transpose(2, 0, 1)
        partes = []
        for c in range(n_canales):
            partes.append(np.mean(v[c], axis=1))
            partes.append(np.var(v[c], axis=1))
            partes.append(stats.skew(v[c], axis=1))
            partes.append(stats.kurtosis(v[c], axis=1))
        return np.column_stack(partes).astype(np.float32)

    else:
        raise ValueError(f"ventanas debe ser 2D o 3D, recibido {ventanas.ndim}D")


def nombres_features_1d() -> list[str]:
    return list(FEATURE_NAMES)


def nombres_features_bandas(nombres_canales: list[str]) -> list[str]:
    """
    Retorna los nombres de features para senal multicanal.

    Args:
        nombres_canales: Lista de nombres de canal, ej. ["delta", "theta", ...].

    Returns:
        Lista de nombres, ej. ["delta_media", "delta_varianza", ...].
    """
    return [f"{canal}_{sufijo}" for canal in nombres_canales for sufijo in FEATURE_NAMES]
