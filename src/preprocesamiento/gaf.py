"""
Generacion de Gramian Angular Fields (GAF) a partir de series temporales EEG.

Esta representacion convierte una ventana 1D de raw EEG en una imagen 2D
que preserva relaciones temporales globales y es adecuada para CNNs.

Flujo:
  1. Reducir la serie a un tamano manejable (ej. 128 puntos)
  2. Normalizar la ventana a [-1, 1]
  3. Convertir a angulos con arccos
  4. Construir la matriz GAF:
       - summation: cos(phi_i + phi_j)
       - difference: sin(phi_i - phi_j)
"""

from __future__ import annotations

import numpy as np


GAF_SUMMATION = "summation"
GAF_DIFFERENCE = "difference"
GAF_METHODS = [GAF_SUMMATION, GAF_DIFFERENCE]


def reescalar_serie(serie: np.ndarray, nuevo_tamano: int) -> np.ndarray:
    """
    Reescala una serie 1D a un numero fijo de puntos via interpolacion lineal.

    Args:
        serie: Array 1D.
        nuevo_tamano: Longitud deseada.

    Returns:
        Array 1D float32 de longitud `nuevo_tamano`.
    """
    serie = np.asarray(serie, dtype=np.float32).ravel()
    if nuevo_tamano < 2:
        raise ValueError("nuevo_tamano debe ser >= 2.")
    if len(serie) == nuevo_tamano:
        return serie.copy()

    x_origen = np.linspace(0.0, 1.0, num=len(serie), dtype=np.float32)
    x_destino = np.linspace(0.0, 1.0, num=nuevo_tamano, dtype=np.float32)
    return np.interp(x_destino, x_origen, serie).astype(np.float32)


def normalizar_serie_gaf(serie: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Normaliza una serie a [-1, 1] por ventana.

    Se centra en la media y se escala por el maximo absoluto para reducir
    sensibilidad al offset DC residual y a diferencias de amplitud global.
    """
    serie = np.asarray(serie, dtype=np.float32)
    serie_centrada = serie - np.mean(serie)
    max_abs = np.max(np.abs(serie_centrada))
    if max_abs < epsilon:
        return np.zeros_like(serie_centrada, dtype=np.float32)
    serie_norm = serie_centrada / max_abs
    return np.clip(serie_norm, -1.0, 1.0).astype(np.float32)


def generar_gaf(
    serie: np.ndarray,
    tamano: int = 128,
    metodo: str = GAF_SUMMATION,
) -> np.ndarray:
    """
    Genera una imagen GAF 2D a partir de una serie 1D.

    Args:
        serie: Array 1D de la ventana temporal.
        tamano: Resolucion final de la imagen cuadrada.
        metodo: "summation" o "difference".

    Returns:
        Array 2D float32 de shape (tamano, tamano), con valores en [-1, 1].
    """
    serie_reescalada = reescalar_serie(serie, tamano)
    serie_norm = normalizar_serie_gaf(serie_reescalada)
    phi = np.arccos(serie_norm)

    if metodo == GAF_SUMMATION:
        matriz = np.cos(phi[:, None] + phi[None, :])
    elif metodo == GAF_DIFFERENCE:
        matriz = np.sin(phi[:, None] - phi[None, :])
    else:
        raise ValueError(
            f"Metodo GAF desconocido: {metodo}. Usar uno de: {', '.join(GAF_METHODS)}"
        )

    return matriz.astype(np.float32)


def generar_gaf_lote(
    ventanas: np.ndarray,
    tamano: int = 128,
    metodo: str = GAF_SUMMATION,
) -> np.ndarray:
    """
    Genera GAF para un lote de ventanas.

    Args:
        ventanas: Array 2D (n_ventanas, muestras_por_ventana).
        tamano: Resolucion final del GAF.
        metodo: "summation" o "difference".

    Returns:
        Array 3D (n_ventanas, tamano, tamano).
    """
    if ventanas.ndim != 2:
        raise ValueError(f"ventanas debe ser 2D, recibido {ventanas.ndim}D")

    n_ventanas = ventanas.shape[0]
    salida = np.empty((n_ventanas, tamano, tamano), dtype=np.float32)
    for i in range(n_ventanas):
        salida[i] = generar_gaf(ventanas[i], tamano=tamano, metodo=metodo)
    return salida


def info_gaf(tamano: int = 128, metodo: str = GAF_SUMMATION) -> dict:
    """Retorna metadata descriptiva de la representacion GAF."""
    return {
        "tipo": "gaf",
        "tamano": tamano,
        "metodo": metodo,
        "shape": [tamano, tamano],
        "rango_teorico": [-1.0, 1.0],
    }
