"""
Ventanas deslizantes (sliding windows) para segmentar senales EEG.

Soporta:
  - Multiples tamanios de ventana: 1s, 2s, 5s, 10s
  - Overlap configurable: 50%, 75%
  - Aplicable a raw EEG (512 Hz) y bandas (~1 Hz)

La segmentacion por ventanas es el paso previo a:
  - Rama A: Extraccion de features estadisticos -> LSTM/Transformer
  - Rama B: Generacion de espectrogramas STFT -> CNN
"""

import numpy as np


# Tamanios de ventana predefinidos (en segundos)
TAMANOS_VENTANA = [1, 2, 5, 10]

# Porcentajes de overlap predefinidos
OVERLAPS = [0.5, 0.75]


def crear_ventanas(
    datos: np.ndarray,
    fs: int,
    ventana_seg: float,
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Segmenta una senal 1D en ventanas deslizantes.

    Args:
        datos: Array 1D con la senal (ej. raw EEG).
        fs: Frecuencia de muestreo en Hz.
        ventana_seg: Duracion de la ventana en segundos.
        overlap: Fraccion de solapamiento [0.0, 1.0). Ej: 0.5 = 50%.

    Returns:
        Array 2D de shape (n_ventanas, muestras_por_ventana).

    Raises:
        ValueError: Si la senal es mas corta que una ventana.
    """
    muestras_ventana = int(ventana_seg * fs)
    paso = int(muestras_ventana * (1.0 - overlap))

    if paso < 1:
        paso = 1

    n_muestras = len(datos)
    if n_muestras < muestras_ventana:
        raise ValueError(
            f"La senal ({n_muestras} muestras) es mas corta que una ventana "
            f"({muestras_ventana} muestras = {ventana_seg}s a {fs} Hz)."
        )

    # Calcular numero de ventanas completas
    n_ventanas = (n_muestras - muestras_ventana) // paso + 1

    # Crear indices con stride tricks para eficiencia (sin copiar datos)
    indices = np.arange(muestras_ventana)[None, :] + np.arange(n_ventanas)[:, None] * paso
    ventanas = datos[indices]

    return ventanas


def crear_ventanas_2d(
    datos: np.ndarray,
    fs: int,
    ventana_seg: float,
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Segmenta una senal multicanal 2D en ventanas deslizantes.

    Args:
        datos: Array 2D (n_muestras, n_canales). Ej: 8 bandas.
        fs: Frecuencia de muestreo en Hz.
        ventana_seg: Duracion de la ventana en segundos.
        overlap: Fraccion de solapamiento [0.0, 1.0).

    Returns:
        Array 3D de shape (n_ventanas, muestras_por_ventana, n_canales).
    """
    n_muestras, n_canales = datos.shape
    muestras_ventana = int(ventana_seg * fs)
    paso = int(muestras_ventana * (1.0 - overlap))

    if paso < 1:
        paso = 1

    if n_muestras < muestras_ventana:
        raise ValueError(
            f"La senal ({n_muestras} muestras) es mas corta que una ventana "
            f"({muestras_ventana} muestras = {ventana_seg}s a {fs} Hz)."
        )

    n_ventanas = (n_muestras - muestras_ventana) // paso + 1

    # Usar stride indexing igual que crear_ventanas 1D
    indices = np.arange(muestras_ventana)[None, :] + np.arange(n_ventanas)[:, None] * paso
    return datos[indices]


def info_ventanas(n_muestras: int, fs: int, ventana_seg: float, overlap: float) -> dict:
    """
    Calcula informacion sobre la segmentacion sin crear las ventanas.

    Returns:
        dict con: muestras_por_ventana, paso, n_ventanas, duracion_total_seg.
    """
    muestras_ventana = int(ventana_seg * fs)
    paso = max(1, int(muestras_ventana * (1.0 - overlap)))
    n_ventanas = max(0, (n_muestras - muestras_ventana) // paso + 1)

    return {
        "muestras_por_ventana": muestras_ventana,
        "paso": paso,
        "n_ventanas": n_ventanas,
        "duracion_total_seg": n_muestras / fs,
        "ventana_seg": ventana_seg,
        "overlap": overlap,
    }
