"""
Generacion de espectrogramas STFT del raw EEG.

Para cada ventana de raw EEG, calcula el espectrograma usando Short-Time
Fourier Transform (STFT) y lo guarda como array NumPy (NO como imagen PNG
para evitar perdida de precision).

Los espectrogramas alimentan los modelos CNN de la Rama B:
  - CNN 2D: espectrograma como "imagen" monocanal
  - CNN + LSTM hibrida: features extraidos por CNN -> secuencia para LSTM

Parametros STFT:
  - nperseg: Muestras por segmento FFT (256 por defecto = 0.5s a 512 Hz)
  - noverlap: Solapamiento entre segmentos (75% por defecto)
  - nfft: Puntos FFT (igual a nperseg)
  - window: Ventana de Hann
"""

import numpy as np
from scipy.signal import stft


# Parametros STFT por defecto
NPERSEG_DEFAULT = 256   # 0.5 s a 512 Hz
NOVERLAP_FRAC = 0.75    # 75% overlap entre segmentos STFT


def generar_espectrograma(
    ventana: np.ndarray,
    fs: int = 512,
    nperseg: int = NPERSEG_DEFAULT,
    noverlap_frac: float = NOVERLAP_FRAC,
    nfft: int = NPERSEG_DEFAULT,
    f_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula el espectrograma STFT de una ventana de raw EEG.

    Args:
        ventana: Array 1D con las muestras raw EEG de la ventana.
        fs: Frecuencia de muestreo.
        nperseg: Muestras por segmento FFT.
        noverlap_frac: Fraccion de solapamiento [0, 1).
        nfft: Puntos FFT.
        f_max: Frecuencia maxima a incluir (Hz). None = Nyquist.

    Returns:
        Tupla (Sxx, frecuencias, tiempos):
          - Sxx: Magnitud del espectrograma, shape (n_frecuencias, n_tiempos).
          - frecuencias: Array de frecuencias (Hz).
          - tiempos: Array de tiempos (s).
    """
    noverlap = int(nperseg * noverlap_frac)

    frecuencias, tiempos, Zxx = stft(
        ventana,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
    )

    # Magnitud (valor absoluto de los coeficientes complejos)
    Sxx = np.abs(Zxx).astype(np.float32)

    # Recortar frecuencias si se especifico f_max
    if f_max is not None:
        mascara = frecuencias <= f_max
        frecuencias = frecuencias[mascara]
        Sxx = Sxx[mascara, :]

    return Sxx, frecuencias, tiempos


def normalizar_espectrograma(Sxx: np.ndarray) -> np.ndarray:
    """
    Normaliza un espectrograma a [0, 1] por ventana.

    Usa escala logaritmica (dB) para mejorar el rango dinamico,
    luego escala MinMax a [0, 1].

    Args:
        Sxx: Espectrograma (n_frecuencias, n_tiempos).

    Returns:
        Espectrograma normalizado a [0, 1].
    """
    # Escala logaritmica (evitar log(0))
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # MinMax a [0, 1]
    smin = Sxx_db.min()
    smax = Sxx_db.max()
    if smax == smin:
        return np.zeros_like(Sxx_db)
    return ((Sxx_db - smin) / (smax - smin)).astype(np.float32)


def generar_espectrogramas_lote(
    ventanas: np.ndarray,
    fs: int = 512,
    nperseg: int = NPERSEG_DEFAULT,
    noverlap_frac: float = NOVERLAP_FRAC,
    nfft: int = NPERSEG_DEFAULT,
    f_max: float | None = None,
    normalizar: bool = True,
) -> np.ndarray:
    """
    Genera espectrogramas para un lote de ventanas.

    Args:
        ventanas: Array 2D (n_ventanas, muestras_por_ventana).
        fs, nperseg, noverlap_frac, nfft, f_max: Parametros STFT.
        normalizar: Si True, normaliza cada espectrograma a [0, 1].

    Returns:
        Array 3D (n_ventanas, n_frecuencias, n_tiempos).
    """
    n_ventanas = ventanas.shape[0]

    # Calcular dimensiones con la primera ventana
    Sxx0, _, _ = generar_espectrograma(ventanas[0], fs, nperseg, noverlap_frac, nfft, f_max)
    n_freq, n_time = Sxx0.shape

    # Pre-allocar el array de salida
    espectrogramas = np.empty((n_ventanas, n_freq, n_time), dtype=np.float32)
    espectrogramas[0] = normalizar_espectrograma(Sxx0) if normalizar else Sxx0

    for i in range(1, n_ventanas):
        Sxx, _, _ = generar_espectrograma(ventanas[i], fs, nperseg, noverlap_frac, nfft, f_max)
        espectrogramas[i] = normalizar_espectrograma(Sxx) if normalizar else Sxx

    return espectrogramas


def info_espectrograma(
    ventana_muestras: int,
    fs: int = 512,
    nperseg: int = NPERSEG_DEFAULT,
    noverlap_frac: float = NOVERLAP_FRAC,
    nfft: int = NPERSEG_DEFAULT,
    f_max: float | None = None,
) -> dict:
    """
    Calcula las dimensiones del espectrograma sin generarlo.

    Returns:
        dict con: n_frecuencias, n_tiempos, resolucion_freq_hz, resolucion_temporal_s.
    """
    noverlap = int(nperseg * noverlap_frac)
    paso = nperseg - noverlap

    n_freq = nfft // 2 + 1
    n_time = (ventana_muestras - nperseg) // paso + 1

    if f_max is not None:
        freq_resolucion = fs / nfft
        n_freq = int(f_max / freq_resolucion) + 1

    return {
        "n_frecuencias": n_freq,
        "n_tiempos": n_time,
        "resolucion_freq_hz": fs / nfft,
        "resolucion_temporal_s": paso / fs,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
    }
