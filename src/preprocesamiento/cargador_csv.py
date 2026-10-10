"""
Cargador de CSVs propios (sesiones capturadas con el TGAM).

Lee los archivos CSV generados por el GrabadorCSV (fase 3) y los convierte
en arrays NumPy listos para el pipeline de preprocesamiento.

Formato esperado:
  - *_raw.csv:   timestamp_epoch, timestamp_hora, raw_eeg, poor_signal
  - *_bandas.csv: timestamp_epoch, timestamp_hora, delta, ..., poor_signal

Los headers con '#' se saltan automaticamente.
"""

import glob
import os

import numpy as np
import pandas as pd


# Columnas numericas de bandas (sin timestamps ni poor_signal)
BANDAS_NOMBRES = [
    "delta", "theta", "low_alpha", "high_alpha",
    "low_beta", "high_beta", "low_gamma", "high_gamma",
]


def cargar_raw_csv(ruta: str) -> pd.DataFrame:
    """
    Carga un CSV de raw EEG generado por el GrabadorCSV.

    Args:
        ruta: Ruta al archivo *_raw.csv.

    Returns:
        DataFrame con columnas: timestamp_epoch, raw_eeg, poor_signal.
    """
    df = pd.read_csv(ruta, comment="#")
    # Verificar columnas minimas
    for col in ["timestamp_epoch", "raw_eeg", "poor_signal"]:
        if col not in df.columns:
            raise ValueError(f"Columna '{col}' no encontrada en {ruta}")
    return df[["timestamp_epoch", "raw_eeg", "poor_signal"]]


def cargar_bandas_csv(ruta: str) -> pd.DataFrame:
    """
    Carga un CSV de bandas + eSense generado por el GrabadorCSV.

    Args:
        ruta: Ruta al archivo *_bandas.csv.

    Returns:
        DataFrame con todas las columnas de bandas y eSense.
    """
    df = pd.read_csv(ruta, comment="#")
    cols_requeridas = BANDAS_NOMBRES + ["attention", "meditation", "poor_signal"]
    for col in cols_requeridas:
        if col not in df.columns:
            raise ValueError(f"Columna '{col}' no encontrada en {ruta}")
    return df


def cargar_sesion(carpeta_sesiones: str, sello: str) -> dict:
    """
    Carga ambos CSVs de una sesion (raw + bandas) por su sello temporal.

    Args:
        carpeta_sesiones: Ruta a datos/sesiones/.
        sello: Sello temporal de la sesion, ej. "2026-04-03_22-52-37".

    Returns:
        dict con clave obligatoria 'raw' y clave opcional 'bandas'.
    """
    ruta_raw = os.path.join(carpeta_sesiones, f"sesion_{sello}_raw.csv")
    ruta_bandas = os.path.join(carpeta_sesiones, f"sesion_{sello}_bandas.csv")

    if not os.path.exists(ruta_raw):
        raise FileNotFoundError(f"No se encontro: {ruta_raw}")

    resultado = {
        "raw": cargar_raw_csv(ruta_raw),
        "bandas": None,
    }
    if os.path.exists(ruta_bandas):
        resultado["bandas"] = cargar_bandas_csv(ruta_bandas)
    return resultado


def listar_sesiones(carpeta_sesiones: str) -> list[str]:
    """
    Lista los sellos temporales de todas las sesiones disponibles.

    Args:
        carpeta_sesiones: Ruta a datos/sesiones/.

    Returns:
        Lista de sellos temporales unicos, ej. ["2026-04-03_22-52-37", ...].
    """
    patron = os.path.join(carpeta_sesiones, "sesion_*_raw.csv")
    archivos = glob.glob(patron)
    sellos = []
    for arch in sorted(archivos):
        nombre = os.path.basename(arch)
        # sesion_YYYY-MM-DD_HH-MM-SS_raw.csv -> extraer sello
        partes = nombre.replace("sesion_", "").replace("_raw.csv", "")
        sellos.append(partes)
    return sellos


def cargar_todas_sesiones(carpeta_sesiones: str) -> list[dict]:
    """
    Carga todas las sesiones disponibles en la carpeta.

    Returns:
        Lista de dicts, cada uno con 'sello', 'raw' (DataFrame) y opcionalmente
        'bandas' (DataFrame o None).
    """
    sellos = listar_sesiones(carpeta_sesiones)
    sesiones = []
    for sello in sellos:
        datos = cargar_sesion(carpeta_sesiones, sello)
        datos["sello"] = sello
        sesiones.append(datos)
    return sesiones


def raw_a_numpy(df_raw: pd.DataFrame) -> np.ndarray:
    """
    Extrae la columna raw_eeg como array NumPy float32.

    Args:
        df_raw: DataFrame cargado con cargar_raw_csv().

    Returns:
        Array 1D de float32 con los valores raw EEG.
    """
    return df_raw["raw_eeg"].values.astype(np.float32)


def bandas_a_numpy(df_bandas: pd.DataFrame) -> np.ndarray:
    """
    Extrae las 8 bandas como matriz NumPy float32.

    Args:
        df_bandas: DataFrame cargado con cargar_bandas_csv().

    Returns:
        Array 2D de shape (n_muestras, 8) con las bandas de frecuencia.
    """
    return df_bandas[BANDAS_NOMBRES].values.astype(np.float32)
