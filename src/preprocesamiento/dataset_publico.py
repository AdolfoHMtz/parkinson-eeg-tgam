"""
Descarga y adaptacion del dataset publico OpenNeuro ds002778.

Dataset: UC San Diego Resting State EEG Data from Patients with Parkinson's Disease
  - 15 pacientes PD + 16 controles sanos
  - 32 canales EEG (BioSemi ActiveTwo), formato BDF/BIDS
  - 2 sesiones por paciente PD: on/off medicacion
  - Resting-state con ojos cerrados

Estrategia:
  - Se extrae el canal Fp1 (frente, el mas cercano a la posicion del TGAM)
  - Se remuestrea a 512 Hz para coincidir con el TGAM
  - Se exporta al mismo formato de array NumPy que los CSVs propios

Dependencias adicionales: mne, mne-bids, openneuro-py
"""

import os
import json

import numpy as np

# Intentar importar MNE (puede no estar instalado todavia)
try:
    import mne
    from mne_bids import BIDSPath, read_raw_bids
    MNE_DISPONIBLE = True
except ImportError:
    MNE_DISPONIBLE = False

try:
    import openneuro
    OPENNEURO_DISPONIBLE = True
except ImportError:
    OPENNEURO_DISPONIBLE = False


# Configuracion del dataset
DATASET_ID = "ds002778"
DATASET_VERSION = "1.0.5"

# Canal a extraer (equivalente a posicion TGAM en frente)
CANAL_FP1 = "Fp1"

# Frecuencia de muestreo objetivo (igual que TGAM)
FS_OBJETIVO = 512

# Sujetos del dataset (IDs reales del archivo participants.tsv)
# Los IDs no son consecutivos: pd3, pd5, pd6, pd9, pd11-pd14, pd16-pd17, pd19, pd22-pd23, pd26, pd28
SUJETOS_PD = ["pd3", "pd5", "pd6", "pd9", "pd11", "pd12", "pd13", "pd14",
              "pd16", "pd17", "pd19", "pd22", "pd23", "pd26", "pd28"]
# HC: hc1, hc2, hc4, hc7-hc8, hc10, hc18, hc20-hc21, hc24-hc25, hc29-hc33
SUJETOS_HC = ["hc1", "hc2", "hc4", "hc7", "hc8", "hc10", "hc18", "hc20",
              "hc21", "hc24", "hc25", "hc29", "hc30", "hc31", "hc32", "hc33"]


def verificar_dependencias():
    """Verifica que MNE y openneuro-py esten instalados."""
    faltantes = []
    if not MNE_DISPONIBLE:
        faltantes.append("mne, mne-bids")
    if not OPENNEURO_DISPONIBLE:
        faltantes.append("openneuro-py")
    if faltantes:
        raise ImportError(
            f"Faltan dependencias: {', '.join(faltantes)}. "
            f"Instalar con: pip install mne mne-bids openneuro-py"
        )


def descargar_dataset(destino: str, sujetos: list[str] | None = None):
    """
    Descarga el dataset ds002778 de OpenNeuro.

    Args:
        destino: Carpeta destino para los datos BIDS.
        sujetos: Lista de IDs de sujeto a descargar (ej. ["pd1", "hc1"]).
                 None descarga todos.
    """
    verificar_dependencias()
    os.makedirs(destino, exist_ok=True)

    if sujetos is None:
        # Descargar todo
        print(f"Descargando dataset completo {DATASET_ID} v{DATASET_VERSION}...")
        openneuro.download(dataset=DATASET_ID, target_dir=destino)
    else:
        # Descargar solo sujetos seleccionados
        # openneuro-py requiere trailing slash para directorios
        includes = [f"sub-{s}/" for s in sujetos]
        print(f"Descargando {len(sujetos)} sujetos: {', '.join(sujetos)}...")
        openneuro.download(
            dataset=DATASET_ID,
            target_dir=destino,
            include=includes,
        )
    print(f"Dataset descargado en: {destino}")


def _leer_raw_bids(bids_root: str, sujeto: str, sesion: str) -> "mne.io.Raw":
    """
    Lee un archivo raw EEG del dataset BIDS.

    Args:
        bids_root: Raiz del dataset BIDS.
        sujeto: ID del sujeto (ej. "pd1").
        sesion: Sesion ("off" o "on" para PD, "off" para HC).

    Returns:
        Objeto MNE Raw con los datos EEG.
    """
    bids_path = BIDSPath(
        root=bids_root,
        subject=sujeto,
        session=sesion,
        task="rest",
        datatype="eeg",
        suffix="eeg",
    )
    raw = read_raw_bids(bids_path=bids_path, verbose="WARNING")
    raw.load_data()
    return raw


def extraer_canal_fp1(raw: "mne.io.Raw") -> np.ndarray:
    """
    Extrae el canal Fp1 del objeto Raw y lo remuestrea a 512 Hz.

    BioSemi guarda datos en voltios con offset DC grande. Se convierte a
    microvoltios y se remueve el offset DC (media) para compatibilidad con
    el rango del TGAM (signed 16-bit, tipicamente -500 a 500).

    Args:
        raw: Objeto MNE Raw.

    Returns:
        Array 1D float32 con el raw EEG del canal Fp1 a 512 Hz en microvoltios,
        centrado en cero (sin offset DC).
    """
    # Seleccionar solo Fp1
    raw_fp1 = raw.copy().pick([CANAL_FP1])

    # Remuestrear si es necesario
    fs_actual = raw_fp1.info["sfreq"]
    if fs_actual != FS_OBJETIVO:
        print(f"  Remuestreando de {fs_actual} Hz a {FS_OBJETIVO} Hz...")
        raw_fp1.resample(FS_OBJETIVO)

    # Extraer datos como array 1D
    datos = raw_fp1.get_data()[0]  # shape (1, n_muestras) -> (n_muestras,)

    # Convertir voltios -> microvoltios y remover offset DC
    datos_uv = datos * 1e6
    datos_uv = datos_uv - np.mean(datos_uv)

    return datos_uv.astype(np.float32)


def extraer_todos_canales(raw: "mne.io.Raw") -> tuple[np.ndarray, list[str]]:
    """
    Extrae todos los canales EEG y remuestrea a 512 Hz.
    Util para CNN multicanal con el dataset publico.

    Convierte voltios a microvoltios y remueve offset DC por canal.

    Args:
        raw: Objeto MNE Raw.

    Returns:
        Tupla (datos, nombres_canales):
          - datos: Array 2D float32 (n_canales, n_muestras) en microvoltios
          - nombres_canales: Lista de nombres de canal
    """
    raw_eeg = raw.copy().pick("eeg")

    fs_actual = raw_eeg.info["sfreq"]
    if fs_actual != FS_OBJETIVO:
        raw_eeg.resample(FS_OBJETIVO)

    datos = raw_eeg.get_data()
    # Voltios -> microvoltios, remover offset DC por canal
    datos = datos * 1e6
    datos = datos - np.mean(datos, axis=1, keepdims=True)

    nombres = raw_eeg.ch_names
    return datos.astype(np.float32), nombres


def procesar_dataset(
    bids_root: str,
    destino: str,
    solo_fp1: bool = False,
):
    """
    Procesa todo el dataset BIDS y exporta arrays NumPy.

    Genera archivos .npz por sujeto/sesion con:
      - raw_fp1: array 1D del canal Fp1 a 512 Hz
      - raw_todos: array 2D de todos los canales a 512 Hz (si solo_fp1=False)
      - canales: nombres de los canales
      - etiqueta: 0=sano, 1=parkinson
      - sujeto: ID del sujeto
      - sesion: ID de la sesion
      - fs: frecuencia de muestreo (512)

    Args:
        bids_root: Raiz del dataset BIDS descargado.
        destino: Carpeta donde guardar los .npz.
        solo_fp1: Si True, solo extrae Fp1 (ahorra espacio).
    """
    verificar_dependencias()
    os.makedirs(destino, exist_ok=True)

    # Determinar sujetos disponibles
    sujetos_info = []

    for sujeto in SUJETOS_PD:
        carpeta = os.path.join(bids_root, f"sub-{sujeto}")
        if os.path.isdir(carpeta):
            # PD tiene sesiones "off" y "on"
            for sesion in ["off", "on"]:
                sesion_dir = os.path.join(carpeta, f"ses-{sesion}")
                if os.path.isdir(sesion_dir):
                    sujetos_info.append((sujeto, sesion, 1))  # etiqueta=1 (PD)

    for sujeto in SUJETOS_HC:
        carpeta = os.path.join(bids_root, f"sub-{sujeto}")
        if os.path.isdir(carpeta):
            # HC tiene sesion "hc" (no "off")
            sesion_dir = os.path.join(carpeta, "ses-hc")
            if os.path.isdir(sesion_dir):
                sujetos_info.append((sujeto, "hc", 0))  # etiqueta=0 (sano)

    print(f"Sujetos encontrados: {len(sujetos_info)} grabaciones")
    print(f"  PD: {sum(1 for _, _, e in sujetos_info if e == 1)}")
    print(f"  HC: {sum(1 for _, _, e in sujetos_info if e == 0)}")

    procesados = 0
    errores = []

    for sujeto, sesion, etiqueta in sujetos_info:
        nombre = f"sub-{sujeto}_ses-{sesion}"
        ruta_salida = os.path.join(destino, f"{nombre}.npz")

        if os.path.exists(ruta_salida):
            print(f"  [SKIP] {nombre} ya procesado")
            procesados += 1
            continue

        try:
            print(f"  Procesando {nombre}...")
            raw = _leer_raw_bids(bids_root, sujeto, sesion)

            datos_npz = {
                "etiqueta": np.array(etiqueta, dtype=np.int8),
                "sujeto": np.array(sujeto),
                "sesion": np.array(sesion),
                "fs": np.array(FS_OBJETIVO, dtype=np.int32),
            }

            # Extraer Fp1
            datos_npz["raw_fp1"] = extraer_canal_fp1(raw)

            # Extraer todos los canales (para CNN multicanal)
            if not solo_fp1:
                todos, nombres = extraer_todos_canales(raw)
                datos_npz["raw_todos"] = todos
                datos_npz["canales"] = np.array(nombres)

            np.savez_compressed(ruta_salida, **datos_npz)
            procesados += 1
            print(f"  [OK] {nombre} -> {ruta_salida}")

        except Exception as e:
            errores.append((nombre, str(e)))
            print(f"  [ERROR] {nombre}: {e}")

    print(f"\nProcesamiento completo: {procesados} OK, {len(errores)} errores")
    if errores:
        for nombre, err in errores:
            print(f"  - {nombre}: {err}")

    # Guardar indice
    indice = {
        "dataset": DATASET_ID,
        "version": DATASET_VERSION,
        "canal_principal": CANAL_FP1,
        "fs": FS_OBJETIVO,
        "solo_fp1": solo_fp1,
        "grabaciones": [
            {"sujeto": s, "sesion": se, "etiqueta": int(e), "archivo": f"sub-{s}_ses-{se}.npz"}
            for s, se, e in sujetos_info
        ],
    }
    ruta_indice = os.path.join(destino, "indice.json")
    with open(ruta_indice, "w", encoding="utf-8") as f:
        json.dump(indice, f, indent=2, ensure_ascii=False)
    print(f"Indice guardado en: {ruta_indice}")


def cargar_grabacion_npz(ruta: str) -> dict:
    """
    Carga una grabacion procesada (.npz).

    Returns:
        dict con claves: raw_fp1, etiqueta, sujeto, sesion, fs,
        y opcionalmente raw_todos, canales.
    """
    datos = np.load(ruta, allow_pickle=True)
    resultado = {
        "raw_fp1": datos["raw_fp1"],
        "etiqueta": int(datos["etiqueta"]),
        "sujeto": str(datos["sujeto"]),
        "sesion": str(datos["sesion"]),
        "fs": int(datos["fs"]),
    }
    if "raw_todos" in datos:
        resultado["raw_todos"] = datos["raw_todos"]
        resultado["canales"] = list(datos["canales"])
    return resultado


def cargar_indice(carpeta: str) -> dict:
    """Carga el indice.json del dataset procesado."""
    ruta = os.path.join(carpeta, "indice.json")
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)
