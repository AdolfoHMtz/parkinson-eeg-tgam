"""
Pipeline principal de preprocesamiento de datos EEG.

Estrategia actual recomendada:
  1. Cargar raw EEG local (TGAM) y/o OpenNeuro Fp1
  2. Segmentar en ventanas deslizantes
  3. Rama A: construir series temporales normalizadas (raw o features)
  4. Rama B: construir imagenes desde la serie temporal (GAF por defecto, STFT legado)
  5. Etiquetar y dividir en train/val/test por sujeto
  6. Guardar salidas listas para entrenamiento local
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

from .cargador_csv import (
    cargar_todas_sesiones,
    raw_a_numpy,
    bandas_a_numpy,
    BANDAS_NOMBRES,
)
from .dataset_publico import (
    cargar_grabacion_npz,
    cargar_indice,
)
from .normalizacion import crear_normalizador
from .ventanas import crear_ventanas, crear_ventanas_2d, info_ventanas
from .features import extraer_features_lote, nombres_features_1d, nombres_features_bandas
from .gaf import generar_gaf_lote, info_gaf, GAF_METHODS
from .espectrogramas import (
    generar_espectrogramas_lote,
    info_espectrograma,
)
from .rebalancear_splits_rama_b import (
    construir_catalogo_sujetos as construir_catalogo_sujetos_balanceado,
    asignar_sujetos_balanceado,
    construir_division,
)
from ..exportacion.grabador_csv import SUJETO_DEFAULT
from .etiquetado import (
    dividir_por_sujeto,
    crear_etiquetas,
    crear_ids_sujeto,
    guardar_division,
    SANO,
    PARKINSON,
)


# Rutas por defecto relativas al proyecto
RUTA_SESIONES = os.path.join("datos", "sesiones")
RUTA_DATASET_PUBLICO = os.path.join("datos", "openneuro_procesado")
RUTA_SALIDA = os.path.join("datos", "procesados")
SUJETO_LOCAL_DEFAULT = SUJETO_DEFAULT


def pipeline_rama_a(
    datos_raw_por_sujeto: list[dict],
    ventana_seg: float,
    overlap: float,
    tipo_norm: str,
    repr_temporal: str = "raw",
    fs: int = 512,
    destino: str = RUTA_SALIDA,
) -> dict:
    """
    Pipeline Rama A: Series temporales.

    Flujo:
      - raw EEG -> ventanas -> normalizacion -> division
      - o raw EEG -> ventanas -> features estadisticos -> normalizacion -> division

    Args:
        datos_raw_por_sujeto: Lista de dicts con:
            - raw: array 1D float32 con raw EEG
            - sujeto: ID del sujeto
            - etiqueta: 0 (sano) o 1 (Parkinson)
        ventana_seg: Duracion de ventana en segundos.
        overlap: Fraccion de solapamiento.
        tipo_norm: "minmax" o "zscore".
        fs: Frecuencia de muestreo.
        destino: Carpeta destino.

    Returns:
        dict con info del procesamiento.
    """
    print(f"\n{'='*60}")
    print(f"RAMA A - Series temporales")
    print(
        f"  Ventana: {ventana_seg}s | Overlap: {overlap*100:.0f}% | "
        f"Norm: {tipo_norm} | Repr: {repr_temporal}"
    )
    print(f"{'='*60}")

    todos_features = []
    todas_etiquetas = []
    todos_sujetos = []

    for datos in datos_raw_por_sujeto:
        raw = datos["raw"]
        sujeto = datos["sujeto"]
        etiqueta = datos["etiqueta"]

        # Verificar que la senal es suficientemente larga
        info = info_ventanas(len(raw), fs, ventana_seg, overlap)
        if info["n_ventanas"] == 0:
            print(f"  [SKIP] {sujeto}: senal muy corta ({len(raw)/fs:.1f}s < {ventana_seg}s)")
            continue

        # Crear ventanas temporales desde raw EEG
        ventanas = crear_ventanas(raw, fs, ventana_seg, overlap)

        if repr_temporal == "raw":
            salida = ventanas.astype(np.float32)
        elif repr_temporal == "features":
            salida = extraer_features_lote(ventanas)
        else:
            raise ValueError(
                f"Representacion temporal desconocida: {repr_temporal}. "
                "Usar 'raw' o 'features'."
            )

        todos_features.append(salida)
        todas_etiquetas.append(crear_etiquetas(len(salida), etiqueta))
        todos_sujetos.append(crear_ids_sujeto(len(salida), sujeto))

        print(f"  {sujeto}: {info['n_ventanas']} ventanas -> {salida.shape}")

    if not todos_features:
        print("  [ERROR] No hay datos suficientes para procesar.")
        return {"error": "sin datos"}

    # Concatenar todo
    X = np.concatenate(todos_features, axis=0)
    y = np.concatenate(todas_etiquetas, axis=0)
    sujetos = np.concatenate(todos_sujetos, axis=0)

    print(f"\n  Total: {X.shape[0]} ventanas, shape por muestra: {list(X.shape[1:])}")
    print(f"  Sano: {np.sum(y == SANO)}, Parkinson: {np.sum(y == PARKINSON)}")

    # Dividir por sujeto
    division = dividir_por_sujeto(X, y, sujetos)

    # Normalizar (ajustar solo en train, aplicar a val y test)
    normalizador = crear_normalizador(tipo_norm)
    normalizador.ajustar(division["X_train"])
    division["X_train"] = normalizador.transformar(division["X_train"])
    division["X_val"] = normalizador.transformar(division["X_val"])
    division["X_test"] = normalizador.transformar(division["X_test"])

    # Guardar
    sufijo = f"rama_a_{repr_temporal}_v{ventana_seg}s_o{int(overlap*100)}_{tipo_norm}"
    carpeta_salida = os.path.join(destino, sufijo)
    os.makedirs(carpeta_salida, exist_ok=True)

    ruta_datos = os.path.join(carpeta_salida, "datos.npz")
    guardar_division(division, ruta_datos)

    # Guardar parametros de normalizacion
    params_norm = normalizador.parametros()
    np.savez(os.path.join(carpeta_salida, "normalizador.npz"), **params_norm)

    # Guardar metadata
    metadata = {
        "rama": "A",
        "tipo": "series_temporales",
        "representacion_temporal": repr_temporal,
        "ventana_seg": ventana_seg,
        "overlap": overlap,
        "normalizacion": tipo_norm,
        "fs": fs,
        "shape_muestra": list(X.shape[1:]),
        "n_features": int(X.shape[1]) if X.ndim == 2 else None,
        "features_nombres": nombres_features_1d() if repr_temporal == "features" else [],
        "info_division": division["info"],
    }
    with open(os.path.join(carpeta_salida, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Guardado en: {carpeta_salida}")
    _imprimir_info_division(division["info"])

    return metadata


def pipeline_rama_b(
    datos_raw_por_sujeto: list[dict],
    ventana_seg: float,
    overlap: float,
    repr_imagen: str = "gaf",
    fs: int = 512,
    destino: str = RUTA_SALIDA,
    f_max: float = 50.0,
    gaf_size: int = 128,
    gaf_method: str = "summation",
    balancear_splits: bool = True,
) -> dict:
    """
    Pipeline Rama B: Espectrogramas STFT.

    Flujo:
      - raw EEG -> ventanas -> GAF -> division
      - o raw EEG -> ventanas -> STFT -> normalizacion [0,1] -> division

    Args:
        datos_raw_por_sujeto: Lista de dicts (misma estructura que Rama A).
        ventana_seg: Duracion de ventana en segundos.
        overlap: Fraccion de solapamiento.
        fs: Frecuencia de muestreo.
        destino: Carpeta destino.
        f_max: Frecuencia maxima en Hz (50 Hz cubre todas las bandas EEG).

    Returns:
        dict con info del procesamiento.
    """
    print(f"\n{'='*60}")
    print(f"RAMA B - Imagenes derivadas de serie temporal")
    print(
        f"  Ventana: {ventana_seg}s | Overlap: {overlap*100:.0f}% | "
        f"Repr: {repr_imagen}"
    )
    print(f"{'='*60}")

    todos_espectrogramas = []
    todas_etiquetas = []
    todos_sujetos = []

    for datos in datos_raw_por_sujeto:
        raw = datos["raw"]
        sujeto = datos["sujeto"]
        etiqueta = datos["etiqueta"]

        info = info_ventanas(len(raw), fs, ventana_seg, overlap)
        if info["n_ventanas"] == 0:
            print(f"  [SKIP] {sujeto}: senal muy corta ({len(raw)/fs:.1f}s < {ventana_seg}s)")
            continue

        # Crear ventanas
        ventanas = crear_ventanas(raw, fs, ventana_seg, overlap)

        if repr_imagen == "gaf":
            imagenes = generar_gaf_lote(
                ventanas,
                tamano=gaf_size,
                metodo=gaf_method,
            )
        elif repr_imagen == "stft":
            imagenes = generar_espectrogramas_lote(
                ventanas, fs=fs, normalizar=True, f_max=f_max
            )
        else:
            raise ValueError(
                f"Representacion de imagen desconocida: {repr_imagen}. "
                "Usar 'gaf' o 'stft'."
            )

        todos_espectrogramas.append(imagenes)
        todas_etiquetas.append(crear_etiquetas(len(imagenes), etiqueta))
        todos_sujetos.append(crear_ids_sujeto(len(imagenes), sujeto))

        print(f"  {sujeto}: {info['n_ventanas']} ventanas -> imagen {imagenes.shape[1:]}")

    if not todos_espectrogramas:
        print("  [ERROR] No hay datos suficientes para procesar.")
        return {"error": "sin datos"}

    # Concatenar
    X = np.concatenate(todos_espectrogramas, axis=0)
    y = np.concatenate(todas_etiquetas, axis=0)
    sujetos = np.concatenate(todos_sujetos, axis=0)

    print(f"\n  Total: {X.shape[0]} imagenes de shape {X.shape[1:]}")
    print(f"  Sano: {np.sum(y == SANO)}, Parkinson: {np.sum(y == PARKINSON)}")

    # Dividir por sujeto
    if balancear_splits:
        catalogo = construir_catalogo_sujetos_balanceado(y, sujetos)
        asignacion = asignar_sujetos_balanceado(catalogo)
        division = construir_division(X, y, sujetos, asignacion)
    else:
        division = dividir_por_sujeto(X, y, sujetos)

    # Guardar
    sufijo = f"rama_b_{repr_imagen}_v{ventana_seg}s_o{int(overlap*100)}"
    if balancear_splits:
        sufijo += "_balanceado"
    carpeta_salida = os.path.join(destino, sufijo)
    os.makedirs(carpeta_salida, exist_ok=True)

    ruta_datos = os.path.join(carpeta_salida, "datos.npz")
    guardar_division(division, ruta_datos)

    muestras_ventana = int(ventana_seg * fs)
    if repr_imagen == "gaf":
        info_repr = info_gaf(tamano=gaf_size, metodo=gaf_method)
    else:
        info_repr = info_espectrograma(muestras_ventana, fs, f_max=f_max)

    metadata = {
        "rama": "B",
        "tipo": "imagenes_temporales",
        "representacion_imagen": repr_imagen,
        "ventana_seg": ventana_seg,
        "overlap": overlap,
        "fs": fs,
        "f_max": f_max,
        "gaf_size": gaf_size,
        "gaf_method": gaf_method,
        "shape_imagen": list(X.shape[1:]),
        "info_representacion": info_repr,
        "info_division": division["info"],
    }
    with open(os.path.join(carpeta_salida, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Guardado en: {carpeta_salida}")
    _imprimir_info_division(division["info"])

    return metadata


def cargar_sujetos_locales(carpeta_sesiones: str) -> dict[str, str]:
    """
    Lee la columna opcional `sujeto` de etiquetas_locales.csv (sello -> sujeto).

    Las sesiones sin sujeto registrado se asignan a SUJETO_LOCAL_DEFAULT: se
    asume que son de la misma persona, que es la opcion segura para no repetir
    a alguien en train y test.
    """
    ruta = os.path.join(carpeta_sesiones, "etiquetas_locales.csv")
    if not os.path.exists(ruta):
        return {}
    with open(ruta, "r", encoding="utf-8", newline="") as f:
        return {
            fila["sello"].strip(): fila["sujeto"].strip()
            for fila in csv.DictReader(f)
            if fila.get("sello") and (fila.get("sujeto") or "").strip()
        }


def _id_sujeto_local(sujetos_por_sello: dict[str, str], sello: str) -> str:
    return f"local_{sujetos_por_sello.get(sello, SUJETO_LOCAL_DEFAULT).removeprefix('local_')}"


def cargar_datos_locales(carpeta_sesiones: str, etiqueta: int = SANO) -> list[dict]:
    """
    Carga sesiones locales (TGAM) como lista de dicts para el pipeline.

    Args:
        carpeta_sesiones: Ruta a datos/sesiones/.
        etiqueta: Etiqueta a asignar a TODAS las sesiones locales:
              SANO=0 o PARKINSON=1.

    Returns:
        Lista de dicts con claves: raw, sujeto, etiqueta.
    """
    sesiones = cargar_todas_sesiones(carpeta_sesiones)
    sujetos_por_sello = cargar_sujetos_locales(carpeta_sesiones)
    datos = []
    for sesion in sesiones:
        raw = raw_a_numpy(sesion["raw"])
        datos.append({
            "raw": raw,
            "sujeto": _id_sujeto_local(sujetos_por_sello, sesion["sello"]),
            "etiqueta": etiqueta,
        })
    return datos


def cargar_etiquetas_locales_csv(ruta_csv: str) -> dict[str, int]:
    """
    Carga un mapa de etiquetas locales por sello de sesion.

    Formato esperado (con o sin encabezado):
        sello,etiqueta
        2026-04-03_22-30-37,sano
        2026-04-03_22-34-12,parkinson

    Valores validos para etiqueta: sano|0, parkinson|1.
    """
    etiquetas_por_sello: dict[str, int] = {}

    with open(ruta_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, start=1):
            if not row:
                continue

            # Permitir encabezado opcional
            if i == 1 and row[0].strip().lower() in {"sello", "sesion", "session"}:
                continue

            if len(row) < 2:
                raise ValueError(
                    f"Fila invalida en {ruta_csv} (linea {i}): se esperan 2 columnas (sello,etiqueta)."
                )

            sello = row[0].strip()
            etiqueta_txt = row[1].strip().lower()

            if not sello:
                raise ValueError(f"Fila invalida en {ruta_csv} (linea {i}): sello vacio.")

            if etiqueta_txt in {"sano", "0"}:
                etiqueta = SANO
            elif etiqueta_txt in {"parkinson", "1"}:
                etiqueta = PARKINSON
            else:
                raise ValueError(
                    f"Etiqueta invalida en {ruta_csv} (linea {i}): '{row[1]}'. "
                    "Usar sano|0 o parkinson|1."
                )

            etiquetas_por_sello[sello] = etiqueta

    return etiquetas_por_sello


def cargar_datos_locales_con_mapa(
    carpeta_sesiones: str,
    etiquetas_por_sello: dict[str, int],
    etiqueta_default: int,
) -> list[dict]:
    """
    Carga sesiones locales usando un mapa de etiquetas por sello.

    Si una sesion no aparece en el mapa, usa etiqueta_default.
    """
    sesiones = cargar_todas_sesiones(carpeta_sesiones)
    sujetos_por_sello = cargar_sujetos_locales(carpeta_sesiones)
    datos = []
    for sesion in sesiones:
        raw = raw_a_numpy(sesion["raw"])
        sello = sesion["sello"]
        etiqueta = etiquetas_por_sello.get(sello, etiqueta_default)
        datos.append({
            "raw": raw,
            "sujeto": _id_sujeto_local(sujetos_por_sello, sello),
            "etiqueta": etiqueta,
        })
    return datos


def cargar_datos_publicos(carpeta_procesado: str) -> list[dict]:
    """
    Carga el dataset publico procesado (.npz) como lista de dicts.

    Args:
        carpeta_procesado: Ruta a datos/openneuro_procesado/.

    Returns:
        Lista de dicts con claves: raw, sujeto, etiqueta.
    """
    indice = cargar_indice(carpeta_procesado)
    datos = []
    for grab in indice["grabaciones"]:
        ruta = os.path.join(carpeta_procesado, grab["archivo"])
        if not os.path.exists(ruta):
            print(f"  [SKIP] {grab['archivo']} no encontrado")
            continue
        npz = cargar_grabacion_npz(ruta)
        # El grupo para dividir es la persona, no la grabacion: los pacientes PD
        # tienen sesion on y off, y ambas deben caer en el mismo conjunto.
        datos.append({
            "raw": npz["raw_fp1"],
            "sujeto": str(npz["sujeto"]),
            "etiqueta": npz["etiqueta"],
        })
    return datos


def _imprimir_info_division(info: dict):
    """Imprime resumen de la division train/val/test."""
    print(f"\n  Division train/val/test:")
    print(f"    Train: {info['n_train']} muestras, {info['n_sujetos_train']} sujetos "
          f"(PD: {info['balance_train']['parkinson']}, HC: {info['balance_train']['sano']})")
    print(f"    Val:   {info['n_val']} muestras, {info['n_sujetos_val']} sujetos "
          f"(PD: {info['balance_val']['parkinson']}, HC: {info['balance_val']['sano']})")
    print(f"    Test:  {info['n_test']} muestras, {info['n_sujetos_test']} sujetos "
          f"(PD: {info['balance_test']['parkinson']}, HC: {info['balance_test']['sano']})")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de preprocesamiento EEG para deteccion de Parkinson"
    )
    parser.add_argument(
        "--fuente", choices=["local", "publico", "ambos"], default="local",
        help="Fuente de datos: 'local' (CSVs TGAM), 'publico' (OpenNeuro), 'ambos'"
    )
    parser.add_argument(
        "--etiqueta-local", choices=["sano", "parkinson"], default="sano",
        help="Etiqueta para sesiones locales: 'sano'=0, 'parkinson'=1 (default: sano)"
    )
    parser.add_argument(
        "--etiquetas-local-csv", default="",
        help=(
            "CSV opcional con etiquetas por sesion local (columnas: sello,etiqueta). "
            "Si existe, tiene prioridad sobre --etiqueta-local."
        )
    )
    parser.add_argument(
        "--ventana", type=float, default=2.0,
        help="Duracion de ventana en segundos (default: 2.0)"
    )
    parser.add_argument(
        "--overlap", type=float, default=0.5,
        help="Fraccion de overlap (default: 0.5)"
    )
    parser.add_argument(
        "--norm", choices=["minmax", "zscore", "ambos"], default="ambos",
        help="Tipo de normalizacion para Rama A (default: ambos)"
    )
    parser.add_argument(
        "--repr-temporal", choices=["raw", "features"], default="raw",
        help="Representacion de Rama A: raw o features (default: raw)"
    )
    parser.add_argument(
        "--repr-imagen", choices=["gaf", "stft"], default="gaf",
        help="Representacion de Rama B: gaf o stft (default: gaf)"
    )
    parser.add_argument(
        "--gaf-size", type=int, default=128,
        help="Tamano de salida del GAF (default: 128)"
    )
    parser.add_argument(
        "--sesiones", default=RUTA_SESIONES,
        help="Carpeta de sesiones locales"
    )
    parser.add_argument(
        "--dataset", default=RUTA_DATASET_PUBLICO,
        help="Carpeta del dataset publico procesado"
    )
    parser.add_argument(
        "--salida", default=RUTA_SALIDA,
        help="Carpeta de salida"
    )
    parser.add_argument(
        "--rama", choices=["a", "b", "ambas"], default="ambas",
        help="Rama a ejecutar (default: ambas)"
    )
    parser.add_argument(
        "--balancear-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balancea la division por sujeto en la Rama B (default: true)",
    )

    args = parser.parse_args()
    t_inicio = time.time()

    print("="*60)
    print("PIPELINE DE PREPROCESAMIENTO EEG")
    print(f"  Fuente: {args.fuente}")
    print(f"  Etiqueta local: {args.etiqueta_local}")
    if args.etiquetas_local_csv:
        print(f"  Etiquetas locales CSV: {args.etiquetas_local_csv}")
    print(f"  Ventana: {args.ventana}s | Overlap: {args.overlap*100:.0f}%")
    print(f"  Normalizacion: {args.norm}")
    print(f"  Repr temporal: {args.repr_temporal}")
    print(f"  Repr imagen: {args.repr_imagen}")
    print(f"  Rama: {args.rama}")
    print("="*60)

    # Cargar datos
    datos = []
    etiqueta_local = SANO if args.etiqueta_local == "sano" else PARKINSON

    if args.fuente in ("local", "ambos"):
        if os.path.isdir(args.sesiones):
            if args.etiquetas_local_csv:
                if not os.path.exists(args.etiquetas_local_csv):
                    print(f"\n[ERROR] CSV de etiquetas locales no encontrado: {args.etiquetas_local_csv}")
                    sys.exit(1)
                etiquetas_por_sello = cargar_etiquetas_locales_csv(args.etiquetas_local_csv)
                datos_locales = cargar_datos_locales_con_mapa(
                    args.sesiones,
                    etiquetas_por_sello=etiquetas_por_sello,
                    etiqueta_default=etiqueta_local,
                )
                print(
                    f"\nDatos locales: {len(datos_locales)} sesiones cargadas "
                    f"(CSV={len(etiquetas_por_sello)} etiquetas; default={etiqueta_local})"
                )
            else:
                datos_locales = cargar_datos_locales(args.sesiones, etiqueta=etiqueta_local)
                print(f"\nDatos locales: {len(datos_locales)} sesiones cargadas (etiqueta={etiqueta_local})")
            datos.extend(datos_locales)
        else:
            print(f"\n[WARN] Carpeta de sesiones no encontrada: {args.sesiones}")

    if args.fuente in ("publico", "ambos"):
        if os.path.isdir(args.dataset):
            datos_publicos = cargar_datos_publicos(args.dataset)
            print(f"\nDatos publicos: {len(datos_publicos)} grabaciones cargadas")
            datos.extend(datos_publicos)
        else:
            print(f"\n[WARN] Dataset publico no encontrado: {args.dataset}")
            print(f"  Ejecutar primero: python -m src.preprocesamiento.dataset_publico")

    if not datos:
        print("\n[ERROR] No se cargaron datos. Verificar rutas y fuentes.")
        sys.exit(1)

    # Ejecutar Rama A
    if args.rama in ("a", "ambas"):
        normalizaciones = ["minmax", "zscore"] if args.norm == "ambos" else [args.norm]
        for tipo_norm in normalizaciones:
            pipeline_rama_a(
                datos, args.ventana, args.overlap, tipo_norm,
                repr_temporal=args.repr_temporal,
                destino=args.salida,
            )

    # Ejecutar Rama B
    if args.rama in ("b", "ambas"):
        pipeline_rama_b(
            datos, args.ventana, args.overlap,
            repr_imagen=args.repr_imagen,
            destino=args.salida,
            gaf_size=args.gaf_size,
            balancear_splits=args.balancear_splits,
        )

    t_total = time.time() - t_inicio
    print(f"\n{'='*60}")
    print(f"Pipeline completado en {t_total:.1f}s")
    print(f"Resultados en: {args.salida}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
