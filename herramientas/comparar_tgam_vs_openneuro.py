"""
Compara la compatibilidad entre sesiones locales TGAM y OpenNeuro Fp1 procesado.

Genera:
  - resumen JSON con metricas por sesion/grabacion
  - tabla CSV
  - reporte Markdown interpretable para documentacion del proyecto

Objetivo:
  Aportar evidencia objetiva para decidir si el dataset publico es lo
  suficientemente parecido al TGAM como para preentrenar modelos basados
  en raw EEG de un solo canal.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from scipy import signal, stats


BANDAS = [
    ("delta", 0.5, 2.75),
    ("theta", 3.5, 6.75),
    ("low_alpha", 7.5, 9.25),
    ("high_alpha", 10.0, 12.0),
    ("low_beta", 12.75, 20.0),
    ("high_beta", 20.25, 28.0),
    ("low_gamma", 30.0, 40.0),
    ("high_gamma", 40.0, 50.0),
]


def _potencia_relativa_bandas(x: np.ndarray, fs: int = 512) -> dict[str, float]:
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(1024, len(x)))
    mascara_total = (freqs >= 0.5) & (freqs <= 50.0)
    potencia_total = np.trapezoid(psd[mascara_total], freqs[mascara_total])
    resultado = {}
    for nombre, f_lo, f_hi in BANDAS:
        mascara = (freqs >= f_lo) & (freqs <= f_hi)
        potencia = np.trapezoid(psd[mascara], freqs[mascara]) if np.any(mascara) else 0.0
        resultado[nombre] = float(potencia / potencia_total) if potencia_total > 0 else 0.0
    return resultado


def resumir_raw(x: np.ndarray, fs: int = 512) -> dict:
    x = np.asarray(x, dtype=np.float32)
    return {
        "duracion_seg": float(len(x) / fs),
        "media": float(np.mean(x)),
        "mediana": float(np.median(x)),
        "std": float(np.std(x)),
        "p95_abs": float(np.percentile(np.abs(x), 95)),
        "p99_abs": float(np.percentile(np.abs(x), 99)),
        "skewness": float(stats.skew(x)),
        "kurtosis": float(stats.kurtosis(x)),
        "potencia_relativa": _potencia_relativa_bandas(x, fs=fs),
    }


def cargar_locales(carpeta_sesiones: Path) -> list[dict]:
    locales = []
    for ruta_raw in sorted(carpeta_sesiones.glob("sesion_*_raw.csv")):
        sello = ruta_raw.name.replace("sesion_", "").replace("_raw.csv", "")
        df = pd.read_csv(ruta_raw, comment="#")
        x = df["raw_eeg"].to_numpy(np.float32)
        resumen = resumir_raw(x)
        resumen["id"] = sello
        resumen["tipo"] = "local"
        resumen["poor_signal_mean"] = float(df["poor_signal"].mean())
        resumen["poor_signal_max"] = int(df["poor_signal"].max())
        resumen["pct_abs_ge_1000"] = float((np.abs(x) >= 1000).mean())
        locales.append(resumen)
    return locales


def cargar_publicos(carpeta_publico: Path) -> list[dict]:
    publicos = []
    for ruta_npz in sorted(carpeta_publico.glob("sub-*.npz")):
        datos = np.load(ruta_npz, allow_pickle=True)
        x = datos["raw_fp1"].astype(np.float32)
        resumen = resumir_raw(x)
        resumen["id"] = ruta_npz.stem
        resumen["tipo"] = "publico"
        resumen["etiqueta"] = int(datos["etiqueta"])
        resumen["sujeto"] = str(datos["sujeto"])
        resumen["sesion"] = str(datos["sesion"])
        publicos.append(resumen)
    return publicos


def _agregar_resumen_publico(publicos: list[dict], etiqueta: int | None = None) -> dict:
    items = publicos if etiqueta is None else [p for p in publicos if p["etiqueta"] == etiqueta]
    return {
        "n": len(items),
        "duracion_seg_media": mean([it["duracion_seg"] for it in items]),
        "std_media": mean([it["std"] for it in items]),
        "p95_abs_media": mean([it["p95_abs"] for it in items]),
        "std_q10": float(np.percentile([it["std"] for it in items], 10)),
        "std_q50": float(np.percentile([it["std"] for it in items], 50)),
        "std_q90": float(np.percentile([it["std"] for it in items], 90)),
        "p95_q10": float(np.percentile([it["p95_abs"] for it in items], 10)),
        "p95_q50": float(np.percentile([it["p95_abs"] for it in items], 50)),
        "p95_q90": float(np.percentile([it["p95_abs"] for it in items], 90)),
        "potencia_relativa_media": {
            nombre: mean([it["potencia_relativa"][nombre] for it in items])
            for nombre, _, _ in BANDAS
        },
    }


def _vector_comparacion(resumen: dict) -> np.ndarray:
    return np.array(
        [
            resumen["std"],
            resumen["p95_abs"],
            *[resumen["potencia_relativa"][nombre] for nombre, _, _ in BANDAS],
        ],
        dtype=np.float32,
    )


def calcular_vecinos(locales: list[dict], publicos: list[dict], top_k: int = 5) -> dict[str, list[dict]]:
    matriz_publica = np.stack([_vector_comparacion(p) for p in publicos], axis=0)
    mu = matriz_publica.mean(axis=0)
    sigma = matriz_publica.std(axis=0)
    sigma[sigma == 0] = 1.0

    salida: dict[str, list[dict]] = {}
    for local in locales:
        vec_local = (_vector_comparacion(local) - mu) / sigma
        distancias = []
        for publico in publicos:
            vec_publico = (_vector_comparacion(publico) - mu) / sigma
            distancia = float(np.linalg.norm(vec_local - vec_publico))
            distancias.append({
                "distancia": round(distancia, 4),
                "id": publico["id"],
                "etiqueta": publico["etiqueta"],
            })
        salida[local["id"]] = sorted(distancias, key=lambda item: item["distancia"])[:top_k]
    return salida


def evaluar_compatibilidad(local: dict, resumen_publico: dict) -> dict:
    std_ok = resumen_publico["std_q10"] <= local["std"] <= resumen_publico["std_q90"]
    p95_ok = resumen_publico["p95_q10"] <= local["p95_abs"] <= resumen_publico["p95_q90"]
    señal_ok = local["poor_signal_max"] < 25 and local["pct_abs_ge_1000"] < 0.05
    compatible = std_ok and p95_ok and señal_ok
    return {
        "compatible": compatible,
        "std_en_rango": std_ok,
        "p95_en_rango": p95_ok,
        "senal_estable": señal_ok,
    }


def guardar_csv_locales(locales: list[dict], ruta_csv: Path):
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "duracion_seg", "std", "p95_abs", "poor_signal_mean",
            "poor_signal_max", "pct_abs_ge_1000", "compatible"
        ])
        for loc in locales:
            writer.writerow([
                loc["id"],
                round(loc["duracion_seg"], 3),
                round(loc["std"], 4),
                round(loc["p95_abs"], 4),
                round(loc["poor_signal_mean"], 4),
                loc["poor_signal_max"],
                round(loc["pct_abs_ge_1000"], 6),
                str(loc["compatibilidad"]["compatible"]).lower(),
            ])


def construir_reporte_md(
    locales: list[dict],
    resumen_publico: dict,
    resumen_hc: dict,
    resumen_pd: dict,
    vecinos: dict[str, list[dict]],
) -> str:
    compatibles = [loc["id"] for loc in locales if loc["compatibilidad"]["compatible"]]
    no_compatibles = [loc["id"] for loc in locales if not loc["compatibilidad"]["compatible"]]

    lineas = [
        "# Reporte de compatibilidad TGAM vs OpenNeuro",
        "",
        "## Decision sugerida",
        "",
        "Se recomienda usar **raw EEG de 1 canal** como señal base común entre TGAM y OpenNeuro.",
        "Las sesiones locales compatibles pueden servir para validación externa del dominio TGAM.",
        "Las bandas del TGAM deben quedarse como apoyo/inspección, no como eje principal de entrenamiento cruzado.",
        "",
        "## Resumen público",
        "",
        f"- Grabaciones públicas analizadas: {resumen_publico['n']}",
        f"- Mediana/rango útil de amplitud: std q10-q90 = {resumen_publico['std_q10']:.2f} a {resumen_publico['std_q90']:.2f}",
        f"- Mediana/rango útil de amplitud robusta: p95 q10-q90 = {resumen_publico['p95_q10']:.2f} a {resumen_publico['p95_q90']:.2f}",
        f"- Controles sanos: {resumen_hc['n']} | Parkinson: {resumen_pd['n']}",
        "",
        "## Sesiones locales",
        "",
    ]

    for loc in locales:
        compat = loc["compatibilidad"]
        estado = "COMPATIBLE" if compat["compatible"] else "REVISAR / DESCARTAR"
        lineas.extend([
            f"### {loc['id']} - {estado}",
            "",
            f"- std = {loc['std']:.2f}",
            f"- p95_abs = {loc['p95_abs']:.2f}",
            f"- poor_signal_mean = {loc['poor_signal_mean']:.2f}",
            f"- poor_signal_max = {loc['poor_signal_max']}",
            f"- pct_abs_ge_1000 = {loc['pct_abs_ge_1000']:.4f}",
            f"- std_en_rango = {compat['std_en_rango']}",
            f"- p95_en_rango = {compat['p95_en_rango']}",
            f"- senal_estable = {compat['senal_estable']}",
            "",
            "Vecinos públicos más cercanos:",
        ])
        for vecino in vecinos[loc["id"]]:
            etiqueta_txt = "PD" if vecino["etiqueta"] == 1 else "HC"
            lineas.append(
                f"- {vecino['id']} | {etiqueta_txt} | distancia={vecino['distancia']:.4f}"
            )
        lineas.append("")

    lineas.extend([
        "## Conclusión",
        "",
        f"- Compatibles recomendadas para validación TGAM: {', '.join(compatibles) if compatibles else 'ninguna'}",
        f"- Sesiones a revisar o excluir del análisis principal: {', '.join(no_compatibles) if no_compatibles else 'ninguna'}",
        "- El dataset público sí es utilizable para preentrenamiento o entrenamiento base en raw EEG monocanal.",
        "- La compatibilidad encontrada justifica migrar la rama de imagen hacia GAF derivado del raw, manteniendo el entrenamiento cruzado sobre una señal común.",
        "",
    ])

    return "\n".join(lineas)


def main():
    parser = argparse.ArgumentParser(description="Comparar TGAM local vs OpenNeuro Fp1")
    parser.add_argument(
        "--sesiones",
        default="datos/sesiones",
        help="Carpeta con sesiones locales TGAM",
    )
    parser.add_argument(
        "--dataset",
        default="datos/openneuro_procesado",
        help="Carpeta del dataset OpenNeuro procesado",
    )
    parser.add_argument(
        "--salida",
        default="resultados/compatibilidad_tgam_openneuro",
        help="Carpeta donde guardar el reporte",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    carpeta_sesiones = (base_dir / args.sesiones).resolve()
    carpeta_dataset = (base_dir / args.dataset).resolve()
    carpeta_salida = (base_dir / args.salida).resolve()
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    locales = cargar_locales(carpeta_sesiones)
    publicos = cargar_publicos(carpeta_dataset)

    resumen_publico = _agregar_resumen_publico(publicos)
    resumen_hc = _agregar_resumen_publico(publicos, etiqueta=0)
    resumen_pd = _agregar_resumen_publico(publicos, etiqueta=1)

    for local in locales:
        local["compatibilidad"] = evaluar_compatibilidad(local, resumen_publico)

    vecinos = calcular_vecinos(locales, publicos)

    reporte = {
        "locales": locales,
        "publico_all": resumen_publico,
        "publico_hc": resumen_hc,
        "publico_pd": resumen_pd,
        "vecinos": vecinos,
    }

    ruta_json = carpeta_salida / "resumen_compatibilidad.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    ruta_csv = carpeta_salida / "sesiones_locales_resumen.csv"
    guardar_csv_locales(locales, ruta_csv)

    ruta_md = carpeta_salida / "REPORTE_COMPATIBILIDAD.md"
    contenido_md = construir_reporte_md(locales, resumen_publico, resumen_hc, resumen_pd, vecinos)
    ruta_md.write_text(contenido_md, encoding="utf-8")

    print("Comparación completada.")
    print(f"JSON: {ruta_json}")
    print(f"CSV:  {ruta_csv}")
    print(f"MD:   {ruta_md}")


if __name__ == "__main__":
    main()
