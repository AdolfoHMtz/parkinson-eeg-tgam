"""
Rebalanceo de particiones para la Rama B.

Toma un datos.npz ya generado y crea una nueva division train/val/test
repartiendo sujetos por clase de forma mas equilibrada.

Esto ayuda cuando el split original deja un conjunto de prueba sesgado y
hace que accuracy sea engañoso.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from .etiquetado import calcular_info_division, id_paciente


RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}


def cargar_datos(ruta_npz: str):
    datos = np.load(ruta_npz, allow_pickle=True)
    X = np.concatenate([datos["X_train"], datos["X_val"], datos["X_test"]], axis=0)
    y = np.concatenate([datos["y_train"], datos["y_val"], datos["y_test"]], axis=0)
    sujetos = np.concatenate([datos["sujetos_train"], datos["sujetos_val"], datos["sujetos_test"]], axis=0)
    # Agrupar por persona: archivos generados antes de la correccion traen
    # "pd11_off"/"pd11_on" como sujetos distintos (fuga de datos entre splits).
    sujetos = np.array([id_paciente(s) for s in sujetos])
    return X, y, sujetos


def construir_catalogo_sujetos(y: np.ndarray, sujetos: np.ndarray):
    catalogo: dict[str, dict[str, int]] = {}
    for subj in np.unique(sujetos):
        mask = sujetos == subj
        labels = y[mask]
        total = int(mask.sum())
        label = int(np.bincount(labels.astype(int)).argmax())
        catalogo[str(subj)] = {
            "label": label,
            "n_muestras": total,
        }
    return catalogo


def _score_split(current_windows: dict[str, int], current_subjects: dict[str, int], target_windows: dict[str, float], target_subjects: dict[str, float], split: str) -> float:
    w = (current_windows[split] + 1) / max(target_windows[split], 1.0)
    s = (current_subjects[split] + 1) / max(target_subjects[split], 1.0)
    return w + 0.35 * s


def asignar_sujetos_balanceado(catalogo: dict[str, dict[str, int]], seed: int = 42):
    rng = np.random.default_rng(seed)
    asignacion: dict[str, str] = {}

    por_clase: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for subj, info in catalogo.items():
        por_clase[info["label"]].append((subj, info["n_muestras"]))

    for label, sujetos_clase in por_clase.items():
        rng.shuffle(sujetos_clase)
        sujetos_clase.sort(key=lambda x: x[1], reverse=True)

        total_windows = sum(n for _, n in sujetos_clase)
        total_subjects = len(sujetos_clase)
        target_windows = {k: total_windows * v for k, v in RATIOS.items()}
        target_subjects = {k: total_subjects * v for k, v in RATIOS.items()}
        current_windows = {k: 0 for k in RATIOS}
        current_subjects = {k: 0 for k in RATIOS}

        for subj, n_muestras in sujetos_clase:
            mejor_split = min(
                RATIOS.keys(),
                key=lambda split: _score_split(
                    current_windows, current_subjects, target_windows, target_subjects, split
                ),
            )
            asignacion[subj] = mejor_split
            current_windows[mejor_split] += n_muestras
            current_subjects[mejor_split] += 1

        print(f"Clase {label}:")
        for split in RATIOS:
            print(
                f"  {split}: sujetos={current_subjects[split]}/{target_subjects[split]:.1f} "
                f"ventanas={current_windows[split]}/{target_windows[split]:.1f}"
            )

    return asignacion


def construir_division(X: np.ndarray, y: np.ndarray, sujetos: np.ndarray, asignacion: dict[str, str]):
    indices = {"train": [], "val": [], "test": []}
    for idx, subj in enumerate(sujetos):
        indices[asignacion[str(subj)]].append(idx)

    division = {}
    for split in ["train", "val", "test"]:
        idx = np.array(indices[split], dtype=int)
        division[f"X_{split}"] = X[idx]
        division[f"y_{split}"] = y[idx]
        division[f"sujetos_{split}"] = sujetos[idx]

    division["info"] = calcular_info_division(division)
    return division


def resumen_split(nombre: str, y: np.ndarray, sujetos: np.ndarray):
    clases, conteos = np.unique(y, return_counts=True)
    print(f"{nombre}:")
    print(f"  ventanas={len(y)} | sujetos={len(np.unique(sujetos))}")
    for clase, conteo in zip(clases, conteos):
        print(f"  clase {int(clase)}: {int(conteo)}")


def main():
    parser = argparse.ArgumentParser(description="Rebalancear split por sujeto para Rama B")
    parser.add_argument("entrada", help="Ruta al datos.npz original")
    parser.add_argument("--salida", required=True, help="Ruta de salida para el nuevo datos.npz")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    X, y, sujetos = cargar_datos(args.entrada)
    catalogo = construir_catalogo_sujetos(y, sujetos)
    asignacion = asignar_sujetos_balanceado(catalogo, seed=args.seed)
    division = construir_division(X, y, sujetos, asignacion)

    os.makedirs(os.path.dirname(args.salida), exist_ok=True)
    np.savez_compressed(args.salida, **{k: v for k, v in division.items() if k != "info"})

    metadata = {
        "entrada": args.entrada,
        "salida": args.salida,
        "ratios": RATIOS,
        "n_total": int(len(y)),
        "n_sujetos": int(len(np.unique(sujetos))),
        "agrupado_por": "paciente",
        "info_division": division["info"],
    }
    with open(os.path.splitext(args.salida)[0] + "_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nResumen final:")
    for split in ["train", "val", "test"]:
        resumen_split(split, division[f"y_{split}"], division[f"sujetos_{split}"])
    print(f"\nNuevo split guardado en: {args.salida}")


if __name__ == "__main__":
    main()
