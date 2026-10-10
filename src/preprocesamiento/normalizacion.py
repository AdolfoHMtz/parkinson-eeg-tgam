"""
Normalizacion de datos EEG.

Implementa dos estrategias:
  - MinMax: escala a [0, 1]
  - Z-score: media=0, std=1

Ambas se aplican por canal/feature. Los parametros de normalizacion
(min, max, media, std) se calculan SOLO sobre el set de entrenamiento
y se guardan para aplicar la misma transformacion a validacion/test.
"""

import numpy as np


class NormalizadorMinMax:
    """
    Escalado MinMax a [0, 1].

    Calcula min y max por columna sobre los datos de entrenamiento.
    Aplica la misma transformacion a datos nuevos.
    """

    def __init__(self):
        self.min_vals: np.ndarray | None = None
        self.max_vals: np.ndarray | None = None
        self._ajustado = False

    def ajustar(self, datos: np.ndarray):
        """
        Calcula min y max por columna.

        Args:
            datos: Array 2D (n_muestras, n_features) de entrenamiento.
        """
        if datos.ndim == 1:
            datos = datos.reshape(-1, 1)
        self.min_vals = np.nanmin(datos, axis=0)
        self.max_vals = np.nanmax(datos, axis=0)
        self._ajustado = True

    def transformar(self, datos: np.ndarray) -> np.ndarray:
        """
        Aplica la normalizacion MinMax.

        Args:
            datos: Array 2D (n_muestras, n_features).

        Returns:
            Array normalizado a [0, 1].
        """
        if not self._ajustado:
            raise RuntimeError("Llamar ajustar() primero con datos de entrenamiento.")
        es_1d = datos.ndim == 1
        if es_1d:
            datos = datos.reshape(-1, 1)

        rango = np.where(self.max_vals == self.min_vals, 1.0, self.max_vals - self.min_vals)
        resultado = (datos - self.min_vals) / rango

        return resultado.ravel() if es_1d else resultado

    def ajustar_transformar(self, datos: np.ndarray) -> np.ndarray:
        """Ajusta y transforma en un solo paso."""
        self.ajustar(datos)
        return self.transformar(datos)

    def parametros(self) -> dict:
        """Retorna los parametros de normalizacion para guardar."""
        return {
            "tipo": "minmax",
            "min": self.min_vals,
            "max": self.max_vals,
        }

    def cargar_parametros(self, params: dict):
        """Carga parametros previamente guardados."""
        self.min_vals = params["min"]
        self.max_vals = params["max"]
        self._ajustado = True


class NormalizadorZScore:
    """
    Estandarizacion Z-score (media=0, std=1).

    Calcula media y desviacion estandar por columna sobre entrenamiento.
    """

    def __init__(self):
        self.media: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self._ajustado = False

    def ajustar(self, datos: np.ndarray):
        """
        Calcula media y std por columna.

        Args:
            datos: Array 2D (n_muestras, n_features) de entrenamiento.
        """
        if datos.ndim == 1:
            datos = datos.reshape(-1, 1)
        self.media = np.nanmean(datos, axis=0)
        self.std = np.nanstd(datos, axis=0)
        self._ajustado = True

    def transformar(self, datos: np.ndarray) -> np.ndarray:
        """
        Aplica la normalizacion Z-score.

        Args:
            datos: Array 2D (n_muestras, n_features).

        Returns:
            Array normalizado (media=0, std=1).
        """
        if not self._ajustado:
            raise RuntimeError("Llamar ajustar() primero con datos de entrenamiento.")
        es_1d = datos.ndim == 1
        if es_1d:
            datos = datos.reshape(-1, 1)

        std = np.where(self.std == 0, 1.0, self.std)
        resultado = (datos - self.media) / std

        return resultado.ravel() if es_1d else resultado

    def ajustar_transformar(self, datos: np.ndarray) -> np.ndarray:
        """Ajusta y transforma en un solo paso."""
        self.ajustar(datos)
        return self.transformar(datos)

    def parametros(self) -> dict:
        """Retorna los parametros de normalizacion para guardar."""
        return {
            "tipo": "zscore",
            "media": self.media,
            "std": self.std,
        }

    def cargar_parametros(self, params: dict):
        """Carga parametros previamente guardados."""
        self.media = params["media"]
        self.std = params["std"]
        self._ajustado = True


def crear_normalizador(tipo: str):
    """
    Factory para crear normalizadores.

    Args:
        tipo: "minmax" o "zscore".

    Returns:
        Instancia de NormalizadorMinMax o NormalizadorZScore.
    """
    if tipo == "minmax":
        return NormalizadorMinMax()
    elif tipo == "zscore":
        return NormalizadorZScore()
    else:
        raise ValueError(f"Tipo de normalizacion desconocido: {tipo}. Usar 'minmax' o 'zscore'.")
