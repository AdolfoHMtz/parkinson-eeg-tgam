"""
Buffer circular thread-safe para almacenar datos EEG en tiempo real.

Mantiene un historial de tamano fijo para cada tipo de dato del TGAM,
permitiendo consultas rapidas para graficar.

Los timestamps se almacenan como epoch time (time.time()) para poder
mostrar la hora real en los ejes.
"""

import threading
import time
import numpy as np


class BufferEEG:
    """
    Almacena datos EEG en buffers circulares de NumPy.

    Dos tipos de buffer segun la frecuencia de muestreo:
    - raw_wave: 512 Hz (buffer grande)
    - bandas/esense: baja frecuencia (tipicamente ~1 Hz, puede variar)

    Todos los timestamps son epoch time (time.time()).
    """

    def __init__(self, duracion_seg=120, fs_raw=512, fs_bandas=8):
        self._lock = threading.Lock()
        self.duracion = duracion_seg
        self.fs_raw = fs_raw
        self.fs_bandas = fs_bandas

        # Buffer para raw wave (512 Hz)
        tam_raw = duracion_seg * fs_raw
        self.raw_wave = np.zeros(tam_raw, dtype=np.int16)
        self.raw_timestamps = np.zeros(tam_raw, dtype=np.float64)
        self.raw_idx = 0
        self.raw_count = 0

        # Buffer para bandas y eSense (baja frecuencia, variable)
        tam_bandas = duracion_seg * fs_bandas
        self.bandas = {
            "delta": np.zeros(tam_bandas, dtype=np.float64),
            "theta": np.zeros(tam_bandas, dtype=np.float64),
            "low_alpha": np.zeros(tam_bandas, dtype=np.float64),
            "high_alpha": np.zeros(tam_bandas, dtype=np.float64),
            "low_beta": np.zeros(tam_bandas, dtype=np.float64),
            "high_beta": np.zeros(tam_bandas, dtype=np.float64),
            "low_gamma": np.zeros(tam_bandas, dtype=np.float64),
            "high_gamma": np.zeros(tam_bandas, dtype=np.float64),
        }
        self.attention = np.zeros(tam_bandas, dtype=np.float64)
        self.meditation = np.zeros(tam_bandas, dtype=np.float64)
        self.poor_signal = np.zeros(tam_bandas, dtype=np.float64)
        self.bandas_timestamps = np.zeros(tam_bandas, dtype=np.float64)
        self.bandas_idx = 0
        self.bandas_count = 0

        self.t_inicio = time.time()

    def agregar_raw(self, valor, timestamp=None):
        """Agrega un valor raw_wave al buffer. timestamp es epoch time."""
        if timestamp is None:
            timestamp = time.time()
        with self._lock:
            tam = len(self.raw_wave)
            idx = self.raw_idx % tam
            self.raw_wave[idx] = valor
            self.raw_timestamps[idx] = timestamp
            self.raw_idx += 1
            self.raw_count = min(self.raw_count + 1, tam)

    def agregar_bandas(self, datos, timestamp=None):
        """Agrega bandas EEG + eSense al buffer. timestamp es epoch time."""
        if timestamp is None:
            timestamp = time.time()
        with self._lock:
            tam = len(self.attention)
            idx = self.bandas_idx % tam
            self.bandas["delta"][idx] = datos.delta
            self.bandas["theta"][idx] = datos.theta
            self.bandas["low_alpha"][idx] = datos.low_alpha
            self.bandas["high_alpha"][idx] = datos.high_alpha
            self.bandas["low_beta"][idx] = datos.low_beta
            self.bandas["high_beta"][idx] = datos.high_beta
            self.bandas["low_gamma"][idx] = datos.low_gamma
            self.bandas["high_gamma"][idx] = datos.high_gamma
            self.attention[idx] = datos.attention
            self.meditation[idx] = datos.meditation
            self.poor_signal[idx] = datos.poor_signal
            self.bandas_timestamps[idx] = timestamp
            self.bandas_idx += 1
            self.bandas_count = min(self.bandas_count + 1, tam)

    def obtener_raw(self, ultimos_seg=None):
        """Retorna (timestamps, valores) del raw wave. Timestamps son epoch."""
        with self._lock:
            if self.raw_count == 0:
                return np.array([]), np.array([])
            tam = len(self.raw_wave)
            n = self.raw_count
            if self.raw_idx >= tam:
                inicio = self.raw_idx % tam
                t = np.roll(self.raw_timestamps, -inicio)[:n]
                v = np.roll(self.raw_wave, -inicio)[:n]
            else:
                t = self.raw_timestamps[:n].copy()
                v = self.raw_wave[:n].copy()
            if ultimos_seg is not None and len(t) > 0:
                t_max = t[-1]
                mask = t >= (t_max - ultimos_seg)
                t = t[mask]
                v = v[mask]
            return t, v.astype(np.float64)

    def obtener_bandas(self, ultimos_seg=None):
        """Retorna (timestamps, dict_bandas, attention, meditation, poor_signal)."""
        with self._lock:
            if self.bandas_count == 0:
                vacio = np.array([])
                return vacio, {k: vacio for k in self.bandas}, vacio, vacio, vacio
            tam = len(self.attention)
            n = self.bandas_count
            if self.bandas_idx >= tam:
                inicio = self.bandas_idx % tam
                t = np.roll(self.bandas_timestamps, -inicio)[:n]
                bandas_out = {
                    k: np.roll(v, -inicio)[:n].copy()
                    for k, v in self.bandas.items()
                }
                att = np.roll(self.attention, -inicio)[:n].copy()
                med = np.roll(self.meditation, -inicio)[:n].copy()
                ps = np.roll(self.poor_signal, -inicio)[:n].copy()
            else:
                t = self.bandas_timestamps[:n].copy()
                bandas_out = {k: v[:n].copy() for k, v in self.bandas.items()}
                att = self.attention[:n].copy()
                med = self.meditation[:n].copy()
                ps = self.poor_signal[:n].copy()
            if ultimos_seg is not None and len(t) > 0:
                t_max = t[-1]
                mask = t >= (t_max - ultimos_seg)
                t = t[mask]
                bandas_out = {k: v[mask] for k, v in bandas_out.items()}
                att = att[mask]
                med = med[mask]
                ps = ps[mask]
            return t, bandas_out, att, med, ps

    @property
    def tiempo_transcurrido(self):
        return time.time() - self.t_inicio
