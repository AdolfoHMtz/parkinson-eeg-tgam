"""
Grabador de sesiones EEG a CSV orientado a ML.

Genera dos archivos por sesion:
  - sesion_YYYY-MM-DD_HH-MM-SS_raw.csv    : raw EEG a 512 Hz
  - sesion_YYYY-MM-DD_HH-MM-SS_bandas.csv : bandas + eSense a ~1 Hz

Escritura incremental por bloques (sin acumular en RAM).
Solo guarda muestras con poor_signal < 50 (senal valida).

Etiquetas validas para sesiones de entrenamiento: "sano" o "parkinson".

Uso:
    grabador = GrabadorCSV(carpeta="datos/sesiones", duracion_seg=120)
    grabador.iniciar(puerto_com="COM5", version_sw="2.0", etiqueta="sano")
    grabador.agregar_raw(valor_raw, timestamp, poor_signal)
    grabador.agregar_bandas(datos_tgam, timestamp)
    grabador.detener()  # o se detiene automaticamente al vencer la duracion
"""

import csv
import os
import threading
import time
from datetime import datetime
from queue import Queue, Empty


# Version del software grabador (se escribe en el header del CSV)
VERSION_SW = "3.0"

# Tamano del bloque de flush (numero de filas antes de forzar escritura a disco)
BLOQUE_FLUSH = 100

# ID de la persona grabada. Todas las sesiones con el mismo sujeto se agrupan
# en el mismo split (train/val/test) para evitar fuga de datos.
SUJETO_DEFAULT = "local_01"

# Umbral de poor_signal para considerar la senal valida
UMBRAL_SENAL = 50

# Duraciones predefinidas en segundos
DURACIONES = {
    "30s": 30,
    "2min": 120,
    "5min": 300,
    "10min": 600,
    "personalizado": None,
}

# Etiquetas permitidas para sesiones de entrenamiento
ETIQUETAS_VALIDAS = {"sano", "parkinson"}


class GrabadorCSV:
    """
    Graba una sesion EEG en dos CSVs con escritura incremental thread-safe.

    El grabador corre un hilo de escritura dedicado que consume una cola
    de filas y las vuelca al disco en bloques, evitando que el hilo de
    lectura serial se bloquee por I/O.

    Atributos publicos:
        grabando (bool)    : True mientras la sesion esta activa.
        muestras_raw (int) : Filas escritas en el CSV raw.
        muestras_bandas (int): Filas escritas en el CSV de bandas.
        t_inicio (float)   : Epoch del inicio de la sesion (0 si no ha iniciado).
        duracion_seg (float|None): Duracion configurada, None = sin limite.
    """

    def __init__(self, carpeta="datos/sesiones", duracion_seg=None):
        """
        Args:
            carpeta: Ruta de la carpeta destino (se crea si no existe).
            duracion_seg: Duracion maxima en segundos. None = manual.
        """
        self.carpeta = carpeta
        self.duracion_seg = duracion_seg

        self.grabando = False
        self.muestras_raw = 0
        self.muestras_bandas = 0
        self.t_inicio = 0.0

        self._lock = threading.Lock()
        self._cola_raw: Queue = Queue()
        self._cola_bandas: Queue = Queue()
        self._detener = threading.Event()
        self._hilo_escritura: threading.Thread | None = None

        self._ruta_raw: str = ""
        self._ruta_bandas: str = ""
        self._ruta_etiquetas: str = os.path.join(self.carpeta, "etiquetas_locales.csv")
        self._sello: str = ""
        self._etiqueta_sesion: str = "sano"
        self._sujeto: str = SUJETO_DEFAULT

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def iniciar(
        self,
        puerto_com="desconocido",
        version_sw=VERSION_SW,
        etiqueta="sano",
        sujeto=SUJETO_DEFAULT,
    ):
        """
        Inicia la grabacion de una nueva sesion de entrenamiento.

        Crea los dos archivos CSV con sus headers y arranca el hilo de
        escritura. Si ya hay una sesion activa, lanza RuntimeError.

        Args:
            puerto_com: Puerto COM del dispositivo (se escribe en header).
            version_sw: Version del software (se escribe en header).
            etiqueta: Etiqueta ML de la sesion: "sano" o "parkinson".
            sujeto: ID de la persona grabada (agrupa sesiones al dividir datos).
        """
        with self._lock:
            if self.grabando:
                raise RuntimeError("Ya hay una sesion de grabacion activa.")

            etiqueta = str(etiqueta).strip().lower()
            if etiqueta not in ETIQUETAS_VALIDAS:
                raise ValueError(
                    f"Etiqueta invalida '{etiqueta}'. Usar: {', '.join(sorted(ETIQUETAS_VALIDAS))}."
                )

            os.makedirs(self.carpeta, exist_ok=True)
            ahora = datetime.now()
            sello = ahora.strftime("%Y-%m-%d_%H-%M-%S")
            fecha_str = ahora.strftime("%Y-%m-%d %H:%M:%S")
            self._sello = sello
            self._etiqueta_sesion = etiqueta
            self._sujeto = str(sujeto).strip() or SUJETO_DEFAULT
            self._ruta_raw = os.path.join(self.carpeta, f"sesion_{sello}_raw.csv")
            self._ruta_bandas = os.path.join(self.carpeta, f"sesion_{sello}_bandas.csv")

            self.t_inicio = time.time()
            self.muestras_raw = 0
            self.muestras_bandas = 0
            self._detener.clear()

            self._escribir_header_raw(fecha_str, puerto_com, version_sw, etiqueta)
            self._escribir_header_bandas(fecha_str, puerto_com, version_sw, etiqueta)
            self._registrar_etiqueta_sesion(fecha_str, puerto_com)

            # Arrancar hilo de escritura
            self._hilo_escritura = threading.Thread(
                target=self._hilo_escritura_fn, daemon=True
            )
            self._hilo_escritura.start()
            self.grabando = True

    def detener(self):
        """
        Detiene la grabacion y espera a que el hilo de escritura vacie las colas.

        Retorna la ruta de ambos archivos generados como tupla (raw, bandas).
        Si no habia sesion activa, retorna ("", "").
        """
        with self._lock:
            if not self.grabando:
                return ("", "")
            self.grabando = False
            self._detener.set()

        if self._hilo_escritura is not None:
            self._hilo_escritura.join(timeout=10)
            self._hilo_escritura = None

        return (self._ruta_raw, self._ruta_bandas)

    def agregar_raw(self, valor_raw: int, timestamp: float, poor_signal: int):
        """
        Encola una muestra raw EEG para escritura.

        Solo encola si la grabacion esta activa y poor_signal < UMBRAL_SENAL.
        Verifica automaticamente si la duracion configura ya se cumplio.

        Args:
            valor_raw:   Valor signed 16-bit del raw EEG.
            timestamp:   Epoch time de la muestra.
            poor_signal: Nivel de senal (0=buena, 200=sin contacto).
        """
        if not self.grabando:
            return
        if self._duracion_cumplida(timestamp):
            self.detener()
            return
        if poor_signal >= UMBRAL_SENAL:
            return
        self._cola_raw.put((timestamp, valor_raw, poor_signal))

    def agregar_bandas(self, datos, timestamp: float):
        """
        Encola un paquete de bandas EEG + eSense para escritura.

        Solo encola si la grabacion esta activa y poor_signal < UMBRAL_SENAL.

        Args:
            datos:     Objeto DatosTGAM con los campos de bandas y eSense.
            timestamp: Epoch time del paquete.
        """
        if not self.grabando:
            return
        if self._duracion_cumplida(timestamp):
            self.detener()
            return
        if datos.poor_signal >= UMBRAL_SENAL:
            return
        self._cola_bandas.put((
            timestamp,
            datos.delta, datos.theta,
            datos.low_alpha, datos.high_alpha,
            datos.low_beta, datos.high_beta,
            datos.low_gamma, datos.high_gamma,
            datos.attention, datos.meditation,
            datos.poor_signal,
        ))

    @property
    def tiempo_transcurrido(self) -> float:
        """Segundos desde el inicio de la sesion activa (0 si no hay sesion)."""
        if self.t_inicio == 0.0:
            return 0.0
        return time.time() - self.t_inicio

    @property
    def progreso(self) -> float:
        """Fraccion de progreso [0.0, 1.0]. 1.0 si no hay duracion configurada."""
        if self.duracion_seg is None or self.duracion_seg <= 0:
            return 1.0
        return min(self.tiempo_transcurrido / self.duracion_seg, 1.0)

    @property
    def ruta_raw(self) -> str:
        return self._ruta_raw

    @property
    def ruta_bandas(self) -> str:
        return self._ruta_bandas

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _duracion_cumplida(self, timestamp: float) -> bool:
        if self.duracion_seg is None:
            return False
        return (timestamp - self.t_inicio) >= self.duracion_seg

    def _escribir_header_raw(self, fecha: str, puerto: str, version: str, etiqueta: str):
        with open(self._ruta_raw, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["# EEG Raw - NeuroSky TGAM"])
            writer.writerow([f"# fecha={fecha}"])
            writer.writerow([f"# puerto_com={puerto}"])
            writer.writerow([f"# etiqueta_local={etiqueta}"])
            writer.writerow(["# fs_raw=512"])
            writer.writerow([f"# version_sw={version}"])
            writer.writerow([f"# umbral_senal={UMBRAL_SENAL}"])
            writer.writerow(["timestamp_epoch", "timestamp_hora", "raw_eeg", "poor_signal"])

    def _escribir_header_bandas(self, fecha: str, puerto: str, version: str, etiqueta: str):
        with open(self._ruta_bandas, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["# EEG Bandas + eSense - NeuroSky TGAM"])
            writer.writerow([f"# fecha={fecha}"])
            writer.writerow([f"# puerto_com={puerto}"])
            writer.writerow([f"# etiqueta_local={etiqueta}"])
            writer.writerow(["# fs_bandas=~1Hz"])
            writer.writerow([f"# version_sw={version}"])
            writer.writerow([f"# umbral_senal={UMBRAL_SENAL}"])
            writer.writerow([
                "timestamp_epoch", "timestamp_hora",
                "delta", "theta", "low_alpha", "high_alpha",
                "low_beta", "high_beta", "low_gamma", "high_gamma",
                "attention", "meditation", "poor_signal",
            ])

    def _registrar_etiqueta_sesion(self, fecha: str, puerto: str):
        """Registra la etiqueta de la sesion en un CSV central para el pipeline."""
        fila = [
            self._sello,
            self._etiqueta_sesion,
            fecha,
            puerto,
            os.path.basename(self._ruta_raw),
            os.path.basename(self._ruta_bandas),
            self._sujeto,
        ]
        encabezado = ["sello", "etiqueta", "fecha", "puerto_com", "archivo_raw", "archivo_bandas", "sujeto"]
        try:
            # Intentar creacion exclusiva: si tiene exito, el archivo es nuevo -> escribir header
            with open(self._ruta_etiquetas, "x", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(encabezado)
                writer.writerow(fila)
        except FileExistsError:
            with open(self._ruta_etiquetas, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(fila)

    def _hilo_escritura_fn(self):
        """Hilo dedicado: consume las colas y escribe en bloques."""
        with (
            open(self._ruta_raw, "a", newline="", encoding="utf-8") as f_raw,
            open(self._ruta_bandas, "a", newline="", encoding="utf-8") as f_bandas,
        ):
            writer_raw = csv.writer(f_raw)
            writer_bandas = csv.writer(f_bandas)
            pendiente_raw = 0
            pendiente_bandas = 0

            while not self._detener.is_set() or not self._cola_raw.empty() or not self._cola_bandas.empty():
                try:
                    while True:
                        ts, raw, ps = self._cola_raw.get_nowait()
                        hora = datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]
                        writer_raw.writerow((ts, hora, raw, ps))
                        self.muestras_raw += 1
                        pendiente_raw += 1
                except Empty:
                    pass

                try:
                    while True:
                        ts, d, th, la, ha, lb, hb, lg, hg, att, med, ps = self._cola_bandas.get_nowait()
                        hora = datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]
                        writer_bandas.writerow((ts, hora, d, th, la, ha, lb, hb, lg, hg, att, med, ps))
                        self.muestras_bandas += 1
                        pendiente_bandas += 1
                except Empty:
                    pass

                # Flush por bloque para no dejar datos en buffer de Python
                if pendiente_raw >= BLOQUE_FLUSH:
                    f_raw.flush()
                    pendiente_raw = 0
                if pendiente_bandas >= BLOQUE_FLUSH:
                    f_bandas.flush()
                    pendiente_bandas = 0

                if self._detener.is_set():
                    break
                time.sleep(0.05)

            # Flush final antes de cerrar
            f_raw.flush()
            f_bandas.flush()
