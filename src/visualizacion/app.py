"""
Aplicacion principal de visualizacion EEG en tiempo real.

Conecta el parser TGAM con la interfaz grafica PyQtGraph.
Soporta dos modos:
  - Dispositivo real: lee datos del puerto COM
  - Demo: genera datos simulados para pruebas sin hardware

Uso:
    python -m src.visualizacion.app              (dispositivo real)
    python -m src.visualizacion.app --demo        (datos simulados)
    python -m src.visualizacion.app COM5           (puerto especifico)
"""

import sys
import os
import time
import threading
from collections import deque

# Asegurar que el directorio raiz del proyecto este en el path
_dir_proyecto = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _dir_proyecto not in sys.path:
    sys.path.insert(0, _dir_proyecto)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from src.parser.tgam_parser import TGAMParser, DatosTGAM
from src.visualizacion.buffer_datos import BufferEEG
from src.visualizacion.graficas import VentanaEEG


class MonitorFlujo:
    """Calcula tasas de llegada en ventana movil para diagnostico en vivo."""

    def __init__(self, ventana_seg=5.0):
        self._lock = threading.Lock()
        self._ventana_seg = float(ventana_seg)
        self._ts_paquetes = deque()
        self._ts_raw = deque()
        self._ts_metricas = deque()
        self.total_paquetes = 0
        self.total_raw = 0
        self.total_metricas = 0

    def _podar(self, ahora):
        limite = ahora - self._ventana_seg
        for cola in (self._ts_paquetes, self._ts_raw, self._ts_metricas):
            while cola and cola[0] < limite:
                cola.popleft()

    def registrar_paquete(self, datos: DatosTGAM, timestamp=None):
        """Registra un paquete TGAM valido (modo real)."""
        if timestamp is None:
            timestamp = time.time()
        with self._lock:
            self.total_paquetes += 1
            self._ts_paquetes.append(timestamp)
            if datos.actualizo_raw_wave:
                self.total_raw += 1
                self._ts_raw.append(timestamp)
            if datos.actualizo_metricas_lentas:
                self.total_metricas += 1
                self._ts_metricas.append(timestamp)
            self._podar(timestamp)

    def registrar_demo(self, raw=False, metricas=False, timestamp=None):
        """Registra eventos sinteticos en modo demo."""
        if timestamp is None:
            timestamp = time.time()
        with self._lock:
            if raw or metricas:
                self.total_paquetes += 1
                self._ts_paquetes.append(timestamp)
            if raw:
                self.total_raw += 1
                self._ts_raw.append(timestamp)
            if metricas:
                self.total_metricas += 1
                self._ts_metricas.append(timestamp)
            self._podar(timestamp)

    def snapshot(self):
        """Retorna tasas actuales y contadores acumulados."""
        with self._lock:
            ahora = time.time()
            self._podar(ahora)
            return {
                "raw_hz": len(self._ts_raw) / self._ventana_seg,
                "metricas_hz": len(self._ts_metricas) / self._ventana_seg,
                "paquetes_hz": len(self._ts_paquetes) / self._ventana_seg,
                "total_raw": self.total_raw,
                "total_metricas": self.total_metricas,
                "total_paquetes": self.total_paquetes,
            }


class LectorVisualizacion:
    """Conecta el parser TGAM al buffer de visualizacion y al grabador CSV."""

    def __init__(self, buffer: BufferEEG, monitor: MonitorFlujo, ventana=None):
        self.buffer = buffer
        self.monitor = monitor
        self.ventana = ventana  # VentanaEEG; puede ser None en modo headless

    def on_datos(self, datos: DatosTGAM):
        """Callback invocado por el parser con cada paquete."""
        t = time.time()
        self.monitor.registrar_paquete(datos, timestamp=t)
        grabador = self.ventana.grabador if self.ventana is not None else None
        if datos.actualizo_raw_wave:
            self.buffer.agregar_raw(datos.raw_wave, timestamp=t)
            if grabador is not None:
                grabador.agregar_raw(datos.raw_wave, t, datos.poor_signal)
        if datos.actualizo_metricas_lentas:
            self.buffer.agregar_bandas(datos, timestamp=t)
            if grabador is not None:
                grabador.agregar_bandas(datos, t)


def hilo_lectura_serial(puerto, parser, detener_evento):
    """Hilo dedicado a leer datos del puerto serial."""
    from src.conexion.serial_manager import conectar, desconectar

    ser = conectar(puerto)
    if ser is None:
        print("ERROR: No se pudo conectar al dispositivo.")
        print("Ejecuta con --demo para usar datos simulados.")
        return

    try:
        while not detener_evento.is_set():
            if ser.in_waiting > 0:
                datos_raw = ser.read(ser.in_waiting)
                for byte in datos_raw:
                    parser.parse_byte(byte)
            else:
                time.sleep(0.001)
    except Exception as e:
        print(f"Error en lectura serial: {e}")
    finally:
        desconectar(ser)


def hilo_demo(buffer, monitor, detener_evento, ventana=None):
    """Genera datos EEG simulados realistas para pruebas."""
    import numpy as np

    t_inicio = time.time()
    fs_raw = 512
    dt_raw = 1.0 / fs_raw
    muestra = 0

    while not detener_evento.is_set():
        ahora = time.time()
        t = ahora - t_inicio  # tiempo relativo para calculos de onda

        # Simular raw wave: suma de ondas tipicas EEG
        raw = (
            80 * np.sin(2 * np.pi * 1.5 * t)     # delta
            + 40 * np.sin(2 * np.pi * 5 * t)      # theta
            + 30 * np.sin(2 * np.pi * 10 * t)     # alpha
            + 15 * np.sin(2 * np.pi * 20 * t)     # beta
            + 8 * np.sin(2 * np.pi * 35 * t)      # gamma
            + np.random.normal(0, 25)              # ruido
        )
        raw_int = int(np.clip(raw, -500, 500))
        buffer.agregar_raw(raw_int, timestamp=ahora)
        monitor.registrar_demo(raw=True, timestamp=ahora)
        grabador = ventana.grabador if ventana is not None else None
        if grabador is not None:
            grabador.agregar_raw(raw_int, ahora, poor_signal=0)

        muestra += 1

        # Cada ~512 muestras (~1 seg), generar bandas y eSense
        if muestra % fs_raw == 0:
            datos = DatosTGAM()
            # Bandas con variacion realista
            base_delta = 300000 + 200000 * np.sin(0.1 * t) + np.random.normal(0, 50000)
            datos.delta = max(0, int(base_delta))
            datos.theta = max(0, int(50000 + 30000 * np.sin(0.15 * t + 1) + np.random.normal(0, 10000)))
            datos.low_alpha = max(0, int(40000 + 25000 * np.sin(0.2 * t + 2) + np.random.normal(0, 8000)))
            datos.high_alpha = max(0, int(20000 + 15000 * np.sin(0.18 * t + 3) + np.random.normal(0, 5000)))
            datos.low_beta = max(0, int(15000 + 10000 * np.sin(0.25 * t + 4) + np.random.normal(0, 4000)))
            datos.high_beta = max(0, int(10000 + 8000 * np.sin(0.22 * t + 5) + np.random.normal(0, 3000)))
            datos.low_gamma = max(0, int(5000 + 3000 * np.sin(0.3 * t + 6) + np.random.normal(0, 1500)))
            datos.high_gamma = max(0, int(3000 + 2000 * np.sin(0.28 * t + 7) + np.random.normal(0, 1000)))
            # eSense con variacion lenta
            datos.attention = int(np.clip(50 + 30 * np.sin(0.08 * t) + np.random.normal(0, 5), 0, 100))
            datos.meditation = int(np.clip(60 + 25 * np.sin(0.06 * t + 1) + np.random.normal(0, 5), 0, 100))
            datos.poor_signal = 0
            buffer.agregar_bandas(datos, timestamp=ahora)
            monitor.registrar_demo(metricas=True, timestamp=ahora)
            if grabador is not None:
                grabador.agregar_bandas(datos, ahora)

        # Mantener la tasa de 512 Hz
        t_esperado = t_inicio + muestra * dt_raw
        espera = t_esperado - time.time()
        if espera > 0:
            time.sleep(espera)


def main():
    puerto = None
    modo_demo = False

    for arg in sys.argv[1:]:
        if arg == "--demo":
            modo_demo = True
        elif arg.upper().startswith("COM") or arg.startswith("/dev/"):
            puerto = arg

    # Crear buffer y componentes
    # fs_bandas mayor a 1 para tolerar variaciones reales de frecuencia de paquetes lentos.
    buffer = BufferEEG(duracion_seg=180, fs_raw=512, fs_bandas=8)
    monitor = MonitorFlujo(ventana_seg=5.0)
    detener = threading.Event()

    # Iniciar Qt
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Tema oscuro
    from PyQt6.QtGui import QPalette, QColor as QC
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QC(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QC(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base, QC(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QC(53, 53, 53))
    palette.setColor(QPalette.ColorRole.Text, QC(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button, QC(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QC(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Highlight, QC(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QC(0, 0, 0))
    app.setPalette(palette)

    puerto_str = puerto if puerto else "autodetectado"
    ventana = VentanaEEG(buffer, monitor=monitor, ventana_seg=30,
                         puerto_com=puerto_str)
    ventana.show()

    # Iniciar hilo de datos
    if modo_demo:
        print("Modo DEMO: generando datos simulados...")
        hilo = threading.Thread(target=hilo_demo, args=(buffer, monitor, detener, ventana), daemon=True)
    else:
        lector = LectorVisualizacion(buffer, monitor, ventana=ventana)
        parser = TGAMParser(on_datos=lector.on_datos)
        hilo = threading.Thread(
            target=hilo_lectura_serial,
            args=(puerto, parser, detener),
            daemon=True,
        )
    hilo.start()

    # Ejecutar loop de Qt
    exit_code = app.exec()

    # Limpiar
    detener.set()
    hilo.join(timeout=2)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
