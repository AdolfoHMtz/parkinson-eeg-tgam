"""
Visualizacion en tiempo real de datos EEG del NeuroSky TGAM.

Replica la interfaz de MindView usando PyQtGraph + PyQt6:
- Panel superior: Raw Wave (eje derecho, semi-transparente) + 8 bandas apiladas (eje izquierdo)
- Panel inferior: Noise, Attention, Meditation (escala 0-100)
- Eje X muestra hora real (HH:MM:SS) como MindView
- Leyenda de colores, estadisticas en vivo
"""

import time
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QFrame, QProgressBar,
    QInputDialog, QMessageBox,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont, QColor

from src.visualizacion.buffer_datos import BufferEEG
from src.exportacion.grabador_csv import GrabadorCSV, DURACIONES


# Colores para las bandas EEG (similar a MindView)
COLORES_BANDAS = {
    "delta":      "#4A90D9",   # azul
    "theta":      "#7B68EE",   # violeta
    "low_alpha":  "#50C878",   # verde
    "high_alpha": "#90EE90",   # verde claro
    "low_beta":   "#FFD700",   # amarillo
    "high_beta":  "#FFA500",   # naranja
    "low_gamma":  "#FF6347",   # rojo-naranja
    "high_gamma": "#FF1493",   # rosa fuerte
}

COLOR_RAW = "#D2691E"          # marron/naranja para raw wave
COLOR_ATTENTION = "#FF6B6B"    # rojo para attention
COLOR_MEDITATION = "#6B9FFF"   # azul para meditation
COLOR_NOISE = "#FFD93D"        # amarillo para noise/poor_signal

NOMBRES_BANDAS_DISPLAY = {
    "delta": "Delta",
    "theta": "Theta",
    "low_alpha": "Low Alpha",
    "high_alpha": "High Alpha",
    "low_beta": "Low Beta",
    "high_beta": "High Beta",
    "low_gamma": "Low Gamma",
    "high_gamma": "High Gamma",
}

# Orden de apilado (de abajo hacia arriba, como MindView)
ORDEN_BANDAS = [
    "delta", "theta", "low_alpha", "high_alpha",
    "low_beta", "high_beta", "low_gamma", "high_gamma",
]


class EjeHoraReal(pg.AxisItem):
    """Eje X personalizado que muestra hora real (HH:MM:SS) en lugar de epoch."""

    def tickStrings(self, values, scale, spacing):
        result = []
        for v in values:
            try:
                if v > 946684800:  # epoch > 2000-01-01 = timestamp valido
                    result.append(time.strftime("%H:%M:%S", time.localtime(v)))
                else:
                    result.append("")
            except (OSError, ValueError, OverflowError):
                result.append("")
        return result


class VentanaEEG(QMainWindow):
    """Ventana principal de visualizacion EEG en tiempo real."""

    def __init__(self, buffer: BufferEEG, monitor=None, ventana_seg=30,
                 puerto_com="desconocido", carpeta_grabacion="datos/sesiones"):
        super().__init__()
        self.buffer = buffer
        self.monitor = monitor
        self.ventana_seg = ventana_seg
        self._puerto_com = puerto_com
        self._carpeta_grabacion = carpeta_grabacion
        self._pausado = False
        self._y_bandas = 1000.0
        self._fps_frames = 0
        self._fps_valor = 0.0
        self._fps_t0 = time.perf_counter()
        self._ultimo_t_bandas = 0.0

        self._grabador: GrabadorCSV | None = None
        self._etiqueta_grabacion_actual: str = "sano"
        self._cero_bandas: np.ndarray = np.zeros(1)

        self.setWindowTitle("MindView Python - NeuroSky TGAM")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        pg.setConfigOptions(antialias=False, useOpenGL=False)

        self._crear_ui()
        self._crear_graficas()
        self._crear_leyendas()

        # Timer de actualizacion (~30 FPS)
        self._timer = QTimer()
        self._timer.timeout.connect(self._actualizar)
        self._timer.start(33)

    def _crear_ui(self):
        """Crea la interfaz: controles superiores + area de graficas."""
        central = QWidget()
        self.setCentralWidget(central)
        layout_principal = QVBoxLayout(central)
        layout_principal.setContentsMargins(6, 6, 6, 6)
        layout_principal.setSpacing(4)

        # --- Barra de controles ---
        barra = QHBoxLayout()

        self._lbl_estado = QLabel("Esperando datos...")
        self._lbl_estado.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self._lbl_estado.setStyleSheet("color: #FFD93D; padding: 2px 8px;")
        barra.addWidget(self._lbl_estado)

        barra.addStretch()

        lbl_ventana = QLabel("Ventana:")
        lbl_ventana.setFont(QFont("Segoe UI", 9))
        barra.addWidget(lbl_ventana)

        self._combo_ventana = QComboBox()
        self._combo_ventana.addItems(["10 seg", "30 seg", "60 seg", "120 seg"])
        self._combo_ventana.setCurrentIndex(1)
        self._combo_ventana.currentIndexChanged.connect(self._cambiar_ventana)
        barra.addWidget(self._combo_ventana)

        self._btn_pausa = QPushButton("Pausar")
        self._btn_pausa.setCheckable(True)
        self._btn_pausa.clicked.connect(self._toggle_pausa)
        self._btn_pausa.setFixedWidth(80)
        barra.addWidget(self._btn_pausa)

        # --- Controles de grabacion ---
        sep_grab = QFrame()
        sep_grab.setFrameShape(QFrame.Shape.VLine)
        sep_grab.setStyleSheet("color: #555;")
        barra.addWidget(sep_grab)

        lbl_dur = QLabel("Duracion:")
        lbl_dur.setFont(QFont("Segoe UI", 9))
        barra.addWidget(lbl_dur)

        self._combo_duracion = QComboBox()
        self._combo_duracion.addItems(["30s", "2min", "5min", "10min", "personalizado", "sin limite"])
        self._combo_duracion.setCurrentIndex(1)
        self._combo_duracion.setFixedWidth(110)
        barra.addWidget(self._combo_duracion)

        lbl_etiqueta = QLabel("Etiqueta:")
        lbl_etiqueta.setFont(QFont("Segoe UI", 9))
        barra.addWidget(lbl_etiqueta)

        self._combo_etiqueta = QComboBox()
        self._combo_etiqueta.addItems(["sano", "parkinson"])
        self._combo_etiqueta.setCurrentIndex(0)
        self._combo_etiqueta.setFixedWidth(110)
        barra.addWidget(self._combo_etiqueta)

        self._btn_grabar = QPushButton("Grabar")
        self._btn_grabar.setCheckable(True)
        self._btn_grabar.clicked.connect(self._toggle_grabacion)
        self._btn_grabar.setFixedWidth(80)
        self._btn_grabar.setStyleSheet("QPushButton { color: #D2691E; font-weight: bold; }"
                                        "QPushButton:checked { color: #FF4444; font-weight: bold; }")
        barra.addWidget(self._btn_grabar)

        # Indicador visual: barra de progreso de la sesion
        self._barra_grabacion = QProgressBar()
        self._barra_grabacion.setRange(0, 100)
        self._barra_grabacion.setValue(0)
        self._barra_grabacion.setFixedWidth(120)
        self._barra_grabacion.setVisible(False)
        self._barra_grabacion.setTextVisible(False)
        self._barra_grabacion.setStyleSheet(
            "QProgressBar { background: #333; border: 1px solid #555; border-radius: 3px; }"
            "QProgressBar::chunk { background: #FF4444; }"
        )
        barra.addWidget(self._barra_grabacion)

        self._lbl_grabacion = QLabel("")
        self._lbl_grabacion.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        self._lbl_grabacion.setStyleSheet("color: #FF4444;")
        self._lbl_grabacion.setVisible(False)
        barra.addWidget(self._lbl_grabacion)

        layout_principal.addLayout(barra)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #555;")
        layout_principal.addWidget(sep)

        self._layout_graficas = QVBoxLayout()
        self._layout_graficas.setSpacing(2)
        layout_principal.addLayout(self._layout_graficas, stretch=1)

        # --- Barra de estadisticas inferior ---
        self._barra_stats = QHBoxLayout()
        fuente_stats = QFont("Consolas", 9)
        self._lbl_paquetes = QLabel("Paquetes: 0")
        self._lbl_tiempo = QLabel("Tiempo: 0:00")
        self._lbl_signal = QLabel("Signal: --")
        self._lbl_att = QLabel("Att: --")
        self._lbl_med = QLabel("Med: --")
        self._lbl_rate_raw = QLabel("Raw/s: --")
        self._lbl_rate_bandas = QLabel("Band/s: --")
        self._lbl_rate_pkt = QLabel("Pkt/s: --")
        self._lbl_fps = QLabel("FPS: --")
        for lbl in [self._lbl_paquetes, self._lbl_tiempo, self._lbl_signal,
                    self._lbl_att, self._lbl_med, self._lbl_rate_raw,
                    self._lbl_rate_bandas, self._lbl_rate_pkt, self._lbl_fps]:
            lbl.setFont(fuente_stats)
            self._barra_stats.addWidget(lbl)
        self._barra_stats.addStretch()

        layout_principal.addLayout(self._barra_stats)

    def _crear_graficas(self):
        """Crea los dos paneles de graficas con PyQtGraph."""
        self._graphics_widget = pg.GraphicsLayoutWidget()
        self._graphics_widget.setBackground("#1E1E1E")
        self._layout_graficas.addWidget(self._graphics_widget)

        # ===== PANEL SUPERIOR: Raw Wave + Bandas apiladas =====
        # Usar eje X con hora real
        eje_hora_sup = EjeHoraReal(orientation="bottom")
        self._plot_superior = self._graphics_widget.addPlot(
            row=0, col=0, axisItems={"bottom": eje_hora_sup}
        )
        self._plot_superior.setLabel("left", "Potencia Bandas EEG")
        self._plot_superior.setLabel("bottom", "Hora")
        self._plot_superior.showGrid(x=True, y=True, alpha=0.3)
        self._plot_superior.setMouseEnabled(x=False, y=False)
        self._plot_superior.getAxis("left").setStyle(tickFont=QFont("Consolas", 8))
        eje_hora_sup.setStyle(tickFont=QFont("Consolas", 8))
        eje_hora_sup.enableAutoSIPrefix(False)

        # Eje derecho para raw wave
        self._eje_raw = pg.ViewBox()
        self._plot_superior.scene().addItem(self._eje_raw)
        self._plot_superior.getAxis("right").linkToView(self._eje_raw)
        self._eje_raw.setXLink(self._plot_superior)
        self._plot_superior.showAxis("right")
        self._plot_superior.setLabel("right", "Raw Wave", color=COLOR_RAW)
        self._plot_superior.getAxis("right").setStyle(tickFont=QFont("Consolas", 8))
        self._plot_superior.getAxis("right").enableAutoSIPrefix(False)

        self._plot_superior.vb.sigResized.connect(self._sync_eje_raw)

        # Crear fill items para bandas apiladas
        self._fills_bandas = {}
        self._curves_bandas = {}
        self._bases_bandas = {}
        for nombre in ORDEN_BANDAS:
            color = QColor(COLORES_BANDAS[nombre])
            color_fill = QColor(COLORES_BANDAS[nombre])
            color_fill.setAlpha(150)
            curve = pg.PlotCurveItem(pen=pg.mkPen(color, width=1))
            base = pg.PlotCurveItem([0], [0])
            fill = pg.FillBetweenItem(curve, base, brush=color_fill)
            self._plot_superior.addItem(fill)
            self._plot_superior.addItem(curve)
            self._fills_bandas[nombre] = fill
            self._curves_bandas[nombre] = curve
            self._bases_bandas[nombre] = base

        # Curva de raw wave — semi-transparente para no tapar las bandas
        raw_color = QColor(COLOR_RAW)
        raw_color.setAlpha(120)
        self._curve_raw = pg.PlotCurveItem(
            pen=pg.mkPen(raw_color, width=1),
        )
        self._eje_raw.addItem(self._curve_raw)

        # ===== PANEL INFERIOR: Noise + Attention + Meditation =====
        eje_hora_inf = EjeHoraReal(orientation="bottom")
        self._plot_inferior = self._graphics_widget.addPlot(
            row=1, col=0, axisItems={"bottom": eje_hora_inf}
        )
        self._plot_inferior.setLabel("left", "Valor (0-100)")
        self._plot_inferior.setLabel("bottom", "Hora")
        self._plot_inferior.showGrid(x=True, y=True, alpha=0.3)
        self._plot_inferior.setYRange(0, 105)
        self._plot_inferior.setMouseEnabled(x=False, y=False)
        self._plot_inferior.getAxis("left").setStyle(tickFont=QFont("Consolas", 8))
        eje_hora_inf.setStyle(tickFont=QFont("Consolas", 8))
        eje_hora_inf.enableAutoSIPrefix(False)
        # Sincronizar eje X entre ambos paneles
        self._plot_inferior.setXLink(self._plot_superior)

        # Proporciones: panel superior 3x mas grande
        self._graphics_widget.ci.layout.setRowStretchFactor(0, 3)
        self._graphics_widget.ci.layout.setRowStretchFactor(1, 1)

        # Areas para el panel inferior - orden: noise detras, luego meditation, luego attention
        def _hacer_fill(color_hex, ancho_pen, alpha):
            brush = QColor(color_hex)
            brush.setAlpha(alpha)
            curve = pg.PlotCurveItem(pen=pg.mkPen(color_hex, width=ancho_pen))
            base = pg.PlotCurveItem([0], [0])
            area = pg.FillBetweenItem(curve, base, brush=brush)
            self._plot_inferior.addItem(area)
            self._plot_inferior.addItem(curve)
            return curve, base, area

        self._fill_noise, self._fill_noise_base, self._fill_noise_area = _hacer_fill(COLOR_NOISE, 1.5, 80)
        self._fill_med, self._fill_med_base, self._fill_med_area = _hacer_fill(COLOR_MEDITATION, 2, 100)
        self._fill_att, self._fill_att_base, self._fill_att_area = _hacer_fill(COLOR_ATTENTION, 2, 100)

    def _crear_leyendas(self):
        """Crea leyendas de color para ambos paneles."""
        leyenda_sup = self._plot_superior.addLegend(
            offset=(10, 10), labelTextSize="8pt",
        )
        for nombre in ORDEN_BANDAS:
            color = COLORES_BANDAS[nombre]
            leyenda_sup.addItem(
                pg.PlotDataItem(pen=pg.mkPen(color, width=3)),
                NOMBRES_BANDAS_DISPLAY[nombre],
            )
        leyenda_sup.addItem(
            pg.PlotDataItem(pen=pg.mkPen(COLOR_RAW, width=3)),
            "Raw Wave",
        )

        leyenda_inf = self._plot_inferior.addLegend(
            offset=(10, 10), labelTextSize="8pt",
        )
        leyenda_inf.addItem(
            pg.PlotDataItem(pen=pg.mkPen(COLOR_NOISE, width=3)), "Noise"
        )
        leyenda_inf.addItem(
            pg.PlotDataItem(pen=pg.mkPen(COLOR_ATTENTION, width=3)), "Attention"
        )
        leyenda_inf.addItem(
            pg.PlotDataItem(pen=pg.mkPen(COLOR_MEDITATION, width=3)), "Meditation"
        )

    @property
    def grabador(self) -> "GrabadorCSV | None":
        """Retorna el GrabadorCSV activo, o None si no hay sesion en curso."""
        return self._grabador if (self._grabador and self._grabador.grabando) else None

    def _sync_eje_raw(self):
        """Sincroniza la geometria del ViewBox del eje derecho."""
        self._eje_raw.setGeometry(self._plot_superior.vb.sceneBoundingRect())

    def _actualizar_fill(self, curva, base, area, t, datos, cero):
        curva.setData(t, datos)
        base.setData(t, cero)
        area.setCurves(curva, base)

    @staticmethod
    def _seg_a_mmss(segundos: float) -> str:
        t = int(segundos)
        return f"{t // 60}:{t % 60:02d}"

    def _cambiar_ventana(self, idx):
        opciones = [10, 30, 60, 120]
        self.ventana_seg = opciones[idx]

    def _toggle_pausa(self, checked):
        self._pausado = checked
        self._btn_pausa.setText("Reanudar" if checked else "Pausar")

    def _toggle_grabacion(self, checked):
        """Inicia o detiene la grabacion CSV al presionar el boton Grabar."""
        if checked:
            # Determinar duracion
            opcion = self._combo_duracion.currentText()
            if opcion == "sin limite":
                duracion = None
            elif opcion == "personalizado":
                valor, ok = QInputDialog.getInt(
                    self, "Duracion personalizada",
                    "Duracion en segundos:", 60, 10, 7200, 10
                )
                if not ok:
                    self._btn_grabar.setChecked(False)
                    return
                duracion = valor
            else:
                duracion = DURACIONES.get(opcion)

            etiqueta = self._combo_etiqueta.currentText()

            self._grabador = GrabadorCSV(
                carpeta=self._carpeta_grabacion,
                duracion_seg=duracion,
            )
            self._etiqueta_grabacion_actual = etiqueta
            try:
                self._grabador.iniciar(puerto_com=self._puerto_com, etiqueta=etiqueta)
            except RuntimeError as e:
                QMessageBox.warning(self, "Error de grabacion", str(e))
                self._btn_grabar.setChecked(False)
                self._grabador = None
                return
            except ValueError as e:
                QMessageBox.warning(self, "Etiqueta invalida", str(e))
                self._btn_grabar.setChecked(False)
                self._grabador = None
                return

            self._btn_grabar.setText("Detener")
            self._barra_grabacion.setVisible(True)
            self._lbl_grabacion.setVisible(True)
            self._combo_duracion.setEnabled(False)
            self._combo_etiqueta.setEnabled(False)
        else:
            self._detener_grabacion()

    def _detener_grabacion(self):
        """Detiene la grabacion y muestra los archivos generados."""
        if self._grabador is None:
            return
        ruta_raw, ruta_bandas = self._grabador.detener()
        n_raw = self._grabador.muestras_raw
        n_bandas = self._grabador.muestras_bandas
        self._grabador = None

        self._btn_grabar.setChecked(False)
        self._btn_grabar.setText("Grabar")
        self._barra_grabacion.setVisible(False)
        self._lbl_grabacion.setVisible(False)
        self._combo_duracion.setEnabled(True)
        self._combo_etiqueta.setEnabled(True)

        if ruta_raw:
            QMessageBox.information(
                self, "Grabacion completada",
                f"Sesion guardada:\n\n"
                f"Etiqueta: {self._etiqueta_grabacion_actual}\n\n"
                f"Raw ({n_raw} muestras):\n{ruta_raw}\n\n"
                f"Bandas ({n_bandas} muestras):\n{ruta_bandas}",
            )

    def _actualizar_indicador_grabacion(self):
        """Actualiza la barra de progreso y el label de tiempo de grabacion."""
        if self._grabador is None or not self._grabador.grabando:
            # La grabacion termino automaticamente (duracion cumplida)
            if self._btn_grabar.isChecked():
                self._detener_grabacion()
            return

        self._lbl_grabacion.setText(f"REC {self._seg_a_mmss(self._grabador.tiempo_transcurrido)}")

        progreso = self._grabador.progreso
        self._barra_grabacion.setValue(int(progreso * 100))

    def _actualizar(self):
        """Callback del timer: actualiza ambas graficas."""
        # Medir FPS de render para diagnosticar fluidez.
        self._fps_frames += 1
        t_fps = time.perf_counter()
        dt_fps = t_fps - self._fps_t0
        if dt_fps >= 1.0:
            self._fps_valor = self._fps_frames / dt_fps
            self._fps_frames = 0
            self._fps_t0 = t_fps

        # Actualizar indicador de grabacion (siempre, aunque este pausado)
        self._actualizar_indicador_grabacion()

        if self._pausado:
            return

        ventana = self.ventana_seg

        # --- Panel superior: Raw Wave ---
        t_raw, v_raw = self.buffer.obtener_raw(ultimos_seg=ventana)
        if len(t_raw) > 0:
            # Diezmar raw si hay demasiados puntos (>5000) para fluidez
            if len(t_raw) > 5000:
                paso = len(t_raw) // 5000
                t_raw_plot = t_raw[::paso]
                v_raw_plot = v_raw[::paso]
            else:
                t_raw_plot = t_raw
                v_raw_plot = v_raw

            self._curve_raw.setData(t_raw_plot, v_raw_plot)
            # Rango fijo para raw: -1500 a 1500 (como MindView)
            self._eje_raw.setRange(
                xRange=(t_raw[-1] - ventana, t_raw[-1]),
                yRange=(-1500, 1500),
                padding=0,
            )

        # --- Panel superior: Bandas apiladas ---
        t_b, bandas, att, med, ps = self.buffer.obtener_bandas(ultimos_seg=ventana)
        if len(t_b) > 1 and t_b[-1] != self._ultimo_t_bandas:
            self._ultimo_t_bandas = t_b[-1]

            # Reutilizar array de ceros para no allocar en cada frame (hot path 30fps)
            if len(self._cero_bandas) != len(t_b):
                self._cero_bandas = np.zeros(len(t_b))
            cero = self._cero_bandas

            # MindView muestra bandas superpuestas desde cero, no acumuladas.
            y_max = 0.0
            series = []
            for nombre in ORDEN_BANDAS:
                y = bandas[nombre]
                self._curves_bandas[nombre].setData(t_b, y)
                self._bases_bandas[nombre].setData(t_b, cero)
                self._fills_bandas[nombre].setCurves(
                    self._curves_bandas[nombre], self._bases_bandas[nombre]
                )
                if len(y) > 0:
                    y_max = max(y_max, float(np.max(y)))
                    series.append(y)

            # Autoescala robusta: evita recortar picos y baja suavemente para estabilidad visual.
            if series:
                y_concat = np.concatenate(series)
                y_p98 = float(np.percentile(y_concat, 98))
                y_obj = max(1000.0, y_max * 1.05, y_p98 * 1.35)
                if y_obj > self._y_bandas:
                    self._y_bandas = y_obj
                else:
                    self._y_bandas = (self._y_bandas * 0.92) + (y_obj * 0.08)
            self._plot_superior.setYRange(0, self._y_bandas)
            self._plot_superior.setXRange(t_b[-1] - ventana, t_b[-1])

            # --- Panel inferior: Noise, Attention, Meditation ---
            noise_norm = np.clip(ps / 2.0, 0, 100)
            self._actualizar_fill(self._fill_noise, self._fill_noise_base, self._fill_noise_area, t_b, noise_norm, cero)
            self._actualizar_fill(self._fill_att,   self._fill_att_base,   self._fill_att_area,   t_b, att,        cero)
            self._actualizar_fill(self._fill_med,   self._fill_med_base,   self._fill_med_area,   t_b, med,        cero)

            self._plot_inferior.setXRange(t_b[-1] - ventana, t_b[-1])

            # --- Actualizar estadisticas ---
            self._lbl_signal.setText(f"Signal: {int(ps[-1])}")
            self._lbl_att.setText(f"Att: {int(att[-1])}")
            self._lbl_med.setText(f"Med: {int(med[-1])}")

            if ps[-1] == 0:
                self._lbl_estado.setStyleSheet("color: #50C878; padding: 2px 8px;")
                self._lbl_estado.setText("Senal EXCELENTE")
            elif ps[-1] < 50:
                self._lbl_estado.setStyleSheet("color: #90EE90; padding: 2px 8px;")
                self._lbl_estado.setText("Senal BUENA")
            elif ps[-1] < 200:
                self._lbl_estado.setStyleSheet("color: #FFD93D; padding: 2px 8px;")
                self._lbl_estado.setText("Senal REGULAR")
            else:
                self._lbl_estado.setStyleSheet("color: #FF6347; padding: 2px 8px;")
                self._lbl_estado.setText("SIN CONTACTO")

        # Tiempo transcurrido y paquetes
        self._lbl_tiempo.setText(f"Tiempo: {self._seg_a_mmss(self.buffer.tiempo_transcurrido)}")
        self._lbl_paquetes.setText(
            f"Paquetes: {self.buffer.bandas_count} | Raw: {self.buffer.raw_count}"
        )
        self._lbl_fps.setText(f"FPS: {self._fps_valor:0.1f}")

        if self.monitor is not None:
            snap = self.monitor.snapshot()
            self._lbl_rate_raw.setText(f"Raw/s: {snap['raw_hz']:0.1f}")
            self._lbl_rate_bandas.setText(f"Band/s: {snap['metricas_hz']:0.1f}")
            self._lbl_rate_pkt.setText(f"Pkt/s: {snap['paquetes_hz']:0.1f}")
