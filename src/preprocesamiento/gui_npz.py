"""
GUI para inspeccionar archivos .npz del proyecto.

Permite:
- abrir grabaciones publicas procesadas (`raw_fp1`)
- abrir datasets procesados de Rama A (`X_train`, etc.)
- abrir datasets procesados de Rama B (GAF)
- visualizar serie temporal y GAF dentro de la misma ventana
- ver claves, shapes, dtypes y rangos
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .gaf import generar_gaf


def _cargar_npz(ruta: str):
    return np.load(ruta, allow_pickle=True)


def _label_to_text(y_val) -> str:
    return "parkinson" if int(y_val) == 1 else "sano"


def _detectar_tipo_npz(datos) -> str:
    claves = set(datos.files)
    if "raw_fp1" in claves:
        return "publico"
    for split in ("train", "val", "test"):
        clave = f"X_{split}"
        if clave in claves:
            muestra = datos[clave][0]
            if muestra.ndim == 1:
                return "rama_a"
            if muestra.ndim == 2:
                return "rama_b"
    return "desconocido"


def _resumen_npz(datos) -> str:
    lineas = [f"Claves: {datos.files}", ""]
    for clave in datos.files:
        valor = datos[clave]
        shape = getattr(valor, "shape", None)
        dtype = getattr(valor, "dtype", None)
        lineas.append(f"- {clave}")
        lineas.append(f"  shape: {shape}")
        lineas.append(f"  dtype: {dtype}")
        if isinstance(valor, np.ndarray) and valor.size > 0 and np.issubdtype(valor.dtype, np.number):
            lineas.append(f"  min: {valor.min():.6f}")
            lineas.append(f"  max: {valor.max():.6f}")
            lineas.append(f"  mean: {valor.mean():.6f}")
        lineas.append("")
    return "\n".join(lineas)


class CanvasSeriesGAF(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(10, 5), tight_layout=True)
        super().__init__(self.fig)
        self.ax_serie = None
        self.ax_img = None

    def dibujar(self, serie, gaf, titulo: str):
        self.fig.clear()
        if serie is not None:
            ax1 = self.fig.add_subplot(1, 2, 1)
            ax2 = self.fig.add_subplot(1, 2, 2)
            ax1.plot(serie, linewidth=1.2, color="#1f77b4")
            ax1.set_title("Serie temporal")
            ax1.set_xlabel("Muestra")
            ax1.set_ylabel("Amplitud")
            ax1.grid(True, alpha=0.25)
            img_ax = ax2
        else:
            img_ax = self.fig.add_subplot(1, 1, 1)

        im = img_ax.imshow(gaf, cmap="RdBu_r", aspect="auto", origin="lower")
        img_ax.set_title("GAF")
        img_ax.set_xlabel("Tiempo j")
        img_ax.set_ylabel("Tiempo i")
        self.fig.colorbar(im, ax=img_ax, fraction=0.046, pad=0.04)
        self.fig.suptitle(titulo)
        self.draw()


class VentanaNPZ(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Inspector NPZ - EEG")
        self.resize(1400, 850)

        self._ruta_npz = ""
        self._ruta_serie_npz = ""
        self._datos = None
        self._tipo = "desconocido"
        self._subject_indices: dict[str, list[int]] = {}
        self._subject_list: list[str] = []

        self._crear_ui()
        self._actualizar_estado()

    def _crear_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Panel izquierdo
        left = QWidget()
        left_layout = QVBoxLayout(left)
        splitter.addWidget(left)

        grp_archivos = QGroupBox("Archivos")
        form_archivos = QGridLayout(grp_archivos)

        self.txt_npz = QLineEdit()
        self.txt_npz.setReadOnly(True)
        btn_npz = QPushButton("Abrir NPZ")
        btn_npz.clicked.connect(self._seleccionar_npz)

        self.txt_serie_npz = QLineEdit()
        self.txt_serie_npz.setReadOnly(True)
        btn_serie_npz = QPushButton("Serie NPZ")
        btn_serie_npz.clicked.connect(self._seleccionar_npz_serie)

        form_archivos.addWidget(QLabel("Archivo principal"), 0, 0)
        form_archivos.addWidget(self.txt_npz, 0, 1)
        form_archivos.addWidget(btn_npz, 0, 2)
        form_archivos.addWidget(QLabel("NPZ de serie opcional"), 1, 0)
        form_archivos.addWidget(self.txt_serie_npz, 1, 1)
        form_archivos.addWidget(btn_serie_npz, 1, 2)

        left_layout.addWidget(grp_archivos)

        grp_controles = QGroupBox("Controles")
        form = QFormLayout(grp_controles)

        self.lbl_tipo = QLabel("Sin archivo")

        self.cmb_split = QComboBox()
        self.cmb_split.addItems(["train", "val", "test"])
        self.cmb_split.currentIndexChanged.connect(self._refrescar_vista)

        self.cmb_sujeto = QComboBox()
        self.cmb_sujeto.currentIndexChanged.connect(self._on_subject_changed)

        self.spn_index = QSpinBox()
        self.spn_index.setRange(0, 0)
        self.spn_index.valueChanged.connect(self._refrescar_vista)

        self.btn_prev = QPushButton("Anterior")
        self.btn_prev.clicked.connect(lambda: self._mover_indice(-1))
        self.btn_next = QPushButton("Siguiente")
        self.btn_next.clicked.connect(lambda: self._mover_indice(1))

        idx_widget = QWidget()
        idx_layout = QHBoxLayout(idx_widget)
        idx_layout.setContentsMargins(0, 0, 0, 0)
        idx_layout.addWidget(self.spn_index)
        idx_layout.addWidget(self.btn_prev)
        idx_layout.addWidget(self.btn_next)

        self.spn_inicio = QDoubleSpinBox()
        self.spn_inicio.setRange(0.0, 100000.0)
        self.spn_inicio.setSingleStep(1.0)
        self.spn_inicio.valueChanged.connect(self._refrescar_vista)

        self.spn_duracion = QDoubleSpinBox()
        self.spn_duracion.setRange(0.5, 30.0)
        self.spn_duracion.setValue(2.0)
        self.spn_duracion.setSingleStep(0.5)
        self.spn_duracion.valueChanged.connect(self._refrescar_vista)

        self.spn_gaf = QSpinBox()
        self.spn_gaf.setRange(32, 512)
        self.spn_gaf.setSingleStep(32)
        self.spn_gaf.setValue(128)
        self.spn_gaf.valueChanged.connect(self._refrescar_vista)

        self.cmb_gaf_method = QComboBox()
        self.cmb_gaf_method.addItems(["summation", "difference"])
        self.cmb_gaf_method.currentIndexChanged.connect(self._refrescar_vista)

        self.lbl_ventanas_sujeto = QLabel("-")
        self.lbl_indice_global = QLabel("-")

        form.addRow("Tipo detectado", self.lbl_tipo)
        form.addRow("Split", self.cmb_split)
        form.addRow("Sujeto", self.cmb_sujeto)
        form.addRow("Indice (sujeto)", idx_widget)
        form.addRow("Ventanas sujeto", self.lbl_ventanas_sujeto)
        form.addRow("Indice global", self.lbl_indice_global)
        form.addRow("Inicio (s)", self.spn_inicio)
        form.addRow("Duracion (s)", self.spn_duracion)
        form.addRow("Tamano GAF", self.spn_gaf)
        form.addRow("Metodo GAF", self.cmb_gaf_method)

        left_layout.addWidget(grp_controles)

        grp_acciones = QGroupBox("Acciones")
        acciones = QHBoxLayout(grp_acciones)
        btn_refrescar = QPushButton("Procesar")
        btn_refrescar.clicked.connect(self._refrescar_vista)
        btn_guardar = QPushButton("Guardar figura")
        btn_guardar.clicked.connect(self._guardar_figura)
        acciones.addWidget(btn_refrescar)
        acciones.addWidget(btn_guardar)
        left_layout.addWidget(grp_acciones)

        grp_guia = QGroupBox("Guia")
        guia_layout = QVBoxLayout(grp_guia)
        self.txt_guia = QPlainTextEdit()
        self.txt_guia.setReadOnly(True)
        self.txt_guia.setPlainText(
            "Casos soportados:\n"
            "1. NPZ publico: muestra ventana raw + GAF calculado\n"
            "2. Rama A raw: muestra serie + GAF calculado\n"
            "3. Rama B GAF: muestra el GAF guardado\n"
            "4. Rama B + Serie NPZ: muestra el GAF guardado y la serie emparejada\n\n"
            "Tip:\n"
            "- Para Rama B, carga tambien el datos.npz de Rama A en 'NPZ de serie opcional' si quieres ver ambas cosas emparejadas."
        )
        guia_layout.addWidget(self.txt_guia)
        left_layout.addWidget(grp_guia)

        self.txt_resumen = QPlainTextEdit()
        self.txt_resumen.setReadOnly(True)
        left_layout.addWidget(self.txt_resumen, stretch=1)

        # Panel derecho
        right = QWidget()
        right_layout = QVBoxLayout(right)
        splitter.addWidget(right)

        self.lbl_titulo = QLabel("Carga un .npz para visualizarlo")
        self.lbl_titulo.setStyleSheet("font-size: 16px; font-weight: bold;")
        right_layout.addWidget(self.lbl_titulo)

        self.canvas = CanvasSeriesGAF()
        right_layout.addWidget(self.canvas, stretch=1)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _seleccionar_npz(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo NPZ",
            "",
            "NumPy NPZ (*.npz);;Todos los archivos (*.*)",
        )
        if ruta:
            self._ruta_npz = ruta
            self.txt_npz.setText(ruta)
            self._cargar_actual()

    def _seleccionar_npz_serie(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar NPZ de serie",
            "",
            "NumPy NPZ (*.npz);;Todos los archivos (*.*)",
        )
        if ruta:
            self._ruta_serie_npz = ruta
            self.txt_serie_npz.setText(ruta)
            self._refrescar_vista()

    def _cargar_actual(self):
        if not self._ruta_npz:
            return
        try:
            self._datos = _cargar_npz(self._ruta_npz)
            self._tipo = _detectar_tipo_npz(self._datos)
            self.txt_resumen.setPlainText(_resumen_npz(self._datos))
            self._actualizar_estado()
            self._refrescar_vista()
        except Exception as e:
            QMessageBox.critical(self, "Error al cargar NPZ", str(e))

    def _actualizar_estado(self):
        tipo_txt = {
            "publico": "Grabacion publica procesada",
            "rama_a": "Dataset Rama A",
            "rama_b": "Dataset Rama B",
            "desconocido": "Desconocido",
        }.get(self._tipo, self._tipo)
        self.lbl_tipo.setText(tipo_txt)

        es_publico = self._tipo == "publico"
        es_dataset = self._tipo in {"rama_a", "rama_b"}
        self.cmb_split.setEnabled(es_dataset)
        self.cmb_sujeto.setEnabled(es_dataset)
        self.spn_index.setEnabled(es_dataset)
        self.btn_prev.setEnabled(es_dataset)
        self.btn_next.setEnabled(es_dataset)
        self.spn_inicio.setEnabled(es_publico)
        self.spn_duracion.setEnabled(es_publico)
        self.spn_gaf.setEnabled(self._tipo in {"publico", "rama_a"})
        self.cmb_gaf_method.setEnabled(self._tipo in {"publico", "rama_a"})

        if self._datos is not None and es_dataset:
            self._actualizar_sujetos()
        else:
            self._subject_indices = {}
            self._subject_list = []
            self.cmb_sujeto.clear()
            self.lbl_ventanas_sujeto.setText("-")
            self.lbl_indice_global.setText("-")
            self.spn_index.setRange(0, 0)

    def _actualizar_sujetos(self):
        split = self.cmb_split.currentText()
        sujetos = [str(s) for s in self._datos[f"sujetos_{split}"]]
        indices: dict[str, list[int]] = {}
        for idx, sujeto in enumerate(sujetos):
            indices.setdefault(sujeto, []).append(idx)

        lista_sujetos = sorted(indices.keys())
        anterior = self.cmb_sujeto.currentText()

        if lista_sujetos != self._subject_list:
            self._subject_list = lista_sujetos
            self.cmb_sujeto.blockSignals(True)
            self.cmb_sujeto.clear()
            self.cmb_sujeto.addItems(lista_sujetos)
            if anterior in indices:
                self.cmb_sujeto.setCurrentText(anterior)
            elif lista_sujetos:
                self.cmb_sujeto.setCurrentIndex(0)
            self.cmb_sujeto.blockSignals(False)

        self._subject_indices = indices
        self._actualizar_rango_indice_sujeto()

    def _actualizar_rango_indice_sujeto(self):
        sujeto = self.cmb_sujeto.currentText()
        indices = self._subject_indices.get(sujeto, [])

        self.spn_index.blockSignals(True)
        if not indices:
            self.spn_index.setRange(0, 0)
            self.spn_index.setValue(0)
            self.lbl_ventanas_sujeto.setText("0")
            self.lbl_indice_global.setText("-")
        else:
            max_idx = len(indices) - 1
            self.spn_index.setRange(0, max_idx)
            if self.spn_index.value() > max_idx:
                self.spn_index.setValue(0)
            self.lbl_ventanas_sujeto.setText(str(len(indices)))
            global_idx = indices[self.spn_index.value()]
            self.lbl_indice_global.setText(str(global_idx))
        self.spn_index.blockSignals(False)

    def _resolver_indice_global(self) -> tuple[int, int, int, str]:
        sujeto = self.cmb_sujeto.currentText()
        indices = self._subject_indices.get(sujeto, [])
        if not indices:
            raise ValueError("No hay indices disponibles para el sujeto seleccionado.")
        local_idx = min(self.spn_index.value(), len(indices) - 1)
        global_idx = indices[local_idx]
        return global_idx, local_idx, len(indices), sujeto

    def _on_subject_changed(self):
        if self._tipo in {"rama_a", "rama_b"}:
            self._actualizar_rango_indice_sujeto()
        self._refrescar_vista()

    def _mover_indice(self, delta: int):
        if not self.spn_index.isEnabled():
            return
        nuevo = self.spn_index.value() + delta
        nuevo = max(self.spn_index.minimum(), min(self.spn_index.maximum(), nuevo))
        self.spn_index.setValue(nuevo)

    def _preparar_payload(self):
        if self._datos is None:
            raise ValueError("Primero debes cargar un archivo .npz.")

        if self._tipo == "publico":
            fs = int(self._datos["fs"])
            raw = self._datos["raw_fp1"].astype(np.float32)
            inicio = int(self.spn_inicio.value() * fs)
            n = int(self.spn_duracion.value() * fs)
            fin = min(len(raw), inicio + n)
            serie = raw[inicio:fin]
            if len(serie) < 2:
                raise ValueError("La ventana seleccionada es demasiado corta.")
            gaf = generar_gaf(
                serie,
                tamano=self.spn_gaf.value(),
                metodo=self.cmb_gaf_method.currentText(),
            )
            titulo = (
                f"{str(self._datos['sujeto'])}_{str(self._datos['sesion'])} | "
                f"{_label_to_text(self._datos['etiqueta'])}"
            )
            return serie, gaf, titulo

        split = self.cmb_split.currentText()
        index, index_local, total_sujeto, sujeto = self._resolver_indice_global()

        if self._tipo == "rama_a":
            serie = self._datos[f"X_{split}"][index].astype(np.float32)
            gaf = generar_gaf(
                serie,
                tamano=self.spn_gaf.value(),
                metodo=self.cmb_gaf_method.currentText(),
            )
            titulo = (
                f"{sujeto} | "
                f"{_label_to_text(self._datos[f'y_{split}'][index])} | "
                f"split={split} | idx={index_local}/{total_sujeto - 1} | global={index}"
            )
            return serie, gaf, titulo

        if self._tipo == "rama_b":
            serie = None
            if self._ruta_serie_npz:
                datos_serie = _cargar_npz(self._ruta_serie_npz)
                serie = datos_serie[f"X_{split}"][index].astype(np.float32)
            gaf = self._datos[f"X_{split}"][index].astype(np.float32)
            titulo = (
                f"{sujeto} | "
                f"{_label_to_text(self._datos[f'y_{split}'][index])} | "
                f"split={split} | idx={index_local}/{total_sujeto - 1} | global={index}"
            )
            return serie, gaf, titulo

        raise ValueError("Tipo de NPZ no soportado.")

    def _refrescar_vista(self):
        if self._datos is None:
            return
        try:
            self._actualizar_estado()
            serie, gaf, titulo = self._preparar_payload()
            self.lbl_titulo.setText(titulo)
            self.canvas.dibujar(serie, gaf, titulo)
        except Exception as e:
            QMessageBox.warning(self, "No se pudo procesar la vista", str(e))

    def _guardar_figura(self):
        if self._datos is None:
            QMessageBox.information(self, "Guardar figura", "Primero carga un archivo .npz.")
            return
        ruta, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar figura",
            "",
            "PNG (*.png);;JPEG (*.jpg);;PDF (*.pdf)",
        )
        if ruta:
            self.canvas.fig.savefig(ruta, dpi=150, bbox_inches="tight")
            QMessageBox.information(self, "Figura guardada", f"Se guardo en:\n{ruta}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ventana = VentanaNPZ()
    ventana.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
