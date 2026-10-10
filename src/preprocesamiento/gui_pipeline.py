"""
Interfaz grafica para ejecutar el pipeline de preprocesamiento EEG.

Objetivo:
- Evitar errores por comandos escritos a mano.
- Permitir seleccionar parametros con controles.
- Ejecutar el pipeline y ver logs en vivo.
- Copiar el comando generado para auditoria/reproducibilidad.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QProcess
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
    QCheckBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QPlainTextEdit,
)




class VentanaPreprocesamiento(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Preprocesamiento EEG - TGAM")
        self.resize(1100, 760)

        self.proyecto_dir = Path(__file__).resolve().parents[2]
        self.python_exe = sys.executable
        self.historial_path = self.proyecto_dir / "resultados" / "historial_preprocesamiento.json"
        self._comando_ultimo = ""
        self._inicio_ejecucion: datetime | None = None

        self.proceso = QProcess(self)
        self.proceso.readyReadStandardOutput.connect(self._on_stdout)
        self.proceso.readyReadStandardError.connect(self._on_stderr)
        self.proceso.finished.connect(self._on_finished)

        self._crear_ui()
        self._cargar_defaults()
        self._actualizar_comando()

    # -------------------------- UI --------------------------

    def _crear_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout_principal = QVBoxLayout(central)

        # Grupo: configuracion
        grp_conf = QGroupBox("Configuracion del pipeline")
        form = QFormLayout(grp_conf)

        self.cmb_fuente = QComboBox()
        self.cmb_fuente.addItems(["local", "publico", "ambos"])
        self.cmb_fuente.currentIndexChanged.connect(self._actualizar_comando)
        self.cmb_fuente.currentIndexChanged.connect(self._ajustar_estado_por_fuente)

        self.cmb_etiqueta_local = QComboBox()
        self.cmb_etiqueta_local.addItems(["sano", "parkinson"])
        self.cmb_etiqueta_local.currentIndexChanged.connect(self._actualizar_comando)

        self.spn_ventana = QDoubleSpinBox()
        self.spn_ventana.setRange(1.0, 10.0)
        self.spn_ventana.setSingleStep(1.0)
        self.spn_ventana.setDecimals(1)
        self.spn_ventana.valueChanged.connect(self._actualizar_comando)

        self.cmb_overlap = QComboBox()
        self.cmb_overlap.addItems(["0.5", "0.75"])
        self.cmb_overlap.currentIndexChanged.connect(self._actualizar_comando)

        self.cmb_rama = QComboBox()
        self.cmb_rama.addItems(["ambas", "a", "b"])
        self.cmb_rama.currentIndexChanged.connect(self._actualizar_comando)

        self.cmb_norm = QComboBox()
        self.cmb_norm.addItems(["ambos", "minmax", "zscore"])
        self.cmb_norm.currentIndexChanged.connect(self._actualizar_comando)

        self.cmb_repr_temporal = QComboBox()
        self.cmb_repr_temporal.addItems(["raw", "features"])
        self.cmb_repr_temporal.currentIndexChanged.connect(self._actualizar_comando)

        self.cmb_repr_imagen = QComboBox()
        self.cmb_repr_imagen.addItems(["gaf", "stft"])
        self.cmb_repr_imagen.currentIndexChanged.connect(self._actualizar_comando)
        self.cmb_repr_imagen.currentIndexChanged.connect(self._ajustar_estado_repr_imagen)

        self.cmb_gaf_size = QComboBox()
        self.cmb_gaf_size.addItems(["128", "256"])
        self.cmb_gaf_size.currentIndexChanged.connect(self._actualizar_comando)

        self.chk_balancear_splits = QCheckBox("Balancear splits de Rama B")
        self.chk_balancear_splits.setChecked(True)
        self.chk_balancear_splits.stateChanged.connect(self._actualizar_comando)

        form.addRow("Fuente", self.cmb_fuente)
        form.addRow("Etiqueta local", self.cmb_etiqueta_local)
        form.addRow("Ventana (seg)", self.spn_ventana)
        form.addRow("Overlap", self.cmb_overlap)
        form.addRow("Rama", self.cmb_rama)
        form.addRow("Normalizacion (Rama A)", self.cmb_norm)
        form.addRow("Repr temporal", self.cmb_repr_temporal)
        form.addRow("Repr imagen", self.cmb_repr_imagen)
        form.addRow("Tamano GAF", self.cmb_gaf_size)
        form.addRow("Balanceo Rama B", self.chk_balancear_splits)

        grp_guia = QGroupBox("Guia rapida")
        guia_layout = QVBoxLayout(grp_guia)
        self.txt_guia = QPlainTextEdit()
        self.txt_guia.setReadOnly(True)
        self.txt_guia.setMaximumHeight(140)
        self.txt_guia.setPlainText(
            "Flujo recomendado:\n"
            "1. Si vas a entrenar con OpenNeuro: fuente=publico\n"
            "2. Si quieres mezclar sesiones TGAM propias: fuente=ambos\n"
            "3. Repr temporal recomendada: raw\n"
            "4. Repr imagen recomendada: gaf\n"
            "5. Tamano GAF recomendado: 128\n"
            "6. Ventana recomendada: 2.0 s | overlap: 0.5\n"
            "7. Para la Rama B conviene dejar activo el balanceo de splits\n"
            "8. Si usas sesiones locales con mezcla sano/parkinson, selecciona etiquetas_locales.csv\n\n"
            "Notas:\n"
            "- El pipeline actual entrena desde raw EEG.\n"
            "- Las bandas del TGAM ya no son requisito para preprocesar local si existe el raw.\n"
            "- STFT queda solo como baseline legacy/opcional."
        )
        guia_layout.addWidget(self.txt_guia)

        # Grupo: rutas
        grp_rutas = QGroupBox("Rutas")
        rutas_layout = QGridLayout(grp_rutas)

        self.txt_sesiones = QLineEdit()
        self.btn_sesiones = QPushButton("Examinar")
        self.btn_sesiones.clicked.connect(lambda: self._seleccionar_carpeta(self.txt_sesiones))

        self.txt_dataset = QLineEdit()
        self.btn_dataset = QPushButton("Examinar")
        self.btn_dataset.clicked.connect(lambda: self._seleccionar_carpeta(self.txt_dataset))

        self.txt_etiquetas_csv = QLineEdit()
        self.btn_etiquetas_csv = QPushButton("Examinar")
        self.btn_etiquetas_csv.clicked.connect(self._seleccionar_csv_etiquetas)

        self.txt_salida = QLineEdit()
        self.btn_salida = QPushButton("Examinar")
        self.btn_salida.clicked.connect(lambda: self._seleccionar_carpeta(self.txt_salida, crear=True))

        self.txt_sesiones.textChanged.connect(self._actualizar_comando)
        self.txt_dataset.textChanged.connect(self._actualizar_comando)
        self.txt_etiquetas_csv.textChanged.connect(self._actualizar_comando)
        self.txt_etiquetas_csv.textChanged.connect(self._ajustar_etiqueta_por_csv)
        self.txt_salida.textChanged.connect(self._actualizar_comando)

        rutas_layout.addWidget(QLabel("Sesiones locales"), 0, 0)
        rutas_layout.addWidget(self.txt_sesiones, 0, 1)
        rutas_layout.addWidget(self.btn_sesiones, 0, 2)

        rutas_layout.addWidget(QLabel("Dataset publico procesado"), 1, 0)
        rutas_layout.addWidget(self.txt_dataset, 1, 1)
        rutas_layout.addWidget(self.btn_dataset, 1, 2)

        rutas_layout.addWidget(QLabel("CSV etiquetas locales (opcional)"), 2, 0)
        rutas_layout.addWidget(self.txt_etiquetas_csv, 2, 1)
        rutas_layout.addWidget(self.btn_etiquetas_csv, 2, 2)

        rutas_layout.addWidget(QLabel("Salida procesada"), 3, 0)
        rutas_layout.addWidget(self.txt_salida, 3, 1)
        rutas_layout.addWidget(self.btn_salida, 3, 2)

        # Grupo: presets
        grp_presets = QGroupBox("Presets rapidos")
        presets_layout = QHBoxLayout(grp_presets)

        self.btn_preset_base = QPushButton("Base recomendada")
        self.btn_preset_base.clicked.connect(self._preset_base)

        self.btn_preset_exploracion = QPushButton("Exploracion")
        self.btn_preset_exploracion.clicked.connect(self._preset_exploracion)

        self.btn_preset_largo = QPushButton("Ventana larga")
        self.btn_preset_largo.clicked.connect(self._preset_largo)

        self.btn_preset_oficial = QPushButton("Reprocesado oficial v1")
        self.btn_preset_oficial.clicked.connect(self._preset_oficial_v1)

        presets_layout.addWidget(self.btn_preset_base)
        presets_layout.addWidget(self.btn_preset_exploracion)
        presets_layout.addWidget(self.btn_preset_largo)
        presets_layout.addWidget(self.btn_preset_oficial)

        # Grupo: comando generado
        grp_cmd = QGroupBox("Comando generado")
        cmd_layout = QHBoxLayout(grp_cmd)

        self.txt_comando = QLineEdit()
        self.txt_comando.setReadOnly(True)
        self.btn_copiar = QPushButton("Copiar comando")
        self.btn_copiar.clicked.connect(self._copiar_comando)

        cmd_layout.addWidget(self.txt_comando)
        cmd_layout.addWidget(self.btn_copiar)

        # Grupo: acciones
        grp_acciones = QGroupBox("Acciones")
        acciones_layout = QHBoxLayout(grp_acciones)

        self.btn_validar = QPushButton("Validar configuracion")
        self.btn_validar.clicked.connect(self._validar_y_mostrar)

        self.btn_ejecutar = QPushButton("Ejecutar pipeline")
        self.btn_ejecutar.clicked.connect(self._ejecutar)

        self.btn_detener = QPushButton("Detener")
        self.btn_detener.setEnabled(False)
        self.btn_detener.clicked.connect(self._detener)

        acciones_layout.addWidget(self.btn_validar)
        acciones_layout.addWidget(self.btn_ejecutar)
        acciones_layout.addWidget(self.btn_detener)

        # Grupo: logs
        grp_logs = QGroupBox("Logs")
        logs_layout = QVBoxLayout(grp_logs)

        self.txt_logs = QTextEdit()
        self.txt_logs.setReadOnly(True)

        barra_logs = QHBoxLayout()
        self.btn_limpiar_logs = QPushButton("Limpiar logs")
        self.btn_limpiar_logs.clicked.connect(self.txt_logs.clear)
        self.btn_ver_historial = QPushButton("Ver historial")
        self.btn_ver_historial.clicked.connect(self._abrir_historial)
        barra_logs.addWidget(self.btn_limpiar_logs)
        barra_logs.addWidget(self.btn_ver_historial)
        barra_logs.addStretch()
        logs_layout.addLayout(barra_logs)
        logs_layout.addWidget(self.txt_logs)

        layout_principal.addWidget(grp_conf)
        layout_principal.addWidget(grp_guia)
        layout_principal.addWidget(grp_rutas)
        layout_principal.addWidget(grp_presets)
        layout_principal.addWidget(grp_cmd)
        layout_principal.addWidget(grp_acciones)
        layout_principal.addWidget(grp_logs, stretch=1)

    def _cargar_defaults(self):
        self.cmb_fuente.setCurrentText("publico")
        self.cmb_etiqueta_local.setCurrentText("sano")
        self.spn_ventana.setValue(2.0)
        self.cmb_overlap.setCurrentText("0.5")
        self.cmb_rama.setCurrentText("ambas")
        self.cmb_norm.setCurrentText("ambos")
        self.cmb_repr_temporal.setCurrentText("raw")
        self.cmb_repr_imagen.setCurrentText("gaf")
        self.cmb_gaf_size.setCurrentText("128")

        self.cmb_fuente.setToolTip("publico: OpenNeuro | local: sesiones TGAM | ambos: mezcla")
        self.cmb_repr_temporal.setToolTip("raw recomendado; features solo como baseline legacy")
        self.cmb_repr_imagen.setToolTip("gaf recomendado; stft solo como baseline legacy")
        self.cmb_gaf_size.setToolTip("128 recomendado para entrenar localmente sin gastar demasiada memoria")
        self.chk_balancear_splits.setToolTip(
            "Rebalancea train/val/test por sujeto en la Rama B para evitar splits sesgados."
        )
        self.txt_etiquetas_csv.setToolTip(
            "CSV opcional con columnas sello,etiqueta. Usa el archivo etiquetas_locales.csv generado por el grabador."
        )

        self.txt_sesiones.setText(str(self.proyecto_dir / "datos" / "sesiones"))
        self.txt_dataset.setText(str(self.proyecto_dir / "datos" / "openneuro_procesado"))
        self.txt_etiquetas_csv.setText("")
        self.txt_salida.setText(str(self.proyecto_dir / "datos" / "procesados"))

        self._ajustar_estado_por_fuente()
        self._ajustar_estado_repr_imagen()

    # -------------------------- Presets --------------------------

    def _preset_base(self):
        self.cmb_fuente.setCurrentText("publico")
        self.cmb_rama.setCurrentText("ambas")
        self.cmb_norm.setCurrentText("ambos")
        self.spn_ventana.setValue(2.0)
        self.cmb_overlap.setCurrentText("0.5")
        self.cmb_repr_temporal.setCurrentText("raw")
        self.cmb_repr_imagen.setCurrentText("gaf")
        self.cmb_gaf_size.setCurrentText("128")
        self.chk_balancear_splits.setChecked(True)

    def _preset_exploracion(self):
        self.cmb_fuente.setCurrentText("ambos")
        self.cmb_rama.setCurrentText("ambas")
        self.cmb_norm.setCurrentText("ambos")
        self.spn_ventana.setValue(2.0)
        self.cmb_overlap.setCurrentText("0.5")
        self.cmb_repr_temporal.setCurrentText("raw")
        self.cmb_repr_imagen.setCurrentText("gaf")
        self.cmb_gaf_size.setCurrentText("128")
        self.chk_balancear_splits.setChecked(True)

    def _preset_largo(self):
        self.cmb_fuente.setCurrentText("publico")
        self.cmb_rama.setCurrentText("a")
        self.cmb_norm.setCurrentText("zscore")
        self.spn_ventana.setValue(5.0)
        self.cmb_overlap.setCurrentText("0.75")
        self.cmb_repr_temporal.setCurrentText("raw")

    def _preset_oficial_v1(self):
        self.cmb_fuente.setCurrentText("publico")
        self.cmb_rama.setCurrentText("ambas")
        self.cmb_norm.setCurrentText("ambos")
        self.spn_ventana.setValue(2.0)
        self.cmb_overlap.setCurrentText("0.5")
        self.cmb_repr_temporal.setCurrentText("raw")
        self.cmb_repr_imagen.setCurrentText("gaf")
        self.cmb_gaf_size.setCurrentText("128")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        salida = self.proyecto_dir / "datos" / "procesados_oficiales" / f"preprocesado_oficial_v1_{ts}"
        salida.mkdir(parents=True, exist_ok=True)
        self.txt_salida.setText(str(salida))
        self._log(f"[OK] Preset oficial v1 aplicado. Salida: {salida}")

    # -------------------------- Helpers --------------------------

    def _seleccionar_carpeta(self, control: QLineEdit, crear: bool = False):
        inicial = control.text().strip() or str(self.proyecto_dir)
        carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta", inicial)
        if carpeta:
            if crear:
                Path(carpeta).mkdir(parents=True, exist_ok=True)
            control.setText(carpeta)

    def _seleccionar_csv_etiquetas(self):
        inicial = self.txt_etiquetas_csv.text().strip() or str(self.proyecto_dir)
        archivo, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar CSV de etiquetas",
            inicial,
            "CSV (*.csv);;Todos los archivos (*.*)",
        )
        if archivo:
            self.txt_etiquetas_csv.setText(archivo)

    def _ajustar_estado_por_fuente(self):
        fuente = self.cmb_fuente.currentText()
        usa_local = fuente in ("local", "ambos")
        usa_publico = fuente in ("publico", "ambos")

        self.cmb_etiqueta_local.setEnabled(usa_local)
        self.txt_sesiones.setEnabled(usa_local)
        self.btn_sesiones.setEnabled(usa_local)
        self.txt_etiquetas_csv.setEnabled(usa_local)
        self.btn_etiquetas_csv.setEnabled(usa_local)

        self.txt_dataset.setEnabled(usa_publico)
        self.btn_dataset.setEnabled(usa_publico)

        # Re-evaluar si el combo etiqueta debe estar habilitado
        self._ajustar_etiqueta_por_csv()

    def _ajustar_etiqueta_por_csv(self):
        """Deshabilita el combo de etiqueta global cuando hay un CSV de etiquetas cargado.

        Con CSV activo, la etiqueta por sesion viene del archivo y el combo global
        no tiene efecto — deshabilitarlo evita confusion sobre cual tiene prioridad.
        """
        fuente = self.cmb_fuente.currentText()
        usa_local = fuente in ("local", "ambos")
        hay_csv = bool(self.txt_etiquetas_csv.text().strip())
        self.cmb_etiqueta_local.setEnabled(usa_local and not hay_csv)
        if hay_csv:
            self.cmb_etiqueta_local.setToolTip(
                "Deshabilitado: las etiquetas se leen del CSV seleccionado."
            )
        else:
            self.cmb_etiqueta_local.setToolTip("")

    def _ajustar_estado_repr_imagen(self):
        usa_gaf = self.cmb_repr_imagen.currentText() == "gaf"
        self.cmb_gaf_size.setEnabled(usa_gaf)
        if usa_gaf:
            self.cmb_gaf_size.setToolTip("")
        else:
            self.cmb_gaf_size.setToolTip("Solo aplica cuando la representacion de imagen es GAF.")

    def _abrir_historial(self):
        """Abre el explorador de archivos con el historial seleccionado."""
        if not self.historial_path.exists():
            QMessageBox.information(self, "Historial", "Aun no hay historial de ejecuciones.")
            return
        subprocess.Popen(["explorer", "/select,", str(self.historial_path)])

    def _build_args(self) -> list[str]:
        args = [
            "-m", "src.preprocesamiento.pipeline",
            "--fuente", self.cmb_fuente.currentText(),
            "--ventana", f"{self.spn_ventana.value():.1f}",
            "--overlap", self.cmb_overlap.currentText(),
            "--rama", self.cmb_rama.currentText(),
            "--norm", self.cmb_norm.currentText(),
            "--repr-temporal", self.cmb_repr_temporal.currentText(),
            "--repr-imagen", self.cmb_repr_imagen.currentText(),
            "--gaf-size", self.cmb_gaf_size.currentText(),
            "--balancear-splits" if self.chk_balancear_splits.isChecked() else "--no-balancear-splits",
            "--sesiones", self.txt_sesiones.text().strip(),
            "--dataset", self.txt_dataset.text().strip(),
            "--salida", self.txt_salida.text().strip(),
        ]

        if self.cmb_fuente.currentText() in ("local", "ambos"):
            args.extend(["--etiqueta-local", self.cmb_etiqueta_local.currentText()])
            ruta_csv = self.txt_etiquetas_csv.text().strip()
            if ruta_csv:
                args.extend(["--etiquetas-local-csv", ruta_csv])

        return args

    def _comando_string(self) -> str:
        partes = [self.python_exe] + self._build_args()
        return " ".join(shlex.quote(p) for p in partes)

    def _actualizar_comando(self):
        self.txt_comando.setText(self._comando_string())

    def _copiar_comando(self):
        QApplication.clipboard().setText(self.txt_comando.text())
        self._log("[OK] Comando copiado al portapapeles.")

    def _validar(self) -> list[str]:
        errores = []
        fuente = self.cmb_fuente.currentText()

        if fuente in ("local", "ambos"):
            ruta_ses = Path(self.txt_sesiones.text().strip())
            if not ruta_ses.is_dir():
                errores.append(f"No existe carpeta de sesiones locales: {ruta_ses}")

            ruta_csv = self.txt_etiquetas_csv.text().strip()
            if ruta_csv:
                ruta_csv_path = Path(ruta_csv)
                if not ruta_csv_path.is_file():
                    errores.append(f"No existe CSV de etiquetas locales: {ruta_csv}")
                else:
                    errores.append(f"No existe CSV de etiquetas locales: {ruta_csv}")

        if fuente in ("publico", "ambos"):
            ruta_pub = Path(self.txt_dataset.text().strip())
            if not ruta_pub.is_dir():
                errores.append(f"No existe carpeta de dataset publico: {ruta_pub}")
            else:
                indice = ruta_pub / "indice.json"
                if not indice.exists():
                    errores.append(
                        f"Falta indice.json en dataset publico ({indice}). "
                        "Primero hay que procesar OpenNeuro."
                    )

        ruta_out = Path(self.txt_salida.text().strip())
        if not ruta_out:
            errores.append("La ruta de salida no puede estar vacia.")

        return errores

    def _validar_y_mostrar(self):
        errores = self._validar()
        if errores:
            QMessageBox.warning(self, "Validacion", "\n".join(errores))
            return
        QMessageBox.information(self, "Validacion", "Configuracion valida.")

    # -------------------------- Ejecucion --------------------------

    def _ejecutar(self):
        if self.proceso.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Pipeline", "Ya hay una ejecucion en curso.")
            return

        errores = self._validar()
        if errores:
            QMessageBox.warning(self, "Validacion", "\n".join(errores))
            return

        Path(self.txt_salida.text().strip()).mkdir(parents=True, exist_ok=True)

        args = self._build_args()
        self._comando_ultimo = self._comando_string()
        self._inicio_ejecucion = datetime.now()
        self.txt_logs.clear()
        self._log("=" * 72)
        self._log("Iniciando pipeline...")
        self._log(self._comando_ultimo)
        self._log("=" * 72)

        self._set_running(True)
        self.proceso.setWorkingDirectory(str(self.proyecto_dir))
        self.proceso.start(self.python_exe, args)

    def _detener(self):
        if self.proceso.state() == QProcess.ProcessState.Running:
            self._log("[WARN] Deteniendo proceso por solicitud del usuario...")
            self.proceso.kill()

    def _set_running(self, running: bool):
        self.btn_ejecutar.setEnabled(not running)
        self.btn_detener.setEnabled(running)
        self.btn_validar.setEnabled(not running)

        # Evitar cambios de configuracion durante ejecucion
        for w in [
            self.cmb_fuente,
            self.cmb_etiqueta_local,
            self.spn_ventana,
            self.cmb_overlap,
            self.cmb_rama,
            self.cmb_norm,
            self.cmb_repr_temporal,
            self.cmb_repr_imagen,
            self.cmb_gaf_size,
            self.chk_balancear_splits,
            self.txt_sesiones,
            self.btn_sesiones,
            self.txt_dataset,
            self.btn_dataset,
            self.txt_etiquetas_csv,
            self.btn_etiquetas_csv,
            self.txt_salida,
            self.btn_salida,
            self.btn_preset_base,
            self.btn_preset_exploracion,
            self.btn_preset_largo,
            self.btn_preset_oficial,
            self.btn_copiar,
            self.btn_ver_historial,
        ]:
            w.setEnabled(not running)
        # Limpiar logs siempre disponible excepto durante ejecucion para no perder output
        self.btn_limpiar_logs.setEnabled(not running)

    def _on_stdout(self):
        data = self.proceso.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if data:
            self._log(data.rstrip("\n"))

    def _on_stderr(self):
        data = self.proceso.readAllStandardError().data().decode("utf-8", errors="replace")
        if data:
            self._log("[stderr] " + data.rstrip("\n"))

    def _on_finished(self, codigo: int, _status):
        self._set_running(False)
        if codigo == 0:
            self._log("\n[OK] Pipeline finalizado correctamente.")
        else:
            self._log(f"\n[ERROR] Pipeline termino con codigo {codigo}.")
        self._guardar_historial(codigo)

    def _log(self, texto: str):
        self.txt_logs.append(texto)
        self.txt_logs.verticalScrollBar().setValue(self.txt_logs.verticalScrollBar().maximum())

    def _guardar_historial(self, codigo: int):
        inicio = self._inicio_ejecucion or datetime.now()
        fin = datetime.now()
        registro = {
            "inicio": inicio.isoformat(timespec="seconds"),
            "fin": fin.isoformat(timespec="seconds"),
            "duracion_seg": round((fin - inicio).total_seconds(), 2),
            "exit_code": int(codigo),
            "estado": "ok" if codigo == 0 else "error",
            "comando": self._comando_ultimo,
            "config": {
                "fuente": self.cmb_fuente.currentText(),
                "etiqueta_local": self.cmb_etiqueta_local.currentText(),
                "ventana": float(self.spn_ventana.value()),
                "overlap": float(self.cmb_overlap.currentText()),
                "rama": self.cmb_rama.currentText(),
                "norm": self.cmb_norm.currentText(),
                "repr_temporal": self.cmb_repr_temporal.currentText(),
                "repr_imagen": self.cmb_repr_imagen.currentText(),
                "gaf_size": int(self.cmb_gaf_size.currentText()),
                "balancear_splits": bool(self.chk_balancear_splits.isChecked()),
                "sesiones": self.txt_sesiones.text().strip(),
                "dataset": self.txt_dataset.text().strip(),
                "etiquetas_csv": self.txt_etiquetas_csv.text().strip(),
                "salida": self.txt_salida.text().strip(),
            },
        }

        self.historial_path.parent.mkdir(parents=True, exist_ok=True)
        historial = []
        if self.historial_path.exists():
            try:
                with open(self.historial_path, "r", encoding="utf-8") as f:
                    historial = json.load(f)
                if not isinstance(historial, list):
                    historial = []
            except Exception:
                historial = []

        historial.append(registro)
        with open(self.historial_path, "w", encoding="utf-8") as f:
            json.dump(historial, f, indent=2, ensure_ascii=False)

        self._log(f"[OK] Historial guardado en: {self.historial_path}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    ventana = VentanaPreprocesamiento()
    ventana.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
