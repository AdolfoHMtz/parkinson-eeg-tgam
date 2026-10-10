# Plan de Implementacion - Deteccion Temprana de Parkinson con EEG
## Servicio Social - NeuroSky TGAM + Redes Neuronales

---

## Objetivo General

Analizar modelos de redes neuronales artificiales a partir de datos de electroencefalogramas
(EEG) de pacientes sanos y con enfermedad de Parkinson, para entrenar modelos variando
parametros, normalizando entradas y definiendo ventanas de tiempo. Verificar exactitudes
de prediccion y sentar las bases para una aplicacion de deteccion temprana en tiempo real
para apoyo a neurologos.

---

## Actualizacion Estrategica

Tras comparar sesiones locales TGAM contra OpenNeuro procesado a un solo canal frontal
(`Fp1`, remuestreado a 512 Hz), se decidio formalizar el siguiente rumbo:

- La **senal base comun** para integracion y entrenamiento sera el **raw EEG monocanal**
- Las **bandas del TGAM** se conservan para inspeccion, trazabilidad y visualizacion,
  pero dejan de ser la entrada principal del entrenamiento cruzado con dataset publico
- La **Rama A** prioriza **series temporales raw normalizadas**; los features estadisticos
  quedan como baseline legacy/opcional
- La **Rama B** migra de **STFT/espectrograma** a **GAF (Gramian Angular Field)** como
  representacion principal para CNN 2D; STFT queda como ruta legacy/opcional

Justificacion:
- `raw` existe tanto en TGAM como en OpenNeuro-Fp1
- GAF nace directamente de la serie temporal, por lo que mantiene coherencia entre
  la rama secuencial y la rama de imagen
- Con 1 electrodo, GAF ofrece una comparacion mas limpia del patron temporal que
  un espectrograma pensado originalmente para analisis tiempo-frecuencia mas rico

La evidencia automatizada de esta decision se genera con:

```bash
python herramientas/comparar_tgam_vs_openneuro.py
```

Salida:
- `herramientas/compatibilidad_tgam_vs_openneuro/resumen_compatibilidad.json`
- `herramientas/compatibilidad_tgam_vs_openneuro/REPORTE_COMPATIBILIDAD.md`

---

## Dispositivo

- **Modulo:** NeuroSky TGAM (ThinkGear ASIC Module)
- **Conexion:** Adaptador BT-USB -> Puerto COM (autodetectable, actualmente COM5)
- **Baud rate:** 57600, 8N1
- **Protocolo:** Binario, paquetes [0xAA][0xAA][LENGTH][PAYLOAD][CHECKSUM]

## Datos que captura el dispositivo

| Dato | Tipo | Descripcion |
|------|------|-------------|
| Raw EEG | signed 16-bit (512 Hz) | Onda cerebral cruda |
| Delta | float | Banda 0.5-2.75 Hz (sueno profundo) |
| Theta | float | Banda 3.5-6.75 Hz (meditacion, somnolencia) |
| Low Alpha | float | Banda 7.5-9.25 Hz (relajacion) |
| High Alpha | float | Banda 10-12 Hz (relajacion ligera) |
| Low Beta | float | Banda 12.75-20 Hz (concentracion) |
| High Beta | float | Banda 20.25-28 Hz (actividad mental intensa) |
| Low Gamma | float | Banda 30-40 Hz (rendimiento mental) |
| High Gamma | float | Banda 40-50 Hz (rendimiento mental pico) |
| Attention | int 0-100 | Nivel de atencion |
| Meditation | int 0-100 | Nivel de meditacion |
| Poor Signal | int 0-200 | Calidad de senal (0 = buena, 200 = sin contacto) |
| Blink Strength | int 0-255 | Fuerza de parpadeo |

---

## Checklist de Cierre de Fase

Al finalizar cada fase se debe completar el siguiente checklist antes de pasar a la siguiente:

- [ ] **Pruebas funcionales:** Verificar que todo funciona con el dispositivo fisico
- [ ] **Revision de codigo:** Ejecutar revision de calidad, eficiencia y reuso (/simplify)
- [ ] **Reporte de fase:** Crear documento en `reportes/FASE_X_REPORTE.md` que incluya:
  - Objetivo de la fase y si se cumplio
  - Que se hizo (actividades realizadas)
  - Que se observo (hallazgos, problemas del codigo original, comportamiento del dispositivo)
  - Decisiones tecnicas tomadas y su justificacion
  - Diferencias respecto al codigo de referencia original y por que se hicieron
  - Problemas encontrados y como se resolvieron
  - Archivos creados o modificados
  - Dependencias instaladas
- [ ] **Actualizacion del plan:** Marcar tareas completadas y actualizar estado de la fase
- [ ] **Manual tecnico:** Agregar seccion correspondiente en `reportes/MANUAL_TECNICO.md`
  - Como funciona lo implementado en esta fase
  - Como ejecutarlo
  - Configuracion necesaria
  - Troubleshooting comun

Estos reportes sirven para dos entregables finales:
1. **Reporte de actividades del servicio social** - Compilacion de todos los reportes de fase
2. **Manual tecnico** - Documento acumulativo para futuros colaboradores del proyecto

---

## Fases del Proyecto

### FASE 1 - Lectura del dispositivo en Python
**Estado:** COMPLETADA
**Objetivo:** Conectar al TGAM y leer datos identicos a los de Realterm/MindView

**Tareas:**
- [x] Instalar dependencias: `pyserial`
- [x] Implementar autodeteccion de puerto COM (no hardcodear COM5)
- [x] Portar parser binario de `03_Codigo_Fuente_Referencia/ReadParseTGAM_C#/Parser.cs`
- [x] Decodificar todos los tipos de paquete (raw, bandas, eSense, signal)
- [x] Validar checksum de paquetes
- [x] Prueba con dispositivo fisico: comparar datos vs Realterm
- [x] Manejo de errores: desconexion, senal pobre, paquetes corruptos
- [x] Revision de codigo (calidad, eficiencia, reuso)
- [x] Reporte de fase: `reportes/FASE_1_REPORTE.md`
- [x] Manual tecnico: `reportes/MANUAL_TECNICO.md`

**Archivos de referencia:**
- `03_Codigo_Fuente_Referencia/ReadParseTGAM_C#/ReadParseTGAM/ReadParseTGAM/Parser.cs`
- `03_Codigo_Fuente_Referencia/ReadParseTGAM_C#/ReadParseTGAM/ReadParseTGAM/COMPortManager.cs`
- `01_Documentacion/TGAM Communications/ThinkGear/ThinkGearStreamParser.c`
- `03_Codigo_Fuente_Referencia/thinkgear_testapp_C++Sample/thinkgear_testapp/thinkgear.h`

**Librerias:** `pyserial`, `serial.tools.list_ports`

---

### FASE 2 - Visualizacion en tiempo real
**Estado:** COMPLETADA
**Objetivo:** Replicar las graficas de MindView en Python

**Graficas implementadas:**
- **Panel superior:** Raw Wave (eje derecho, semi-transparente) + 8 bandas de frecuencia apiladas (eje izquierdo)
- **Panel inferior:** Noise, Attention, Meditation como areas coloreadas (escala 0-100)
- **Eje X:** Hora real (HH:MM:SS) como MindView, no segundos relativos
- **Escalado:** Percentil 95 para evitar que picos extremos de delta aplaste las demas bandas

**Tareas:**
- [x] Evaluar matplotlib vs PyQtGraph para graficas en tiempo real
- [x] Implementar graficas con PyQtGraph + PyQt6 (descartando matplotlib por rendimiento)
- [x] Buffer circular thread-safe con NumPy para datos en tiempo real
- [x] Doble eje Y en panel superior (bandas izquierda, raw derecho)
- [x] 8 bandas apiladas como areas coloreadas (stacked area chart)
- [x] Raw wave semi-transparente (alpha=120) superpuesta sin tapar bandas
- [x] Leyenda de colores por banda de frecuencia en ambos paneles
- [x] Selector de ventana temporal (10, 30, 60, 120 segundos)
- [x] Boton de pausa/reanudar
- [x] Indicador de calidad de senal con colores (excelente/buena/regular/sin contacto)
- [x] Eje X con hora real (HH:MM:SS) mediante AxisItem personalizado
- [x] Escalado Y con percentil 95 para manejar picos extremos de delta
- [x] Diezmado de raw wave (>5000 puntos) para mantener fluidez
- [x] Modo demo con datos sinteticos realistas para pruebas sin hardware
- [x] Monitor de flujo (Raw/s, Band/s, Pkt/s, FPS) en barra de estadisticas
- [x] Tema oscuro completo (fondo #1E1E1E, paleta Fusion)
- [x] Prueba simultanea con MindView usando VSPE (splitter de puertos COM virtuales)

**Librerias:** `PyQt6`, `pyqtgraph`, `numpy`

**Nota sobre cambio de libreria:**
Se planeo inicialmente usar `matplotlib` pero se descarto porque `matplotlib.animation`
no podia mantener 30 FPS fluidos con 512 Hz de raw EEG + 8 bandas apiladas actualizandose
en tiempo real. PyQtGraph renderiza directamente en GPU via Qt, logrando >30 FPS estables.

**Nota sobre prueba simultanea:**
Para validar que la aplicacion Python muestra los mismos datos que MindView, se uso
VSPE (Virtual Serial Port Emulator) como splitter: COM5 (TGAM real) -> COM8 + COM9.
MindView leia de COM8 y Python de COM9 simultaneamente. Se intentó primero con com0com
+ hub4com pero el driver no se activa correctamente en Windows 11 con Secure Boot.
Las capturas comparativas confirmaron que ambas aplicaciones muestran los mismos datos.

**Cierre de fase:**
- [x] Pruebas funcionales con dispositivo (simultanea con MindView via VSPE)
- [x] Reporte: `reportes/FASE_2_REPORTE.md`
- [x] Manual tecnico actualizado

---

### FASE 3 - Exportacion a CSV orientada a ML
**Estado:** COMPLETADA
**Objetivo:** Guardar sesiones de captura en formato CSV estructurado y listo para
alimentar directamente ambas ramas del pipeline de aprendizaje automatico (series
temporales y generacion de espectrogramas).

**Contexto de la decision (ver seccion "Estrategia de datos para ML" mas abajo):**
El CSV es la unica fuente de verdad. Las imagenes para CNN se generan on-the-fly
en la fase de preprocesamiento a partir de este CSV, no se guardan aqui.

**Dos archivos por sesion:**

1. `sesion_YYYY-MM-DD_HH-MM-SS_raw.csv` — raw EEG a 512 Hz
   ```
   timestamp_epoch, timestamp_hora, raw_eeg, poor_signal
   ```

2. `sesion_YYYY-MM-DD_HH-MM-SS_bandas.csv` — bandas + eSense a ~1 Hz
   ```
   timestamp_epoch, timestamp_hora, delta, theta, low_alpha, high_alpha,
   low_beta, high_beta, low_gamma, high_gamma, attention, meditation, poor_signal
   ```

**Tareas:**
- [x] Implementar grabador de sesion (`src/exportacion/grabador_csv.py`)
- [x] Escritura incremental (no acumular en RAM, flush por bloque)
- [x] Metadatos en header: fecha, duracion, puerto COM, fs_raw, version software
- [x] Duracion configurable: 30s, 2min, 5min, 10min, personalizado, sin limite
- [x] Boton "Grabar" integrado en la ventana de visualizacion (PyQtGraph)
- [x] Indicador visual de grabacion activa (barra de progreso + timer REC MM:SS)
- [x] Nombre automatico: `sesion_YYYY-MM-DD_HH-MM-SS_{tipo}.csv`
- [x] Carpeta destino configurable (default: `datos/sesiones/`)
- [x] Solo guardar muestras con `poor_signal < 50` (senal valida)

**Librerias:** `csv` (stdlib)

**Cierre de fase:**
- [x] Pruebas funcionales con dispositivo (modo demo confirmado; pendiente validar con TGAM fisico)
- [x] Revision de codigo (/simplify)
- [x] Reporte: `reportes/FASE_3_REPORTE.md`
- [x] Manual tecnico actualizado

---

### FASE 4 - Preprocesamiento de datos
**Estado:** COMPLETADA
**Objetivo:** Preparar representaciones del raw EEG para entrenamiento local:
series temporales normalizadas y una imagen derivada de la serie temporal.

**Rama A — Series temporales (raw / baseline features):**
- [x] Normalizacion MinMax y Z-score (comparar cual generaliza mejor)
- [x] Ventanas deslizantes: tamanios 1s, 2s, 5s, 10s con overlap 50% y 75%
- [x] Serie temporal raw por ventana como representacion principal
- [x] Extraccion de features estadisticos por ventana como baseline legacy
- [x] Etiquetado: sano=0, Parkinson=1
- [x] Etiqueta local global por CLI (`--etiqueta-local`)
- [x] Etiquetado por sesion local via CSV (`--etiquetas-local-csv`)

**Rama B — GAF para CNN:**
- [x] Generar GAF (Gramian Angular Field) del raw EEG por ventana
- [x] Guardar como arrays NumPy (NO como imagenes PNG — evita perdida de precision)
- [x] Mantener STFT como baseline legacy/opcional para comparacion interna
- [x] Mismo etiquetado: sano=0, Parkinson=1

**Datos:**
- [x] Buscar e integrar datasets publicos EEG-Parkinson (PhysioNet, OpenNeuro)
- [x] Adaptar formato publico al mismo esquema que los CSVs propios
- [x] Unificar y mezclar: datos propios + datos publicos
- [x] Procesamiento OpenNeuro configurable a solo canal Fp1 (`solo_fp1=True`)

**Usabilidad y operacion:**
- [x] GUI para preprocesamiento (`preprocesar_datos_gui.py`)
- [x] Presets de ejecucion (incluye `Reprocesado oficial v1`)
- [x] Historial JSON de corridas (`resultados/historial_preprocesamiento.json`)
- [x] Registro automatico de etiquetas locales en CSV central por carpeta

**Dataset publico seleccionado:** OpenNeuro ds002778 (UC San Diego)
  - 15 pacientes PD + 16 controles sanos, 32 canales BioSemi, formato BIDS
  - Se extrae canal Fp1 (posicion equivalente al TGAM) y se remuestrea a 512 Hz
  - Todos los canales disponibles para CNN multicanal

**Librerias:** `numpy`, `scipy`, `pandas`, `scikit-learn`, `mne`, `mne-bids`, `openneuro-py`

**Cierre de fase:**
- [x] Pruebas funcionales con datos reales (pipeline ejecutado con 5 sesiones locales)
- [ ] Revision de codigo (/simplify)
- [x] Reporte: `reportes/FASE_4_REPORTE.md`
- [x] Manual tecnico actualizado

---

### FASE 5 - Analisis de modelos de redes neuronales
**Estado:** COMPLETADA para Rama B (CNN 2D sobre GAF). Rama A queda como trabajo futuro.
**Objetivo:** Entrenar y comparar arquitecturas sobre una senal comun compatible entre
TGAM y OpenNeuro: raw EEG monocanal por ventana, tanto en forma secuencial como en
forma de imagen GAF.

**Rama A — Modelos de series temporales:**
| Modelo | Entrada | Fortaleza |
|--------|---------|-----------|
| 1D CNN | Ventanas raw normalizadas | Baseline fuerte, eficiente y local |
| LSTM | Ventanas raw normalizadas | Dependencias temporales largas |
| GRU | Ventanas raw normalizadas | Alternativa eficiente a LSTM |
| Transformer | Ventanas raw normalizadas | Atencion global, estado del arte |

**Rama B — Modelos sobre imagen temporal:**
| Modelo | Entrada | Fortaleza |
|--------|---------|-----------|
| CNN 2D | GAF por ventana | Patrones globales de la serie temporal |
| CNN + LSTM hibrida | GAF -> features -> LSTM | Combina representacion espacial y secuencial |
| CNN 2D (legacy) | Espectrograma STFT por ventana | Baseline historico opcional |

**Tareas:**
- [x] Implementar CNN 2D sobre GAF (Rama B) — `src/modelos/entrenar_cnn_rama_b.py`
- [x] Entrenar con OpenNeuro ds002778 (canal Fp1, 512 Hz)
- [x] Variar hiperparametros (learning rate, batch size, dropout, capas densas, funcion de perdida) — versiones v1 a v5
- [x] Metricas: accuracy, balanced accuracy, precision, recall, F1, ROC AUC, PR AUC
- [x] Matriz de confusion y curvas de aprendizaje por corrida
- [x] Umbral de decision calibrado en validacion
- [x] Corregir division desbalanceada del test (v5)
- [x] Corregir fuga de datos por paciente: division agrupada por persona (v6)
- [x] Seleccionar modelo de referencia con justificacion (v6, ver `reportes/FASE_5_REPORTE.md`)
- [ ] Rama A: 1D CNN / LSTM / GRU / Transformer sobre ventanas raw (trabajo futuro)
- [ ] CNN + LSTM hibrida (trabajo futuro)
- [ ] Variar tamano de ventana (trabajo futuro)
- [ ] Tabla comparativa Rama A vs Rama B (cuando exista Rama A)
- [ ] Repetir con distintas particiones de personas para medir la variabilidad

**Librerias:** `tensorflow`/`keras`, `scikit-learn`

**Cierre de fase:**
- [x] Pruebas con datos de validacion y test (personas no vistas)
- [x] Reporte: `reportes/FASE_5_REPORTE.md` y `reportes/FASE_5_ENTRENAMIENTO_EXPLICADO.md`
- [x] Manual tecnico actualizado

---

### FASE 6 - Aplicacion de diagnostico en tiempo real (futuro)
**Estado:** Futuro
**Objetivo:** Integrar el mejor modelo con la lectura del dispositivo para diagnostico preventivo

**Tareas:**
- [ ] Reescribir la inferencia por sesion (`evaluar_paciente.py`, se perdio) sobre el modelo Keras
- [ ] Cargar modelo entrenado
- [ ] Pipeline en tiempo real: lectura -> preprocesamiento -> prediccion
- [ ] Interfaz grafica para neurologos
- [ ] Indicadores: probabilidad Parkinson, confianza del modelo, senal de alerta
- [ ] Historial de sesiones por paciente
- [ ] Exportar reportes

**Cierre de fase:**
- [ ] Pruebas de integracion completas
- [ ] Revision de codigo (/simplify)
- [ ] Reporte: `reportes/FASE_6_REPORTE.md`
- [ ] Manual tecnico actualizado
- [ ] Reporte final compilado de todas las fases

---

---

## Estrategia de Datos para ML: Raw Monocanal + GAF

### Decision vigente

La estrategia vigente del proyecto usa una **senal base comun** entre TGAM y OpenNeuro:

- **Raw EEG monocanal** como entrada principal
- **Series temporales raw normalizadas** para modelos secuenciales
- **GAF (Gramian Angular Field)** para la rama de imagen con CNN
- **Bandas TGAM** solo como apoyo de trazabilidad e inspeccion
- **STFT/espectrograma** solo como baseline legacy, no como ruta principal

### Resumen del enfoque vigente

- El entrenamiento principal parte del `raw`
- La rama secuencial usa ventanas raw normalizadas
- La rama de imagen usa GAF generado desde esas mismas ventanas
- Las bandas del TGAM ya no se usan como representacion principal para mezclar con OpenNeuro

### Decision historica (obsoleta)

### Decision tomada

Se adopta un **enfoque hibrido** que entrena y compara dos familias de modelos:

1. **Modelos de series temporales** (LSTM, GRU, Transformer) — trabajan directamente
   con los vectores numericos de las 8 bandas y/o el raw EEG normalizado.
2. **Modelos sobre espectrogramas** (CNN 2D, CNN+LSTM hibrida) — trabajan con la
   representacion tiempo-frecuencia del raw EEG convertida a imagen numerica via STFT.

### Por que no solo series temporales

Con 1 solo electrodo (TGAM), los vectores de bandas pre-calculadas tienen poca dimensionalidad.
El espectrograma del raw EEG captura patron tiempo-frecuencia que los vectores de bandas
no representan directamente (ej. fragmentacion ritmica, marcadores de temblor a 4-6 Hz).

### Por que no solo espectrogramas

- Transfer learning con CNN preentrenada (ImageNet) no aplica directamente a espectrogramas
  EEG de 1 canal — requiere entrenamiento desde cero con datos escasos.
- Las series temporales son mas ligeras, mas faciles de interpretar, y varios papers de
  deteccion de Parkinson con EEG consiguen >90% de accuracy con LSTM sin imagenes.
- Con 1 electrodo, el espectrograma no tiene la riqueza espacial de sistemas de 32/64 canales.

### Por que el hibrido

- Comparacion justa: si CNN gana, hay justificacion cientifica para ese approach.
  Si LSTM gana, tambien. El servicio social requiere analisis comparativo, no solo implementar uno.
- Los datasets publicos de Parkinson-EEG (PhysioNet) tienen multiples canales — se pueden
  usar completos para CNN y solo el canal equivalente a Fp1 para LSTM, permitiendo
  evaluar el impacto de la limitacion de 1 electrodo.
- El costo computacional extra es razonable: ambas ramas comparten el mismo CSV de origen.

### Separacion critica: visualizacion vs ML

La interfaz grafica (PyQtGraph, fase 2) y el pipeline de ML son completamente independientes:

```
TGAM -> BufferEEG -> [VentanaEEG]          <- para el medico (fase 2)
                  -> [GrabadorCSV]          <- fuente de verdad (fase 3)
                       |
                       v
              [Preprocesamiento]            <- fase 4
              /                \
   [Series temporales]    [Espectrogramas]
          |                      |
   [LSTM/GRU/Transformer]  [CNN 2D / CNN+LSTM]
          \                      /           <- fase 5
           [Tabla comparativa final]
```

Los espectrogramas para CNN se generan en fase 4 desde el CSV, nunca desde capturas
de pantalla de la interfaz. Esto garantiza precision numerica completa sin degradacion
por renderizado o compresion de imagen.

### Pros y contras por enfoque

| Aspecto | Series temporales | Espectrogramas (CNN) |
|---------|------------------|----------------------|
| Precision de datos | Total (valores float) | Ligera perdida por discretizacion STFT |
| Peso del dataset | Bajo (CSVs numericos) | Mayor (arrays STFT por ventana) |
| Velocidad de entrenamiento | Alta | Moderada |
| Interpretabilidad clinica | Moderada | Alta (imagen visible) |
| Transfer learning | No aplica | Limitado (1 canal vs RGB) |
| Requiere raw EEG | Optativo (bandas solas bastan) | Si (necesita raw para STFT) |
| Papers con buenos resultados | Si (LSTM >90% en varios) | Si (CNN sobre wavelet scalogram) |
| Adecuado para 1 electrodo | Si | Parcialmente |

---

## Estructura del Proyecto

```
07_Proyecto_Python/
|-- src/
|   |-- conexion/          # Autodeteccion COM + comunicacion serial
|   |-- parser/            # Decodificacion protocolo binario TGAM
|   |-- visualizacion/     # Graficas en tiempo real (estilo MindView)
|   |-- exportacion/       # Guardado CSV con metadatos
|   |-- preprocesamiento/  # Normalizacion, ventanas, espectrogramas
|   |-- modelos/           # Redes neuronales (CNN, LSTM, hibrida, etc.)
|-- datos/                 # CSVs capturados + datasets publicos
|-- resultados/            # Metricas, graficas comparativas
|-- reportes/              # Reportes de fase + manual tecnico
|   |-- FASE_X_REPORTE.md  # Un reporte por cada fase completada
|   |-- MANUAL_TECNICO.md  # Manual acumulativo para futuros colaboradores
|-- requirements.txt       # Dependencias del proyecto
|-- PLAN_DE_IMPLEMENTACION.md  # Este archivo
```

## Stack Tecnologico

| Categoria | Tecnologia |
|-----------|-----------|
| Lenguaje | Python 3.13 |
| Comunicacion serial | pyserial |
| Visualizacion | PyQt6 + pyqtgraph (reemplazo de matplotlib por rendimiento) |
| Datos | numpy, pandas |
| Procesamiento senal | scipy |
| Machine Learning | scikit-learn |
| Deep Learning | tensorflow/keras o pytorch |
| Exportacion | csv, pandas |
