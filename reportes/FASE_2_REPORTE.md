# Reporte de Fase 2 - Visualizacion en Tiempo Real

## Servicio Social: Analisis de Redes Neuronales para Deteccion Temprana de Parkinson

**Estado:** COMPLETADA

---

## 1. Objetivo de la Fase

Replicar la interfaz de visualizacion de MindView en Python, mostrando en tiempo real:

- Panel superior: onda cruda (raw wave) superpuesta sobre las 8 bandas de frecuencia EEG (modo superpuesto, no acumulado)
- Panel inferior: Noise, Attention y Meditation como areas coloreadas (escala 0-100)
- Mismos rangos, colores, leyendas y hora real en el eje X que MindView

**Resultado:** Objetivo cumplido al 100%. La aplicacion Python replica la funcionalidad
de MindView con mejoras adicionales (selector de ventana temporal, estadisticas en vivo,
tema oscuro, modo demo). Validado con prueba simultanea contra MindView usando VSPE.

---

## 2. Actividades Realizadas

### 2.1 Evaluacion de librerias de visualizacion

Se evaluo la libreria inicialmente planeada (`matplotlib`) y se descarto a favor de
`PyQtGraph` + `PyQt6`. Ver seccion 4.1 para la justificacion detallada.

### 2.2 Implementacion del buffer circular thread-safe

Se creo `buffer_datos.py` con la clase `BufferEEG`:

- Buffers circulares basados en NumPy para raw wave (512 Hz) y bandas (~1 Hz)
- Thread-safe con `threading.Lock` para lectura/escritura concurrente
- Timestamps almacenados como epoch time (`time.time()`) para mostrar hora real
- Consultas eficientes por ventana temporal (ultimos N segundos)

### 2.3 Implementacion de la interfaz grafica

Se creo `graficas.py` con la ventana principal `VentanaEEG`:

- **Panel superior:** 8 bandas EEG superpuestas (cada una desde cero, no acumuladas) con colores diferenciados,
  mas la raw wave superpuesta en un eje Y derecho independiente
- **Panel inferior:** Noise, Attention y Meditation como areas semi-transparentes
- **Eje X personalizado:** Clase `EjeHoraReal` que extiende `pg.AxisItem` para mostrar
  la hora real (HH:MM:SS) en vez de timestamps epoch o segundos relativos
- **Controles:** Selector de ventana (10/30/60/120 seg), boton pausa/reanudar
- **Barra de estadisticas:** Paquetes, tiempo, senal, Att, Med, Raw/s, Band/s, Pkt/s, FPS

### 2.4 Implementacion de la aplicacion principal

Se creo `app.py` con:

- `LectorVisualizacion`: conecta el parser TGAM al buffer de graficas
- `MonitorFlujo`: calcula tasas de llegada en ventana movil para diagnostico
- `hilo_demo`: genera datos EEG simulados realistas (ondas sinusoidales + ruido)
- `hilo_lectura_serial`: lectura del puerto COM en hilo dedicado
- Tema oscuro completo con paleta Fusion de Qt

### 2.5 Pruebas con modo demo

Primera ronda de pruebas usando datos sinteticos:

- Verificada la fluidez de renderizado (~30 FPS)
- Verificada la superposicion correcta de las 8 bandas
- Verificados los controles de ventana temporal y pausa

**Captura de referencia:** `Capturas/Demo.png`

### 2.6 Pruebas con dispositivo fisico

Segunda ronda de pruebas conectando al TGAM real:

**Problemas detectados:**

1. La banda delta domina con valores de millones mientras las demas bandas son de miles,
   aplastando visualmente todas las demas bandas contra el eje X
2. El eje X mostraba segundos transcurridos en vez de hora real (HH:MM:SS) como MindView

**Captura del problema:** `Capturas/dispositivo.png` - Se observa que solo se ve la raw
wave en naranja con escala descontrolada, y las bandas son invisibles por la escala

### 2.7 Correcciones post-prueba

**Fix 1 - Escalado con percentil 95:**
Se reemplazo el escalado Y basado en `max()` por un enfoque de percentil:

```python
y_p95 = np.percentile(acumulado_prev, 95)
y_techo = max(y_p95 * 1.3, y_max * 0.5)
y_techo = max(y_techo, 1000)  # minimo visible
```

Esto permite que picos extremos de delta se clipeen en vez de aplastar toda la grafica.

**Fix 2 - Eje X con hora real:**
Se cambio el almacenamiento de timestamps de relativos a epoch (`time.time()`) en todo
el pipeline (buffer, app, demo). Se creo la clase `EjeHoraReal` que convierte epoch a
HH:MM:SS para mostrar en el eje.

**Fix 3 - Raw wave semi-transparente:**
Se aplico `alpha=120` a la raw wave y rango fijo Y de -1500 a 1500 para que no tape
las bandas.

### 2.8 Prueba simultanea con MindView (validacion)

Para validar que la aplicacion Python muestra exactamente los mismos datos que MindView,
se realizo una prueba simultanea donde ambas aplicaciones leian del mismo dispositivo TGAM
al mismo tiempo.

**Proceso de la prueba simultanea:**

1. **Intento 1 - com0com + hub4com (fallido):**
   - Se instalo com0com para crear puertos COM virtuales emparejados (COM6 <-> COM7)
   - El driver de com0com no se activaba correctamente en Windows 11 con Secure Boot
   - Los dispositivos aparecian con estado "Error" en el administrador de dispositivos
   - Se descarto esta solucion

2. **Intento 2 - VSPE (exitoso):**
   - Se instalo VSPE (Virtual Serial Port Emulator) de Eterlogic
   - Se configuro un Splitter: COM5 (TGAM fisico) -> COM8 + COM9 (virtuales)
   - MindView se conecto a COM8, Python a COM9
   - Ambas aplicaciones recibieron los mismos bytes en tiempo real

**Resultado:** Las capturas lado a lado (`Capturas/Combibnado1.png` a `Combibnado4.png`,
`Capturas/FInal Grafica.png` y `FInal Grafica2.png`) confirman que ambas aplicaciones
muestran los mismos patrones de bandas, raw wave, attention y meditation con los mismos
valores en los mismos momentos.

### 2.9 Correccion de bug critico de compresion temporal (post-validacion)

Durante las pruebas con hardware real se detecto un bug adicional: en algunos periodos,
las graficas de bandas/metricas aparecian en una franja minima al final del eje X.

**Causa raiz tecnica:**

- El callback del parser se ejecuta por cada paquete valido (muchos son paquetes raw a alta frecuencia).
- `DatosTGAM` conserva el ultimo valor conocido de cada campo.
- La logica de visualizacion agregaba bandas usando condiciones basadas en valores persistidos,
  por lo que terminaba insertando metricas lentas en demasiados paquetes.
- Esto inundaba el buffer lento y comprimian la ventana temporal visible.

**Fix aplicado:**

- Se agregaron flags por paquete en `DatosTGAM`:
  - `actualizo_raw_wave`, `actualizo_eeg_power`, `actualizo_attention`,
    `actualizo_meditation`, `actualizo_poor_signal`, `actualizo_blink_strength`
- La app solo agrega raw cuando `actualizo_raw_wave` es `True`.
- La app solo agrega metricas lentas cuando `actualizo_metricas_lentas` es `True`.
- Se aumento `fs_bandas` de 1 a 8 en `BufferEEG` para mayor robustez ante rafagas reales.

**Resultado del fix:**

- Desaparece la compresion de la grafica en "un cachito".
- La ventana temporal se mantiene estable con datos reales.
- Las tasas en vivo (`Raw/s`, `Band/s`, `Pkt/s`) quedan coherentes con el stream.

### 2.10 Ajuste de escala visual para alinearse con MindView

Tras la comparacion lado a lado se observo que, aunque los datos coincidian,
la representacion visual en Python podia verse "mas grande" que en MindView.

**Causa visual:**

- El modo apilado suma bandas y puede sobredimensionar la altura total.

**Fix aplicado:**

- Cambio de bandas a modo superpuesto (no acumulado).
- Autoescala robusta con percentil 98 + margen y suavizado temporal.
- Desactivacion de prefijos SI en eje de tiempo para evitar notacion cientifica distractora.

**Resultado:**

- Mejor correspondencia visual con MindView en magnitud relativa por banda.
- Menos saltos bruscos de escala entre frames.

---

## 3. Hallazgos y Observaciones

### 3.1 Limitaciones de matplotlib para visualizacion EEG en tiempo real

`matplotlib.animation.FuncAnimation` no pudo mantener un framerate fluido con:

- 512 muestras/seg de raw wave
- 8 bandas apiladas actualizandose ~1/seg
- Panel inferior con 3 series de datos

El principal cuello de botella es que matplotlib redibuja toda la figura en cada frame
(incluyendo ejes, leyendas, fondos) porque es una libreria de graficas estaticas adaptada
para animacion, no una herramienta de renderizado en tiempo real.

En cambio, PyQtGraph renderiza directamente mediante Qt (con posibilidad de aceleracion
OpenGL), actualizando solo los datos que cambian sin redibujar el frame completo.

### 3.2 Comportamiento de las bandas EEG con datos reales

Con el dispositivo fisico se observo que:

- **Delta** domina con valores de 100,000 a 5,000,000+ (tipico en adultos en reposo)
- **Theta** fluctua entre 5,000 y 200,000
- **Alpha** (low+high) entre 2,000 y 100,000
- **Beta** (low+high) entre 1,000 y 50,000
- **Gamma** (low+high) entre 100 y 20,000

Esta distribucion es normal fisiologicamente (mayor potencia en bandas bajas), pero
presenta un reto de visualizacion: un escalado lineal basado en max() hace que solo
se vea delta y las demas bandas sean invisibles.

La solucion del percentil 95 es un compromiso: los picos extremos de delta se clipean
visualmente, pero se preserva la proporcion relativa entre las bandas medias y altas.

### 3.3 Diferencias visuales respecto a MindView

| Aspecto           | MindView                          | Python (nuestro)                         |
| ----------------- | --------------------------------- | ---------------------------------------- |
| Fondo             | Blanco/gris claro                 | Oscuro (#1E1E1E)                         |
| Bandas            | Areas grises solapadas            | Areas coloreadas superpuestas con bordes |
| Raw wave          | Naranja opaca, domina visualmente | Naranja semi-transparente (alpha=120)    |
| Eje X             | Hora real HH:MM:SS                | Hora real HH:MM:SS (igual)               |
| Ventana temporal  | Fija                              | Selector: 10/30/60/120 seg               |
| Estadisticas      | Solo barra inferior basica        | Paquetes, Raw/s, Band/s, FPS, Signal     |
| Escalado Y bandas | Estable y moderado                | Autoescala robusta (p98 + suavizado)     |
| Tema              | Clasico Windows                   | Fusion oscuro                            |

### 3.4 Dificultades con puertos COM virtuales en Windows 11

com0com es la herramienta estandar de codigo abierto para puertos COM virtuales en Windows,
pero su driver no esta firmado para Secure Boot de Windows 11. Los dispositivos se crean
pero quedan en estado "Error". VSPE de Eterlogic resolvio esto inmediatamente al usar
un driver del kernel firmado y compatible con Windows 11.

---

## 4. Decisiones Tecnicas y Justificacion

### 4.1 PyQtGraph + PyQt6 en vez de matplotlib

**Decision:** Usar PyQtGraph con PyQt6 como motor de visualizacion en tiempo real.

**Justificacion:**

- matplotlib redibuja toda la figura en cada frame, causando lag visible a 512 Hz
- PyQtGraph actualiza solo los datos modificados, manteniendo >30 FPS estables
- PyQt6 proporciona una interfaz nativa de Windows con controles (combo, boton, labels)
- Se integra naturalmente con el loop de eventos de Qt (timer de 33ms = ~30 FPS)
- Es la libreria estandar de facto para visualizacion cientifica en tiempo real en Python

### 4.2 Buffer circular con NumPy en vez de listas de Python

**Decision:** Implementar buffers circulares con arrays NumPy de tamano fijo.

**Justificacion:**

- Las listas de Python crecen indefinidamente y requieren `pop(0)` que es O(n)
- NumPy permite operaciones vectorizadas (slicing, percentiles) sin copias
- El tamano fijo garantiza uso de memoria constante (~2.5 MB para 180 seg a 512 Hz)
- Thread-safe con un solo `threading.Lock` por operacion

### 4.3 Timestamps epoch en vez de relativos

**Decision:** Almacenar `time.time()` (epoch) en vez de segundos desde el inicio.

**Justificacion:**

- MindView muestra la hora real del reloj (ej. "12:37:22"), no "45 seg"
- Los timestamps epoch se convierten facilmente a HH:MM:SS con `time.localtime()`
- Permite correlacionar datos con eventos externos (logs, notas del usuario)
- El eje personalizado `EjeHoraReal` maneja la conversion en el momento de renderizar

### 4.4 Escalado adaptativo de bandas (evolucion p95 -> p98 + suavizado)

**Decision:** Iniciar con enfoque p95 y evolucionar a autoescala robusta con p98,
maximo y suavizado temporal.

**Justificacion:**

- Con `max()`: picos extremos de delta (millones) aplasta todas las demas bandas
- Con escala fija: no se adapta a diferentes sujetos o estados de actividad
- Con p95 se resolvio la primera etapa (delta no aplasta todo el grafico)
- Con p98 + maximo + suavizado se mejora la comparacion visual con MindView
- El suavizado evita saltos bruscos de escala entre frames consecutivos

### 4.5 Raw wave semi-transparente con rango Y fijo

**Decision:** Aplicar alpha=120 a la raw wave y fijar el rango Y a [-1500, 1500].

**Justificacion:**

- Sin transparencia, la raw wave (512 Hz, lineas densas) tapa completamente las bandas
- Alpha=120 permite ver las bandas a traves de la raw wave
- El rango fijo [-1500, 1500] coincide con el de MindView y evita que valores atipicos
  compriman la escala. Los valores tipicos con buen contacto estan entre -500 y 500

### 4.6 Modo demo con ondas sinusoidales realistas

**Decision:** Incluir un modo `--demo` que genera datos EEG sinteticos.

**Justificacion:**

- Permite desarrollar y probar la interfaz sin el dispositivo fisico conectado
- Los datos simulados usan frecuencias reales de EEG (1.5 Hz delta, 5 Hz theta, etc.)
- Las magnitudes replican la distribucion real (delta >> theta >> alpha >> beta >> gamma)
- Util para demostraciones en presentaciones sin depender del hardware

### 4.7 VSPE para prueba simultanea en vez de com0com

**Decision:** Usar VSPE (Virtual Serial Port Emulator) para la prueba simultanea.

**Justificacion:**

- com0com requiere un driver no firmado que Windows 11 bloquea con Secure Boot
- VSPE usa un driver firmado compatible con Windows 11 sin necesidad de deshabilitar Secure Boot
- La configuracion del splitter es visual e inmediata (COM5 -> COM8, COM9)
- Se considero tambien WSL + socat, pero VSPE fue mas directo y sin dependencias adicionales

### 4.8 Flags por paquete para desacoplar raw y metricas lentas

**Decision:** Marcar explicitamente que campos se actualizaron en cada paquete TGAM y
consumir solo esos campos en la capa de visualizacion.

**Justificacion:**

- Evita interpretar valores persistidos como datos nuevos.
- Previene la saturacion del buffer lento por paquetes raw.
- Mejora la fidelidad temporal de la grafica respecto al stream real.

### 4.9 Bandas superpuestas en lugar de apiladas para comparacion con MindView

**Decision:** Renderizar bandas en modo superpuesto (cada banda desde cero) para la vista
principal de comparacion.

**Justificacion:**

- MindView se percibe visualmente como series solapadas, no suma acumulativa coloreada.
- Reduce la sobredimension visual de picos cuando Delta domina.
- Facilita comparar tamano relativo de bandas (ej. amarillas/verde) entre ambas apps.

---

## 5. Problemas Encontrados y Soluciones

| Problema                               | Causa                                                        | Solucion                                                         |
| -------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------- |
| matplotlib no mantiene 30 FPS          | Redibuja toda la figura en cada frame                        | Migrar a PyQtGraph + PyQt6                                       |
| `PlotItem.setBackground()` no existe   | Se confundio PlotItem con GraphicsLayoutWidget               | Poner fondo en `GraphicsLayoutWidget.setBackground()`            |
| `OSError: Invalid argument` en eje X   | `time.localtime(0)` falla en Windows al iniciar              | Verificar `v > 946684800` antes de convertir                     |
| Delta aplasta las demas bandas         | Valores de millones vs miles                                 | Escalado adaptativo robusto (p95 inicial, p98 final + suavizado) |
| Grafica en "cachito" al final          | Buffer lento inundado por paquetes raw + valores persistidos | Flags por paquete + insercion condicional + `fs_bandas=8`        |
| Bandas se ven mas grandes que MindView | Modo apilado acumulaba amplitudes                            | Cambiar a modo superpuesto + autoescala robusta                  |
| Eje X mostraba segundos relativos      | Timestamps almacenados como offset desde inicio              | Cambiar a epoch time + eje HH:MM:SS personalizado                |
| Raw wave tapa las bandas               | Opacidad 100% y lineas muy densas a 512 Hz                   | Alpha=120 + rango fijo [-1500, 1500]                             |
| Lag con >15,000 puntos de raw          | 512 Hz x 30 seg = 15,360 puntos por frame                    | Diezmado a maximo 5,000 puntos visibles                          |
| com0com no funciona en Win 11          | Driver no firmado, bloqueado por Secure Boot                 | Usar VSPE (driver firmado)                                       |
| COM6/COM7 ocupados por Bluetooth       | Windows reserva puertos para dispositivos BT                 | Usar COM8/COM9 en VSPE                                           |
| `ModuleNotFoundError: PyQt6`           | Dependencias no instaladas                                   | `pip install PyQt6 pyqtgraph numpy pyserial`                     |

---

## 6. Capturas de Referencia

Las siguientes capturas documentan la evolucion de la fase 2:

| Captura                        | Descripcion                                               |
| ------------------------------ | --------------------------------------------------------- |
| `Capturas/MindviewGrafica.png` | MindView original (referencia a replicar)                 |
| `Capturas/prueba2graficas.png` | Primer prototipo con matplotlib (descartado)              |
| `Capturas/prueba4graficas.png` | Segundo intento con matplotlib (aun lento)                |
| `Capturas/Demo.png`            | Modo demo con PyQtGraph funcionando correctamente         |
| `Capturas/dispositivo.png`     | Prueba con dispositivo real - problema de escala          |
| `Capturas/FInalGrafica.png`    | Version final: Python (izq) vs MindView (der) simultaneo  |
| `Capturas/FInalGrafica2.png`   | Version final: otra captura lado a lado                   |
| `Capturas/Combibnado1.png`     | Comparacion simultanea #1: MindView (izq) vs Python (der) |
| `Capturas/Combibnado2.png`     | Comparacion simultanea #2                                 |
| `Capturas/Combibnado3.png`     | Comparacion simultanea #3                                 |
| `Capturas/Combibnado4.png`     | Comparacion simultanea #4: bandas bien diferenciadas      |

---

## 7. Archivos Creados o Modificados

| Archivo                             | Descripcion                                           |
| ----------------------------------- | ----------------------------------------------------- |
| `src/visualizacion/__init__.py`     | Init del paquete visualizacion                        |
| `src/visualizacion/buffer_datos.py` | Buffer circular thread-safe con NumPy                 |
| `src/visualizacion/graficas.py`     | Ventana PyQtGraph con dos paneles de graficas         |
| `src/visualizacion/app.py`          | Aplicacion principal (lector, demo, monitor de flujo) |
| `visualizar_eeg.py`                 | Lanzador desde la raiz del proyecto                   |
| `requirements.txt`                  | Actualizado con PyQt6, pyqtgraph, numpy               |

## 8. Dependencias Instaladas

| Paquete     | Version | Proposito                                                |
| ----------- | ------- | -------------------------------------------------------- |
| `PyQt6`     | >=6.5   | Framework GUI nativo para la ventana y controles         |
| `pyqtgraph` | >=0.13  | Renderizado de graficas en tiempo real (GPU-accelerated) |
| `numpy`     | >=1.24  | Buffers circulares y operaciones vectorizadas            |

**Herramientas externas usadas para testing:**

| Herramienta                         | Proposito                                                    |
| ----------------------------------- | ------------------------------------------------------------ |
| VSPE (Virtual Serial Port Emulator) | Splitter COM5 -> COM8 + COM9 para prueba simultanea          |
| com0com (descartado)                | Puertos COM virtuales - no compatible con Win 11 Secure Boot |

---

## 9. Decision Arquitectural: Estrategia de Datos para ML

Al cerrar esta fase se analizo y definio como se van a usar los datos capturados para
entrenar las redes neuronales en fases 4 y 5. Esta decision afecta directamente el
diseno del CSV de la fase 3, por lo que se documenta aqui.

### Pregunta debatida

¿Se alimenta la red neuronal con **imagenes** (espectrogramas del EEG) o con los
**datos numericos planos** (series temporales de bandas y raw)?

### Decision: enfoque hibrido

Se entrenaran y compararan ambas familias de modelos:

- **Rama A - Series temporales:** LSTM, GRU, Transformer sobre ventanas de las 8 bandas
  y/o raw EEG normalizado.
- **Rama B - Espectrogramas:** CNN 2D y CNN+LSTM hibrida sobre la transformada STFT del
  raw EEG, generada desde el CSV en fase 4 (nunca desde capturas de pantalla).

### Justificacion

El TGAM tiene **1 solo electrodo**. Con tan poca dimensionalidad espacial, ninguno de
los dos enfoques es claramente superior a priori:

- El **espectrograma STFT** del raw EEG captura patrones tiempo-frecuencia que los vectores
  de bandas no representan directamente (ej. fragmentacion ritmica, marcadores de temblor
  a 4-6 Hz tipicos de Parkinson).
- Los **modelos de series temporales** son mas ligeros, mas interpretables, y varios papers
  con EEG de 1 canal consiguen >90% de accuracy con LSTM sin necesidad de imagenes.

El enfoque hibrido permite un **analisis comparativo rigoroso**: si CNN gana, hay
justificacion tecnica para el costo extra. Si LSTM gana, se tiene evidencia de que la
representacion numerica es suficiente para este dispositivo.

### Impacto en el diseno del CSV (fase 3)

Para soportar ambas ramas se necesitan dos archivos por sesion:

| Archivo                 | Contenido                                  | Uso                       |
| ----------------------- | ------------------------------------------ | ------------------------- |
| `sesion_..._raw.csv`    | timestamp + raw_eeg a 512 Hz + poor_signal | Rama B: generar STFT      |
| `sesion_..._bandas.csv` | timestamp + 8 bandas + eSense a ~1 Hz      | Rama A: series temporales |

El raw EEG a 512 Hz es indispensable para Rama B (no se puede reconstruir desde las bandas),
mientras que para Rama A las bandas pre-calculadas son suficientes y mas ligeras.

### Separacion critica: visualizacion vs ML

La interfaz PyQtGraph de la fase 2 y el pipeline de ML son completamente independientes:
los espectrogramas se generan desde el CSV numerico en fase 4, **no** desde la pantalla.
Esto garantiza precision float completa sin degradacion por renderizado.

---

## 10. Siguiente Fase

**Fase 3 - Exportacion a CSV orientada a ML:** Implementar la grabacion de sesiones EEG
en dos archivos CSV por sesion (raw a 512 Hz y bandas a ~1 Hz), con duracion configurable,
metadatos en header, y boton de grabacion integrado en la interfaz de visualizacion.
El formato esta disenado para alimentar directamente ambas ramas del pipeline de ML
(series temporales y generacion de espectrogramas) sin transformaciones adicionales.
