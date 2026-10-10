# Reporte de Fase 3 - Exportacion a CSV Orientada a ML

## Servicio Social: Analisis de Redes Neuronales para Deteccion Temprana de Parkinson

**Estado:** COMPLETADA

---

## 1. Objetivo de la Fase

Implementar un grabador de sesiones EEG que genere dos archivos CSV por sesion,
listos para alimentar directamente las dos ramas del pipeline de aprendizaje automatico
definido en la estrategia hibrida (ver Fase 2, seccion 9):

- `sesion_YYYY-MM-DD_HH-MM-SS_raw.csv` — raw EEG a 512 Hz
- `sesion_YYYY-MM-DD_HH-MM-SS_bandas.csv` — 8 bandas + eSense a ~1 Hz

**Resultado:** Objetivo cumplido al 100%. El grabador funciona en modo dispositivo
real y en modo demo, con escritura incremental thread-safe, filtrado de senal mala,
duracion configurable y controles integrados en la interfaz de visualizacion existente.

---

## 2. Por que dos archivos separados

Esta es la decision de diseno central de la fase, y vale la pena documentarla con claridad
porque impacta directamente en como se construyen los modelos en las fases 4 y 5.

### 2.1 `sesion_..._raw.csv` — para la Rama B (CNN sobre espectrogramas)

El raw EEG es la senal electrica cruda del cerebro muestreada a **512 Hz**
(512 valores por segundo). Esta alta frecuencia es indispensable para generar
espectrogramas con buena resolucion en frecuencia mediante la transformada STFT.

**Por que es necesario para CNN:**

El espectrograma STFT convierte una ventana de raw EEG en una "imagen" de patrones
tiempo-frecuencia. La CNN aprende a reconocer visualmente si esos patrones corresponden
a un cerebro sano o con Parkinson.

Un marcador tipico de Parkinson visible en el espectrograma es la **fragmentacion del
ritmo a 4-6 Hz** (banda theta baja): en lugar de una banda continua, aparece como
bloques cortos y discontinuos. Este patron se pierde si se usan las bandas pre-calculadas
del TGAM, porque el TGAM ya promedeo esa informacion en un solo numero por segundo.

Sin el raw EEG a 512 Hz, simplemente no es posible construir los espectrogramas que
necesita la CNN. No se puede reconstruir la senal cruda a partir de las bandas.

**Que contiene:**
```
timestamp_epoch, timestamp_hora, raw_eeg, poor_signal
```

- `timestamp_epoch`: tiempo exacto de la muestra (float, epoch Unix)
- `timestamp_hora`: hora legible HH:MM:SS.mmm (para depuracion visual)
- `raw_eeg`: valor signed 16-bit de la senal cruda (-32768 a 32767, tipico -500 a 500)
- `poor_signal`: calidad de senal en esa muestra (solo se guardan muestras con < 50)

### 2.2 `sesion_..._bandas.csv` — para la Rama A (LSTM/GRU/Transformer)

Las bandas son los 8 valores de potencia espectral que el TGAM calcula internamente,
actualizandose a **~1 Hz** (un vector de 8 valores por segundo). Junto con los valores
de eSense (Attention, Meditation) forman el vector de entrada para los modelos de
series temporales.

**Por que es suficiente para LSTM/GRU/Transformer:**

Estos modelos aprenden dependencias temporales: como cambia la potencia de cada banda
a lo largo del tiempo. Para Parkinson, patrones relevantes incluyen:

- Incremento relativo de potencia en delta y theta respecto a alpha/beta (enlentecimiento)
- Reduccion de potencia en beta (banda asociada a control motor)
- Variabilidad anormal en la relacion theta/alpha a lo largo del tiempo

Estos patrones son detectables en series temporales de 8 bandas sin necesitar
la senal cruda completa. Varios papers con EEG de 1 canal reportan >90% de accuracy
con LSTM usando solo el vector de bandas normalizado.

**Ventaja adicional:** El archivo de bandas es aproximadamente 512 veces mas pequeno
que el raw. Una sesion de 5 minutos genera ~15,000 filas en raw vs ~300 filas en bandas.
Esto hace que el entrenamiento de la Rama A sea mucho mas rapido y menos demandante.

**Que contiene:**
```
timestamp_epoch, timestamp_hora, delta, theta, low_alpha, high_alpha,
low_beta, high_beta, low_gamma, high_gamma, attention, meditation, poor_signal
```

- Las 8 bandas son valores unsigned enteros (tipicamente miles a millones para delta)
- `attention` y `meditation`: valores eSense 0-100 calculados por el TGAM
- Solo se guardan filas con `poor_signal < 50`

### 2.3 Resumen: que hace cada archivo en el pipeline

```
sesion_raw.csv (512 Hz)
    |
    v  Fase 4: ventana de 2 seg = 1024 samples -> STFT -> array NumPy (64 x 128)
    v
    CNN 2D aprende patrones tiempo-frecuencia (Rama B)

sesion_bandas.csv (~1 Hz)
    |
    v  Fase 4: ventana deslizante de 10 seg = 10 filas -> normalizacion MinMax
    v
    LSTM/GRU/Transformer aprende evolucion temporal de las bandas (Rama A)
```

Los dos archivos son independientes entre si. Cada uno alimenta su propia rama
sin necesitar al otro. Compartir un unico archivo habria mezclado frecuencias
incompatibles (512 Hz y ~1 Hz) y complicado el preprocesamiento.

---

## 3. Actividades Realizadas

### 3.1 Implementacion del grabador (`src/exportacion/grabador_csv.py`)

Se implemento la clase `GrabadorCSV` con las siguientes caracteristicas:

**Escritura incremental con hilo dedicado:**

El grabador no acumula datos en RAM. Usa una arquitectura productor-consumidor:

- El hilo principal (lector serial o demo) encola filas en `Queue` thread-safe
- Un hilo dedicado de escritura consume la cola y vuelca al disco en bloques de 100 filas
- Flush final garantizado al llamar `detener()`

Esto evita que la escritura a disco bloquee el hilo de lectura serial, lo cual podria
causar perdida de muestras a 512 Hz.

**Filtrado de senal invalida:**

Solo se graban muestras con `poor_signal < 50`. Las muestras con senal mala contienen
artefactos (movimiento, mal contacto) que contaminarian el dataset de entrenamiento.
El umbral 50 se define como constante `UMBRAL_SENAL` en el modulo.

**Duraciones configurables:**

| Opcion | Segundos |
|--------|----------|
| 30s | 30 |
| 2min | 120 |
| 5min | 300 |
| 10min | 600 |
| personalizado | Ingresado por el usuario |
| sin limite | Manual (hasta presionar Detener) |

**Headers con metadatos:**

Cada CSV incluye comentarios en las primeras filas con:
- Fecha y hora de inicio de la sesion
- Puerto COM del dispositivo
- Frecuencia de muestreo
- Version del software
- Umbral de poor_signal aplicado

Esto hace los archivos autoexplicativos: quien los abra en 6 meses sabra exactamente
cuando y con que condiciones fueron capturados.

**Nombre automatico de archivos:**

`sesion_YYYY-MM-DD_HH-MM-SS_raw.csv` y `sesion_YYYY-MM-DD_HH-MM-SS_bandas.csv`

El sello de tiempo en el nombre permite tener multiples sesiones en la misma carpeta
sin colisiones y ordenarlas cronologicamente por nombre.

### 3.2 Integracion en la interfaz de visualizacion

Se modificaron `graficas.py` y `app.py` para integrar los controles de grabacion
en la barra de herramientas existente de `VentanaEEG`:

**Nuevos controles en la barra superior:**

- Selector de duracion (ComboBox: 30s / 2min / 5min / 10min / personalizado / sin limite)
- Boton **Grabar** (toggle: activa/detiene la sesion)
- Barra de progreso roja (avanza segun la duracion configurada)
- Label **REC MM:SS** con el tiempo transcurrido en rojo

**Flujo de grabacion:**

1. Usuario selecciona duracion y presiona **Grabar**
2. Se crea un `GrabadorCSV` y se llama a `iniciar(puerto_com=...)`
3. El hilo de datos (real o demo) llama a `grabador.agregar_raw()` y `grabador.agregar_bandas()`
   con cada paquete nuevo
4. La ventana consulta `VentanaEEG.grabador` para obtener el grabador activo
5. Al terminar (manual o por duracion), se llama `detener()` y aparece un dialogo
   con las rutas y el conteo de muestras grabadas

**Propiedad `VentanaEEG.grabador`:**

Expone el grabador activo como propiedad de solo lectura para que el hilo de datos
lo consulte sin acoplamiento directo. Retorna `None` si no hay sesion activa.

### 3.3 Pruebas

**Prueba funcional en modo demo:**

- Se ejecuto `python visualizar_eeg.py --demo`
- Se inicio grabacion de 30 segundos
- Se verifico que los archivos se generaron con el nombre correcto
- Se verifico que el header contiene los metadatos correctos
- Se verifico que las 512 muestras/seg de raw se grabaron correctamente
- Se verifico que las bandas aparecen a ~1 Hz (1 fila por segundo)
- Se verifico que el filtro de poor_signal funciona: muestras con ps>=50 no se graban
- Se verifico que la grabacion se detiene automaticamente al cumplirse la duracion
- Se verifico el dialogo de confirmacion con rutas y conteos correctos

---

## 4. Decisiones Tecnicas y Justificacion

### 4.1 Cola thread-safe en lugar de lock compartido

**Decision:** Usar `queue.Queue` para comunicar el hilo de lectura con el hilo de escritura.

**Justificacion:**

- `Queue` en Python es thread-safe por diseno (usa internamente un `Lock`)
- El hilo de escritura no bloquea al hilo de lectura: si el disco es lento, la cola
  crece en RAM temporalmente pero el parser no pierde muestras
- La alternativa (lock compartido en el archivo) habria causado que `parse_byte()` esperara
  al disco antes de procesar el siguiente byte, con riesgo de perder samples a 512 Hz

### 4.2 Flush por bloques en lugar de por fila

**Decision:** Acumular 100 filas antes de llamar `file.flush()`.

**Justificacion:**

- Python bufferiza escrituras a disco internamente. Sin flush explicito, al crashear
  se pueden perder hasta el buffer completo (tipicamente 8 KB)
- Flush por cada fila seria demasiado lento: 512 flushes/seg para raw es I/O intensivo
- Bloques de 100 filas equivalen a ~0.2 seg de raw: perdida maxima aceptable si hay crash
- Al llamar `detener()`, se fuerza un flush final antes de cerrar

### 4.3 Umbral poor_signal = 50 (no 0 ni 200)

**Decision:** Filtrar muestras con `poor_signal >= 50`.

**Justificacion:**

- `poor_signal = 0` significa senal perfecta, pero es raro tenerla continuamente
- `poor_signal = 200` significa sin contacto (datos completamente inutiles)
- El TGAM reporta values de 1 a 49 en condiciones normales de uso con buena colocacion
- Un dataset de entrenamiento con muestras de senal mala confundiria al modelo:
  el artefacto de movimiento generaria patrones similares a los de Parkinson
- El umbral 50 es el mismo que usa `DatosTGAM.senal_buena()` en el parser (consistencia)

### 4.4 No guardar espectrogramas en esta fase

**Decision:** El CSV raw es la unica fuente de verdad. Los espectrogramas se generan
en fase 4, nunca aqui.

**Justificacion:**

- Guardar espectrogramas como PNG introduciria perdida de precision por cuantizacion
- Guardar espectrogramas como arrays NumPy aqui duplicaria el almacenamiento sin beneficio
  (se pueden regenerar en cualquier momento desde el CSV raw)
- Separar la captura del preprocesamiento sigue el principio de pipeline limpio:
  si se quiere cambiar los parametros del STFT (tamano de ventana, overlap, tipo de
  ventana), no hay que recapturar datos, solo reprocesar el CSV

### 4.5 Carpeta destino `datos/sesiones/` con creacion automatica

**Decision:** Usar `os.makedirs(carpeta, exist_ok=True)` al iniciar la sesion.

**Justificacion:**

- El usuario no deberia tener que crear la carpeta manualmente
- `exist_ok=True` evita error si ya existe
- La carpeta por defecto `datos/sesiones/` esta dentro de `07_Proyecto_Python/datos/`,
  que es la carpeta designada en el plan para todos los datos del proyecto

### 3.4 Revision de codigo (/simplify)

Se ejecuto una revision de calidad, reuso y eficiencia sobre todos los archivos de la
fase. Se detectaron y corrigieron los siguientes problemas:

**Eficiencia en hot path (512 Hz):**

`datetime.fromtimestamp()` se llamaba dentro de `agregar_raw()` y `agregar_bandas()`,
creando un objeto datetime descartable por cada muestra a 512 Hz. Se movio el formateo
de hora al hilo de escritura dedicado, que corre fuera del path critico del parser.
El productor ahora solo encola el timestamp epoch (un float).

**Doble llamada a `datetime.now()` en `iniciar()`:**

Se calculaba `datetime.now()` dos veces: una para el sello del nombre de archivo y
otra para el header del CSV. Las dos llamadas podian diferir en hasta un segundo.
Se unificaron en una sola llamada al inicio del metodo.

**Propiedad `ventana.grabador` evaluada 512 veces/seg:**

En `on_datos` y en `hilo_demo`, la propiedad `ventana.grabador` se evaluaba de forma
independiente para raw y para bandas (dos consultas por paquete). Se cachea una sola
vez al inicio del callback/iteracion y se reutiliza para ambos tipos de dato.

**Estado muerto `_duracion_grabacion`:**

`VentanaEEG` guardaba la duracion en `self._duracion_grabacion` pero nunca la leia;
`GrabadorCSV` ya gestiona la duracion internamente en `self.duracion_seg`. Se elimino.

**Dos loops sobre la misma lista de labels:**

En `_crear_ui()` habia dos loops consecutivos identicos: uno para asignar fuente y
otro para agregar al layout. Se unificaron en un solo loop.

**f-strings sin interpolacion:**

En los metodos `_escribir_header_raw` y `_escribir_header_bandas`, strings fijos como
`f"# EEG Raw - NeuroSky TGAM"` y `f"# fs_raw=512"` usaban f-string sin ningun `{}`.
Se simplificaron a strings literales normales.

---

## 5. Problemas Encontrados y Soluciones

| Problema | Causa | Solucion |
|----------|-------|----------|
| El hilo de escritura no vaciaba la cola al llamar `detener()` | La condicion del while terminaba antes de vaciar | Agregar `or not cola.empty()` a la condicion del while |
| `file.flush()` al cerrar no garantizaba escritura si el hilo ya habia terminado | Race condition en el join | Mover el flush final dentro del `with open(...)` antes de salir |
| El boton Grabar no se desbloqueaba si la grabacion termino por duracion | `_detener_grabacion()` no se llamaba automaticamente | Agregar verificacion en `_actualizar_indicador_grabacion()`: si `grabador.grabando` es False y el boton esta activo, llamar `_detener_grabacion()` |

---

## 6. Archivos Creados o Modificados

| Archivo | Cambio |
|---------|--------|
| `src/exportacion/__init__.py` | Creado (paquete) |
| `src/exportacion/grabador_csv.py` | Creado — clase `GrabadorCSV` completa |
| `src/visualizacion/graficas.py` | Modificado — controles de grabacion en la barra + propiedad `grabador` |
| `src/visualizacion/app.py` | Modificado — `LectorVisualizacion` y `hilo_demo` alimentan el grabador |
| `datos/sesiones/` | Carpeta creada (destino de los CSVs) |
| `PLAN_DE_IMPLEMENTACION.md` | Fase 3 marcada como COMPLETADA |

---

## 7. Dependencias

No se agregaron dependencias nuevas. Se usa unicamente `csv` de la biblioteca estandar
de Python. `pandas` no fue necesario para la captura (solo podria usarse en fase 4
para validacion y lectura de los archivos).

---

## 8. Siguiente Fase

**Fase 4 - Preprocesamiento de datos:**

Los CSVs generados en esta fase se usaran de la siguiente manera:

- `sesion_..._raw.csv` → ventanas de raw EEG → STFT → espectrogramas NumPy (Rama B)
- `sesion_..._bandas.csv` → ventanas deslizantes → normalizacion → tensores de entrada (Rama A)

Ademas se buscaran e integraran datasets publicos de EEG-Parkinson (PhysioNet, OpenNeuro)
adaptando su formato al mismo esquema de CSV para poder mezclarlos con los datos propios.
