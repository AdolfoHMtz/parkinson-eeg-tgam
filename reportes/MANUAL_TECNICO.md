# Manual Tecnico - Deteccion Temprana de Parkinson con EEG
## Para futuros colaboradores del servicio social

---

## Introduccion

Este manual documenta la implementacion tecnica del proyecto de deteccion temprana de
Parkinson usando datos de electroencefalograma (EEG) capturados con el dispositivo
NeuroSky TGAM y procesados con redes neuronales en Python.

El manual se actualiza al cierre de cada fase del proyecto.

---

## Actualizacion Estrategica

El proyecto migro su estrategia principal de entrenamiento para alinearse mejor con
la compatibilidad real entre el TGAM y el dataset publico OpenNeuro:

- La **senal base comun** ahora es el **raw EEG monocanal**
- Las **bandas del TGAM** se conservan para inspeccion y trazabilidad, pero ya no son
  la entrada principal del entrenamiento cruzado con OpenNeuro
- La **Rama A** prioriza **series temporales raw normalizadas**
- La **Rama B** prioriza **GAF (Gramian Angular Field)** como imagen para CNN 2D
- **STFT/espectrograma** queda disponible solo como baseline legacy/opcional

La evidencia de esta decision se genera con:

```bash
python herramientas/comparar_tgam_vs_openneuro.py
```

Archivos generados:
- `herramientas/compatibilidad_tgam_vs_openneuro/resumen_compatibilidad.json`
- `herramientas/compatibilidad_tgam_vs_openneuro/REPORTE_COMPATIBILIDAD.md`

---

## Requisitos del Sistema

- **Sistema operativo:** Windows 10/11
- **Python:** 3.10 o superior (probado con 3.13.1)
- **Hardware:** Modulo NeuroSky TGAM + Adaptador BT-USB
- **Puerto:** COM serial autodetectable (tipicamente COM5)

## Instalacion

```bash
# 1. Instalar dependencias
cd 07_Proyecto_Python
pip install -r requirements.txt

# 2. Verificar que pyserial esta instalado
python -c "import serial; print(serial.VERSION)"
```

---

## Fase 1 - Lectura del Dispositivo

### Arquitectura

```
leer_tgam.py                    Programa principal
    |
    |-- src/conexion/
    |   |-- serial_manager.py    Autodeteccion COM + conexion serial
    |
    |-- src/parser/
        |-- tgam_parser.py       Parser protocolo binario TGAM
```

### Protocolo de comunicacion TGAM

El dispositivo transmite datos via protocolo binario serial a 57600 baud, 8N1.

**Estructura de paquete:**
```
Byte 0:    0xAA (SYNC)
Byte 1:    0xAA (SYNC)
Byte 2:    PLENGTH (longitud del payload, max 170)
Byte 3..N: PAYLOAD (datos codificados)
Byte N+1:  CHECKSUM (complemento a 1 de la suma del payload, AND 0xFF)
```

**Validacion de checksum:**
```python
checksum_esperado = (~suma_payload) & 0xFF
if byte_recibido != checksum_esperado:
    # paquete corrupto, descartar
```

**Codigos de datos en el payload:**

Cada dato dentro del payload tiene la estructura:
- Codigo < 0x80: valor de 1 byte
- Codigo >= 0x80: siguiente byte indica longitud, luego N bytes de valor

| Codigo | Nombre | Bytes | Rango | Frecuencia |
|--------|--------|-------|-------|------------|
| 0x02 | Poor Signal Level | 1 | 0-200 (0=buena) | ~1/seg |
| 0x04 | Attention | 1 | 0-100 | ~1/seg |
| 0x05 | Meditation | 1 | 0-100 | ~1/seg |
| 0x16 | Blink Strength | 1 | 0-255 | Al parpadear |
| 0x80 | Raw Wave | 2 | signed 16-bit | 512 Hz |
| 0x83 | EEG Power | 24 | 8 x unsigned 24-bit | ~1/seg |

**EEG Power (0x83) - 8 bandas, 3 bytes cada una (big-endian):**

| Posicion | Banda | Rango Hz |
|----------|-------|----------|
| Bytes 0-2 | Delta | 0.5 - 2.75 |
| Bytes 3-5 | Theta | 3.5 - 6.75 |
| Bytes 6-8 | Low Alpha | 7.5 - 9.25 |
| Bytes 9-11 | High Alpha | 10 - 12 |
| Bytes 12-14 | Low Beta | 12.75 - 20 |
| Bytes 15-17 | High Beta | 20.25 - 28 |
| Bytes 18-20 | Low Gamma | 30 - 40 |
| Bytes 21-23 | High Gamma | 40 - 50 |

### Como ejecutar

**Requisitos previos:**
1. Conectar el adaptador BT-USB al puerto USB de la computadora
2. Asegurarse de que el dispositivo TGAM este encendido
3. Cerrar cualquier programa que use el puerto COM (Realterm, MindView)

**Ejecucion:**
```bash
# Autodeteccion de puerto
python leer_tgam.py

# Puerto especifico
python leer_tgam.py COM5

# Solo datos raw (512 Hz, para depuracion)
python leer_tgam.py --raw
```

**Salida esperada (modo normal):**
```
=================================================================
  NEUROSKY TGAM - Lectura en tiempo real
=================================================================

  Calidad de senal:    0/200  (EXCELENTE)

  Attention:     [########------------]  44/100
  Meditation:    [################----]  83/100

  Raw EEG:         -135

  Banda                Valor  Grafica
  --------------------------------------------------
  delta              395,460  #########################
  theta                5,168
  low_alpha            4,593
  high_alpha           2,159
  low_beta             1,217
  high_beta            1,630
  low_gamma              278
  high_gamma             120
```

### Modulos Python

#### serial_manager.py

Funciones disponibles:

| Funcion | Descripcion |
|---------|-------------|
| `listar_puertos()` | Devuelve lista de puertos COM con descripcion |
| `autodetectar_puerto()` | Detecta automaticamente el puerto del TGAM |
| `seleccionar_puerto()` | Autodetecta o pide al usuario elegir |
| `conectar(puerto=None)` | Abre conexion serial (57600, 8N1) |
| `desconectar(ser)` | Cierra la conexion serial |

Constantes: `BAUD_RATE=57600`, `TIMEOUT=1`

#### tgam_parser.py

**Clase `DatosTGAM`** (dataclass):
- Campos: `poor_signal`, `attention`, `meditation`, `blink_strength`, `raw_wave`,
  `delta`, `theta`, `low_alpha`, `high_alpha`, `low_beta`, `high_beta`,
     `low_gamma`, `high_gamma`
- Flags por paquete: `actualizo_poor_signal`, `actualizo_attention`,
     `actualizo_meditation`, `actualizo_blink_strength`, `actualizo_raw_wave`,
     `actualizo_eeg_power`
- Propiedad `actualizo_metricas_lentas`: True cuando el paquete trae EEG power/eSense/senal
- Metodo `bandas_dict()`: devuelve las 8 bandas como diccionario
- Metodo `senal_buena()`: True si poor_signal < 50

**Clase `TGAMParser`**:
- Constructor recibe `on_datos`: callback invocado con `DatosTGAM` por cada paquete valido
- Metodo `parse_byte(byte_val)`: alimentar byte por byte del stream serial
- Atributos `paquetes_ok` y `paquetes_error`: contadores de estadisticas

Ejemplo de uso:
```python
from src.parser.tgam_parser import TGAMParser, DatosTGAM
from src.conexion.serial_manager import conectar

def mi_callback(datos: DatosTGAM):
    print(f"Attention: {datos.attention}, Meditation: {datos.meditation}")

ser = conectar()  # autodetecta puerto
parser = TGAMParser(on_datos=mi_callback)

while True:
    if ser.in_waiting > 0:
        for byte in ser.read(ser.in_waiting):
            parser.parse_byte(byte)
```

### Troubleshooting

| Problema | Posible causa | Solucion |
|----------|--------------|----------|
| "No se encontraron puertos COM" | Adaptador BT-USB desconectado | Conectar el adaptador USB |
| "ERROR al conectar a COMx" | Otro programa usa el puerto | Cerrar Realterm/MindView |
| `poor_signal = 200` permanente | Sensor sin contacto con la piel | Ajustar el electrodo en la frente |
| Datos raw oscilan entre -2048 y 2047 | Sin contacto, senal saturada | Verificar poor_signal, ajustar electrodo |
| Solo aparecen datos raw, sin bandas | Las bandas tardan ~3-5 segundos en aparecer | Esperar a que el dispositivo calibre |
| Error de checksum frecuente | Interferencia electromagnetica o mala conexion | Verificar conexion USB, alejar de fuentes de ruido |
| `pip install pyserial` falla | Sin internet o timeout | Reintentar con `pip install pyserial --timeout 60` |

---

## Fase 2 - Visualizacion en Tiempo Real

### Arquitectura

```
visualizar_eeg.py                   Lanzador (raiz del proyecto)
    |
    |-- src/visualizacion/
        |-- app.py                  Aplicacion principal, hilos de lectura/demo
        |-- graficas.py             Ventana PyQtGraph con dos paneles
        |-- buffer_datos.py         Buffer circular thread-safe con NumPy
```

**Flujo de datos:**
```
Puerto COM (TGAM)
     |
     v
hilo_lectura_serial    (hilo dedicado, lee bytes del puerto)
     |
     v
TGAMParser.parse_byte  (parser binario de la Fase 1)
     |
     v
LectorVisualizacion.on_datos  (callback, almacena en buffer)
     |
     v
BufferEEG              (buffer circular thread-safe, almacena raw + bandas)
     |
     v
VentanaEEG._actualizar (timer Qt cada 33ms, lee buffer y renderiza)
     |
     v
PyQtGraph              (renderizado GPU-accelerated de las graficas)
```

**Nota clave de integridad temporal:**
- `LectorVisualizacion` agrega al buffer solo cuando el paquete realmente actualiza
     ese tipo de dato (usando flags del parser).
- Esto evita inundar el buffer de metricas lentas con paquetes de raw.

### Componentes principales

#### buffer_datos.py - BufferEEG

Buffer circular thread-safe que almacena datos EEG en arrays NumPy de tamano fijo.

| Metodo | Descripcion |
|--------|-------------|
| `agregar_raw(valor, timestamp)` | Agrega un sample de raw wave (512 Hz). Timestamp es epoch |
| `agregar_bandas(datos, timestamp)` | Agrega 8 bandas + eSense (~1 Hz). Timestamp es epoch |
| `obtener_raw(ultimos_seg)` | Retorna (timestamps, valores) de los ultimos N segundos |
| `obtener_bandas(ultimos_seg)` | Retorna (timestamps, bandas_dict, att, med, ps) |
| `tiempo_transcurrido` | Propiedad: segundos desde el inicio |

Parametros del constructor:
- `duracion_seg`: tamano del buffer en segundos (default: 180)
- `fs_raw`: frecuencia de muestreo raw (default: 512)
- `fs_bandas`: frecuencia estimada de bandas (default: 8, holgura para variaciones)

**Nota sobre timestamps:** Todos los timestamps se almacenan como epoch time
(`time.time()`) para poder mostrar la hora real (HH:MM:SS) en los ejes de las graficas,
replicando el comportamiento de MindView.

#### graficas.py - VentanaEEG

Ventana principal con dos paneles de graficas:

**Panel superior (75% del espacio):**
- Eje izquierdo: 8 bandas EEG superpuestas (cada banda desde cero, no acumuladas)
     - Permite comparar mejor magnitud relativa por banda contra MindView
     - Cada banda con color unico y borde visible
- Eje derecho: Raw wave semi-transparente (alpha=120, rango fijo -1500 a 1500)
- Escalado Y adaptativo robusto (percentil 98 + maximo + suavizado temporal)

**Panel inferior (25% del espacio):**
- Noise (amarillo): poor_signal / 2, normalizado a 0-100
- Attention (rojo): valor eSense 0-100
- Meditation (azul): valor eSense 0-100
- Todas como areas semi-transparentes

**Clase EjeHoraReal:**
Extiende `pg.AxisItem` para convertir timestamps epoch a formato HH:MM:SS.
Incluye proteccion contra valores invalidos (epoch < anio 2000) que PyQtGraph puede
generar antes de que lleguen datos reales.
Tambien se desactiva el prefijo SI automatico para evitar formato cientifico en el eje X.

#### app.py - Aplicacion y modos de ejecucion

| Clase/Funcion | Descripcion |
|---------------|-------------|
| `MonitorFlujo` | Calcula tasas de llegada (Raw/s, Band/s, Pkt/s) en ventana movil de 5 seg |
| `LectorVisualizacion` | Conecta el callback del parser TGAM al buffer de graficas usando flags por paquete |
| `hilo_lectura_serial()` | Lee bytes del puerto COM en un hilo dedicado |
| `hilo_demo()` | Genera datos EEG sinteticos realistas (ondas seno + ruido) |
| `main()` | Punto de entrada: crea Qt app, buffer, ventana, lanza hilo |

### Como ejecutar

**Requisitos previos:**
```bash
pip install PyQt6 pyqtgraph numpy pyserial
```

**Ejecucion:**
```bash
# Dispositivo real (autodetecta puerto COM)
python visualizar_eeg.py

# Dispositivo real en puerto especifico
python visualizar_eeg.py COM5

# Modo demo (datos simulados, sin hardware)
python visualizar_eeg.py --demo
```

**Controles de la interfaz:**
- **Ventana:** Selector desplegable para cambiar el rango temporal visible (10/30/60/120 seg)
- **Pausar/Reanudar:** Congela la grafica sin detener la captura de datos
- **Barra de estado:** Muestra calidad de senal con colores (verde=excelente, rojo=sin contacto)
- **Barra inferior:** Paquetes, tiempo, Signal, Att, Med, Raw/s, Band/s, Pkt/s, FPS

### Prueba simultanea con MindView

Para validar que la aplicacion Python muestra los mismos datos que MindView, se puede
ejecutar ambas al mismo tiempo usando un splitter de puertos COM virtuales:

**Herramienta:** VSPE (Virtual Serial Port Emulator) - https://www.eterlogic.com

**Configuracion:**
1. Instalar VSPE
2. Crear un dispositivo tipo "Splitter":
   - Source port: COM5 (TGAM fisico)
   - Destino: COM8 y COM9
3. Iniciar la emulacion (boton Play)
4. Conectar MindView a COM8
5. Ejecutar: `python visualizar_eeg.py COM9`

**Nota:** No usar com0com en Windows 11 con Secure Boot — el driver no firmado
causa que los dispositivos queden en estado "Error". VSPE es la alternativa funcional.

### Colores de las bandas EEG

| Banda | Color | Codigo hex |
|-------|-------|------------|
| Delta | Azul | #4A90D9 |
| Theta | Violeta | #7B68EE |
| Low Alpha | Verde | #50C878 |
| High Alpha | Verde claro | #90EE90 |
| Low Beta | Amarillo | #FFD700 |
| High Beta | Naranja | #FFA500 |
| Low Gamma | Rojo-naranja | #FF6347 |
| High Gamma | Rosa fuerte | #FF1493 |
| Raw Wave | Marron/naranja | #D2691E |

### Troubleshooting

| Problema | Posible causa | Solucion |
|----------|--------------|----------|
| `ModuleNotFoundError: PyQt6` | Dependencias no instaladas | `pip install PyQt6 pyqtgraph numpy` |
| Ventana negra sin datos | Dispositivo no conectado o puerto incorrecto | Verificar COM, o usar `--demo` |
| FPS bajo (<15) | Demasiados puntos sin diezmar | Ya incluido: diezmado automatico a 5000 pts |
| Delta aplasta las bandas | Valores de millones (normal) | Ya incluido: autoescala robusta (p98 + max + suavizado) |
| Bandas demasiado grandes vs MindView | Visualizacion acumulada (apilada) | Ya incluido: modo superpuesto (no acumulado) |
| Grafica se ve en un "cachito" al final | Insercion de metricas lentas en paquetes raw por valores persistidos | Ya incluido: flags por paquete + insercion condicional + `fs_bandas=8` |
| Eje X muestra numeros grandes | Timestamps epoch sin formatear | Ya incluido: `EjeHoraReal` convierte a HH:MM:SS |
| Eje X muestra notacion cientifica | Prefijo SI automatico de PyQtGraph | Ya incluido: `enableAutoSIPrefix(False)` |
| Error al iniciar VSPE splitter | COM5 ocupado por otra app | Cerrar MindView/Realterm antes de crear el splitter |
| com0com devices en "Error" | Driver no firmado en Win 11 | Usar VSPE en su lugar |

### Nota de calibracion visual (comparacion con MindView)

Aunque ambas aplicaciones reciban los mismos bytes via splitter COM, la percepcion visual
puede variar por decisiones de renderizado (alpha, modo de areas, estrategia de autoescala).
Para comparacion justa, priorizar:
- mismas ventanas temporales (30 seg vs 30 seg)
- misma escala raw (-1500 a 1500)
- bandas en modo superpuesto
- revisar tasas en vivo (`Raw/s`, `Band/s`, `Pkt/s`) para confirmar flujo equivalente

---

## Estrategia de Datos para ML

### Arquitectura del pipeline completo

```
TGAM (fisico)
     |
     v
hilo_lectura_serial (pyserial)
     |
     v
TGAMParser (protocolo binario)
     |
     +---> BufferEEG --> VentanaEEG (PyQtGraph)    <- visualizacion para el medico
     |
     +---> GrabadorCSV                              <- fase 3
               |
               +-- sesion_raw.csv     (512 Hz)
               +-- sesion_bandas.csv  (~1 Hz)
                        |
                        v
               Preprocesamiento                     <- fase 4
               /                \
   Series temporales        Imagen temporal GAF
   (raw normalizado)        (desde raw, arrays NumPy)
          |                        |
   1D CNN / LSTM / GRU        CNN 2D / CNN+LSTM      <- fase 5
          \                        /
           Tabla comparativa final
```

### Dos ramas de modelos

**Por que raw + series temporales:**
Es la senal comun real entre TGAM y OpenNeuro-Fp1. Permite comparar dominios sin
depender de calculos propietarios del chip TGAM.

**Por que GAF para la rama de imagen:**
GAF nace directamente de la serie temporal y conserva relaciones globales entre
muestras, lo cual es mas coherente con el objetivo de usar la misma senal base en
ambas ramas del proyecto.

**Por que se conserva STFT solo como legado:**
Puede servir como baseline historico, pero ya no es la representacion principal del
proyecto porque el enfoque actual prioriza consistencia entre dominio TGAM y dominio
OpenNeuro en un escenario de 1 solo electrodo.

### Formato de los CSVs (fase 3)

**`sesion_YYYY-MM-DD_HH-MM-SS_raw.csv`** — raw EEG a 512 Hz:
```
# EEG Raw - NeuroSky TGAM
# fecha=2026-04-21 16:46:26
# puerto_com=autodetectado
# etiqueta_local=sano
# fs_raw=512
# version_sw=3.0
# umbral_senal=50
timestamp_epoch,timestamp_hora,raw_eeg,poor_signal
1776811586.266,16:46:26.266,59,0
...
```

**`sesion_YYYY-MM-DD_HH-MM-SS_bandas.csv`** — bandas + eSense a ~1 Hz:
```
# EEG Bandas + eSense - NeuroSky TGAM
# fecha=2026-04-21 16:46:26
# puerto_com=autodetectado
# etiqueta_local=sano
# fs_bandas=~1Hz
# version_sw=3.0
# umbral_senal=50
timestamp_epoch,timestamp_hora,delta,theta,low_alpha,high_alpha,low_beta,high_beta,low_gamma,high_gamma,attention,meditation,poor_signal
1776811587.159,16:46:27.159,153745,157742,19259,21889,12936,41469,86258,1617,40,44,0
...
```

Ademas, cada carpeta de sesiones tiene un `etiquetas_locales.csv` con una fila por sesion:

```
sello,etiqueta,fecha,puerto_com,archivo_raw,archivo_bandas,sujeto
2026-04-21_16-46-26,sano,2026-04-21 16:46:26,autodetectado,sesion_..._raw.csv,sesion_..._bandas.csv,local_01
```

La columna `sujeto` identifica a la **persona** grabada. Todas las sesiones con el mismo
`sujeto` caen en el mismo conjunto (train, val o test) al dividir los datos. Si falta, se
asume `local_01` (todas las sesiones son de la misma persona), que es la opcion segura.

### Nota sobre GAF

La representacion principal para CNN ya no son espectrogramas guardados como imagen,
sino **GAF generado como array NumPy directamente desde la ventana raw**. Esto evita
perdida de precision por rasterizacion y mantiene la transformacion unida a la serie
temporal original.

```python
from src.preprocesamiento.gaf import generar_gaf

ventana = df_raw["raw_eeg"].values[:1024]
imagen_gaf = generar_gaf(ventana, tamano=128, metodo="summation")
```

---

## Fase 3 - Exportacion a CSV Orientada a ML

### Arquitectura

```
src/exportacion/
    |-- grabador_csv.py     Clase GrabadorCSV (grabador incremental thread-safe)

datos/sesiones/             Carpeta destino de los archivos generados
    |-- sesion_YYYY-MM-DD_HH-MM-SS_raw.csv
    |-- sesion_YYYY-MM-DD_HH-MM-SS_bandas.csv
```

**Flujo de grabacion:**
```
hilo_lectura_serial / hilo_demo
     |
     v
LectorVisualizacion.on_datos  (o hilo_demo directamente)
     |
     +-- grabador.agregar_raw(valor, timestamp, poor_signal)
     +-- grabador.agregar_bandas(datos, timestamp)
              |
              v  (Queue thread-safe — solo epoch + valores numericos,
                  sin formatear hora para no bloquear el parser a 512 Hz)
         hilo_escritura  (formatea hora, consume cola, escribe en bloques de 100)
              |
              +-- sesion_..._raw.csv
              +-- sesion_..._bandas.csv
```

### Para que sirve cada archivo CSV

**`sesion_..._raw.csv` — raw EEG a 512 Hz:**

Contiene la senal electrica cruda del cerebro, 512 muestras por segundo.
Es la entrada principal para ambas ramas modernas del pipeline:

1. En fase 4 se divide en ventanas (ej. 2 seg = 1024 samples)
2. La ventana se usa como serie temporal raw normalizada para modelos secuenciales
3. La misma ventana se transforma a GAF para la CNN 2D

Sin este archivo no es posible construir ni las secuencias raw ni los GAF: las bandas
pre-calculadas del TGAM ya promediaron esa informacion y no son reversibles.

**`sesion_..._bandas.csv` — bandas + eSense a ~1 Hz:**

Contiene los 8 valores de potencia espectral que el TGAM calcula internamente,
mas Attention y Meditation. Un vector de 10 valores por segundo.
Actualmente se conserva como **apoyo de inspeccion y trazabilidad**, no como entrada
principal del entrenamiento cruzado con OpenNeuro.

Este archivo es ~512 veces mas pequeno que el raw: 5 min de sesion = ~300 filas
vs ~150,000 filas en raw. Sigue siendo util para revisar rapidamente la sesion y
comparar tendencias generales del TGAM.

### Modulo grabador_csv.py

#### Clase GrabadorCSV

```python
from src.exportacion.grabador_csv import GrabadorCSV

grabador = GrabadorCSV(carpeta="datos/sesiones", duracion_seg=120)
grabador.iniciar(puerto_com="COM5")

# En el callback del parser:
grabador.agregar_raw(datos.raw_wave, timestamp, datos.poor_signal)
grabador.agregar_bandas(datos, timestamp)

# Para detener (o se detiene automaticamente al cumplirse la duracion):
ruta_raw, ruta_bandas = grabador.detener()
```

**Constructor:**
- `carpeta`: ruta destino (se crea si no existe, default: `datos/sesiones`)
- `duracion_seg`: duracion maxima en segundos, `None` para grabacion manual

**Metodos:**

| Metodo | Descripcion |
|--------|-------------|
| `iniciar(puerto_com, version_sw)` | Crea archivos, escribe headers, arranca hilo de escritura |
| `detener()` | Detiene grabacion, vacia cola, retorna `(ruta_raw, ruta_bandas)` |
| `agregar_raw(valor, timestamp, poor_signal)` | Encola muestra raw (ignora si ps >= 50) |
| `agregar_bandas(datos, timestamp)` | Encola paquete de bandas (ignora si datos.poor_signal >= 50) |

**Propiedades:**

| Propiedad | Descripcion |
|-----------|-------------|
| `grabando` | `True` mientras la sesion esta activa |
| `muestras_raw` | Numero de filas escritas en el CSV raw |
| `muestras_bandas` | Numero de filas escritas en el CSV de bandas |
| `tiempo_transcurrido` | Segundos desde el inicio de la sesion |
| `progreso` | Fraccion [0.0, 1.0] de la duracion configurada |
| `ruta_raw` / `ruta_bandas` | Rutas de los archivos generados |

**Constantes configurables en el modulo:**
- `UMBRAL_SENAL = 50` — poor_signal maximo para grabar una muestra
- `BLOQUE_FLUSH = 100` — filas acumuladas antes de forzar flush a disco
- `VERSION_SW = "3.0"` — version que se escribe en los headers

#### Formato de los archivos generados

**Header del CSV raw:**
```
# EEG Raw - NeuroSky TGAM
# fecha=2026-04-03 19:45:00
# puerto_com=COM5
# fs_raw=512
# version_sw=3.0
# umbral_senal=50
timestamp_epoch,timestamp_hora,raw_eeg,poor_signal
1775267400.001,19:45:00.001,-145,0
1775267400.003,19:45:00.003,23,0
...
```

**Header del CSV de bandas:**
```
# EEG Bandas + eSense - NeuroSky TGAM
# fecha=2026-04-03 19:45:00
# puerto_com=COM5
# fs_bandas=~1Hz
# version_sw=3.0
# umbral_senal=50
timestamp_epoch,timestamp_hora,delta,theta,low_alpha,high_alpha,low_beta,high_beta,low_gamma,high_gamma,attention,meditation,poor_signal
1775267401.000,19:45:01.000,395460,5168,4593,2159,1217,1630,278,120,44,83,0
...
```

### Como usar los controles de grabacion en la interfaz

1. Ejecutar la aplicacion: `python visualizar_eeg.py` (o `--demo`)
2. Seleccionar el **Modo** de grabacion:
     - `entrenamiento_sano`
     - `entrenamiento_parkinson`
     - `paciente_evaluacion`
2. Seleccionar la duracion en el desplegable **Duracion** (derecha de la barra)
3. En modo de entrenamiento, verificar/ajustar la **Etiqueta** (`sano` o `parkinson`)
4. Presionar **Grabar** — el boton cambia a rojo **Detener**
5. La barra de progreso roja avanza con el tiempo; el label **REC MM:SS** muestra
   el tiempo transcurrido
6. La grabacion termina automaticamente al cumplirse la duracion, o manualmente
   al presionar **Detener**
7. Aparece un dialogo con rutas, modo, etiqueta y conteo de muestras

**Notas:**
- Si la calidad de senal es mala durante toda la sesion, los archivos se generan pero
  quedan casi vacios (el filtro poor_signal descarta las muestras)
- El selector de duracion se deshabilita mientras hay una sesion activa
- El selector de modo/etiqueta tambien se bloquea durante la sesion activa
- La opcion **personalizado** abre un dialogo para ingresar segundos (10 a 7200)
- La opcion **sin limite** graba hasta que se presione Detener manualmente

### Carpeta de salida por modo de grabacion

- `entrenamiento_sano` y `entrenamiento_parkinson` guardan en `datos/sesiones/`
- `paciente_evaluacion` guarda en `datos/evaluacion_pacientes/`

Esto evita mezclar sesiones clinicas de evaluacion con datasets de entrenamiento.

### Etiquetado automatico para preprocesamiento

Cada sesion grabada registra su etiqueta en:

- `datos/sesiones/etiquetas_locales.csv` (entrenamiento)
- `datos/evaluacion_pacientes/etiquetas_locales.csv` (evaluacion)

Adicionalmente, el header de cada CSV incluye:

- `# etiqueta_local=<sano|parkinson|evaluacion>`
- `# uso_sesion=<entrenamiento|evaluacion>`

### Notas de implementacion

**Por que el formateo de hora ocurre en el hilo de escritura y no en el productor:**

`agregar_raw()` se llama 512 veces por segundo desde el hilo del parser serial.
Calcular `datetime.fromtimestamp()` en ese contexto crearia 512 objetos Python
descartables por segundo, agregando trabajo al path critico donde cada microsegundo
cuenta para no perder muestras. El hilo de escritura corre a ~20 ciclos/seg y tiene
tiempo de sobra para formatear al momento de escribir cada fila.

**Por que `ventana.grabador` se cachea al inicio del callback:**

La propiedad `VentanaEEG.grabador` evalua una condicion en cada acceso. En `on_datos`
y en `hilo_demo` se podria necesitar tanto para raw como para bandas en la misma
invocacion, lo que implicaria evaluarla dos veces. Se cachea en variable local al
inicio del callback para evaluacion unica por paquete.

### Troubleshooting

| Problema | Causa | Solucion |
|----------|-------|----------|
| Los archivos se generan vacios o con muy pocas filas | `poor_signal` alto durante la sesion | Verificar el electrodo, esperar a que `Signal` baje de 50 en la barra de estado |
| No aparece el dialogo de confirmacion al terminar | La duracion se cumplio antes de que el timer de Qt lo detectara | Se detecta en el siguiente tick del timer (33ms), esperar un momento |
| `RuntimeError: Ya hay una sesion activa` | Se llamo a `iniciar()` dos veces | Llamar `detener()` antes de iniciar una nueva sesion |
| Los archivos no aparecen en `datos/sesiones/` | La carpeta no existe y fallo la creacion | Verificar permisos de escritura en el directorio del proyecto |

---

## Fase 4 - Preprocesamiento de Datos EEG

### Arquitectura

```
src/preprocesamiento/
    |-- __init__.py           Paquete del modulo
    |-- cargador_csv.py       Carga CSVs propios (sesiones TGAM)
    |-- dataset_publico.py    Descarga/adaptacion OpenNeuro ds002778
    |-- normalizacion.py      MinMax [0,1] y Z-score (media=0, std=1)
    |-- ventanas.py           Ventanas deslizantes 1D y 2D con overlap
    |-- features.py           Features estadisticos por ventana
    |-- gaf.py                Serie temporal -> Gramian Angular Field
    |-- espectrogramas.py     STFT -> espectrogramas normalizados (legacy)
    |-- etiquetado.py         Etiquetado y division train/val/test por sujeto
    |-- pipeline.py           Script principal del pipeline completo

datos/procesados/              Salida del pipeline
    |-- rama_a_v{X}s_o{Y}_{norm}/   Series temporales
    |   |-- datos.npz                X_train, y_train, ..., X_test, y_test
    |   |-- normalizador.npz         Parametros de normalizacion
    |   |-- metadata.json            Configuracion del pipeline
    |-- rama_b_gaf_v{X}s_o{Y}/      Imagenes GAF
    |-- rama_b_stft_v{X}s_o{Y}/     Espectrogramas STFT (legacy)
        |-- datos.npz                Espectrogramas como arrays NumPy
        |-- metadata.json            Config STFT, shape, division

datos/openneuro_procesado/     Dataset publico procesado
    |-- sub-{id}_ses-{sesion}.npz   Grabaciones individuales
    |-- indice.json                  Inventario con etiquetas
```

**Flujo de datos del pipeline:**
```
CSVs propios (fase 3)     Dataset OpenNeuro ds002778
sesion_*_raw.csv          sub-{id}_ses-{sesion}.npz
         \                       /
          v                     v
         raw EEG (float32, 512 Hz)
                  |
                  v
         Ventanas deslizantes (1s, 2s, 5s, 10s × overlap 50%, 75%)
              /                    \
             v                      v
    RAMA A                        RAMA B
    Serie temporal raw            Imagen GAF
    (normalizada)                 (desde la misma ventana raw)
             |                      |
             v                      v
    Normalizacion                 GAF determinista por ventana
    (MinMax o Z-score)
             |                      |
             v                      v
    Division train/val/test por sujeto (GroupShuffleSplit)
             |                      |
             v                      v
    datos.npz + normalizador.npz    datos.npz + metadata.json
```

### Como ejecutar el pipeline

**Requisitos previos:**
```bash
pip install scipy pandas scikit-learn
# Para dataset publico (opcional):
pip install mne mne-bids openneuro-py
```

**Ejecucion:**
```bash
cd 07_Proyecto_Python

# Pipeline basico con datos locales
python -m src.preprocesamiento.pipeline --fuente local --ventana 2 --overlap 0.5

# Solo Rama A con Z-score
python -m src.preprocesamiento.pipeline --fuente local --rama a --norm zscore

# Rama principal recomendada: raw temporal + GAF
python -m src.preprocesamiento.pipeline --fuente publico --rama ambas --repr-temporal raw --repr-imagen gaf --gaf-size 128 --ventana 2 --overlap 0.5

# Baseline legacy con STFT
python -m src.preprocesamiento.pipeline --fuente publico --rama b --repr-imagen stft --ventana 2

# Con dataset publico (requiere descarga previa)
python -m src.preprocesamiento.pipeline --fuente publico --ventana 2 --overlap 0.5

# Ambas fuentes combinadas
python -m src.preprocesamiento.pipeline --fuente ambos --ventana 2 --overlap 0.5

# Etiqueta global para sesiones locales (si no se usa CSV por sesion)
python -m src.preprocesamiento.pipeline --fuente local --etiqueta-local parkinson --ventana 2 --overlap 0.5

# Etiquetado por sesion desde CSV (recomendado cuando hay mezcla sano/parkinson)
python -m src.preprocesamiento.pipeline --fuente local --etiquetas-local-csv datos/sesiones/etiquetas_locales.csv --ventana 2 --overlap 0.5 --rama ambas --norm ambos
```

**Argumentos del pipeline:**

| Argumento | Valores | Default | Descripcion |
|-----------|---------|---------|-------------|
| `--fuente` | local, publico, ambos | local | Fuente de datos |
| `--etiqueta-local` | sano, parkinson | sano | Etiqueta global para sesiones locales |
| `--etiquetas-local-csv` | ruta CSV | (vacio) | Mapa por sesion `sello,etiqueta` (prioridad sobre `--etiqueta-local`) |
| `--ventana` | float (segundos) | 2.0 | Duracion de ventana |
| `--overlap` | float [0, 1) | 0.5 | Fraccion de solapamiento |
| `--norm` | minmax, zscore, ambos | ambos | Normalizacion (Rama A) |
| `--repr-temporal` | raw, features | raw | Repr de la serie temporal |
| `--repr-imagen` | gaf, stft | gaf | Repr de imagen |
| `--gaf-size` | int | 128 | Tamano del GAF |
| `--rama` | a, b, ambas | ambas | Rama a ejecutar |
| `--sesiones` | ruta | datos/sesiones | Carpeta de CSVs locales |
| `--dataset` | ruta | datos/openneuro_procesado | Carpeta del dataset publico |
| `--salida` | ruta | datos/procesados | Carpeta de salida |

### Descargar y procesar el dataset publico

```bash
# 1. Instalar dependencias
pip install mne mne-bids openneuro-py

# 2. Desde Python:
from src.preprocesamiento.dataset_publico import descargar_dataset, procesar_dataset

# Descargar (puede tardar varios minutos)
descargar_dataset(destino="datos/openneuro_raw")

# Procesar: extraer Fp1, remuestrear a 512 Hz, guardar como .npz
procesar_dataset(
    bids_root="datos/openneuro_raw",
    destino="datos/openneuro_procesado",
    solo_fp1=False,  # True para ahorrar espacio (~80% menos)
)

# Opcion recomendada para simular TGAM 1-canal y ahorrar espacio:
procesar_dataset(
     bids_root="datos/openneuro_raw",
     destino="datos/openneuro_procesado",
     solo_fp1=True,
)
```

**Dataset: OpenNeuro ds002778 (UC San Diego)**
- 15 pacientes Parkinson + 16 controles sanos
- 32 canales EEG (BioSemi ActiveTwo), formato BDF/BIDS
- 2 sesiones por paciente PD: on/off medicacion dopaminergica
- Se extrae canal Fp1 (posicion frontal, equivalente al TGAM)
- Se remuestrea de la frecuencia original a 512 Hz

### Modulos Python detallados

#### cargador_csv.py

| Funcion | Entrada | Salida |
|---------|---------|--------|
| `cargar_raw_csv(ruta)` | Ruta a `*_raw.csv` | DataFrame (timestamp_epoch, raw_eeg, poor_signal) |
| `cargar_bandas_csv(ruta)` | Ruta a `*_bandas.csv` | DataFrame con 8 bandas + eSense |
| `cargar_sesion(carpeta, sello)` | Carpeta + sello temporal | dict con 'raw' y 'bandas' DataFrames |
| `listar_sesiones(carpeta)` | Carpeta de sesiones | Lista de sellos temporales |
| `cargar_todas_sesiones(carpeta)` | Carpeta de sesiones | Lista de dicts con raw + bandas |
| `raw_a_numpy(df)` | DataFrame raw | Array 1D float32 |
| `bandas_a_numpy(df)` | DataFrame bandas | Array 2D (n, 8) float32 |

#### normalizacion.py

Dos clases con interfaz identica:

```python
# Crear normalizador
from src.preprocesamiento.normalizacion import crear_normalizador
norm = crear_normalizador("minmax")  # o "zscore"

# Ajustar SOLO con datos de entrenamiento
norm.ajustar(X_train)

# Transformar todos los conjuntos con los mismos parametros
X_train = norm.transformar(X_train)
X_val = norm.transformar(X_val)
X_test = norm.transformar(X_test)

# Guardar parametros para produccion
params = norm.parametros()  # dict con min/max o media/std
```

**MinMax:** Escala a [0, 1]. Features constantes (rango=0) → 0.
**Z-score:** Media=0, std=1. Features constantes (std=0) → 0.

#### ventanas.py

```python
from src.preprocesamiento.ventanas import crear_ventanas, crear_ventanas_2d

# Senal 1D (raw EEG)
ventanas = crear_ventanas(raw, fs=512, ventana_seg=2.0, overlap=0.5)
# Shape: (n_ventanas, 1024)

# Senal multicanal (8 bandas)
ventanas = crear_ventanas_2d(bandas, fs=1, ventana_seg=10.0, overlap=0.5)
# Shape: (n_ventanas, 10, 8)
```

#### features.py

```python
from src.preprocesamiento.features import extraer_features_lote

# Para ventanas 1D: (n_ventanas, muestras) -> (n_ventanas, 4)
features = extraer_features_lote(ventanas_1d)

# Para ventanas multicanal: (n_ventanas, muestras, 8) -> (n_ventanas, 32)
features = extraer_features_lote(ventanas_3d)
```

Features por canal: media, varianza, skewness, kurtosis.

#### gaf.py

```python
from src.preprocesamiento.gaf import generar_gaf_lote

# Ventanas de raw EEG -> imagenes GAF
imagenes = generar_gaf_lote(
    ventanas,      # (n_ventanas, 1024)
    tamano=128,
    metodo="summation",
)
# Shape: (n_ventanas, 128, 128)
```

Parametros recomendados: `tamano=128`, `metodo="summation"`.

#### etiquetado.py

```python
from src.preprocesamiento.etiquetado import dividir_por_sujeto

division = dividir_por_sujeto(X, y, sujetos)
# division["X_train"], division["y_train"], division["sujetos_train"]
# division["X_val"], division["y_val"], division["sujetos_val"]
# division["X_test"], division["y_test"], division["sujetos_test"]
# division["info"] -> conteos, balance, sujetos por conjunto
```

Division: 70% train, 15% val, 15% test. Agrupado por sujeto (sin data leakage).

### Cargar datos procesados para entrenamiento (Fase 5)

```python
import numpy as np

# Rama A
datos = np.load("datos/procesados/rama_a_v2.0s_o50_minmax/datos.npz", allow_pickle=True)
X_train, y_train = datos["X_train"], datos["y_train"]
X_val, y_val = datos["X_val"], datos["y_val"]
X_test, y_test = datos["X_test"], datos["y_test"]

# Rama B
datos = np.load("datos/procesados/rama_b_gaf_v2.0s_o50/datos.npz", allow_pickle=True)
X_train = datos["X_train"]  # shape: (n, 128, 128)
# Agregar dimension de canal para CNN: (n, 128, 128, 1)
X_train = X_train[..., np.newaxis]
```

### Troubleshooting

| Problema | Causa | Solucion |
|----------|-------|----------|
| `ModuleNotFoundError: scipy` | Dependencias no instaladas | `pip install scipy pandas scikit-learn` |
| `senal muy corta (0.0s < 2.0s)` | Sesion de grabacion demasiado corta | Usar ventana mas pequena o grabar sesiones mas largas |
| `No se cargaron datos` | Ruta de sesiones incorrecta | Verificar `--sesiones datos/sesiones` |
| `Columna 'X' no encontrada` | CSV con formato diferente al esperado | Verificar que el CSV fue generado por GrabadorCSV de la fase 3 |
| Division desbalanceada (todo sano) | Solo datos locales sin dataset publico | Descargar y procesar dataset OpenNeuro |
| `ImportError: mne` | MNE no instalado | `pip install mne mne-bids openneuro-py` |
| Error al descargar OpenNeuro | Sin internet o timeout | Reintentar; descargar sujetos individuales en vez de todo |
| `Fp1 not found` en dataset publico | Nombre de canal diferente | Verificar nombres con `raw.ch_names` en MNE |

### GUI de preprocesamiento

Se agrego una interfaz grafica para evitar errores de comandos manuales.

**Lanzamiento:**
```bash
python preprocesar_datos_gui.py
```

**Funciones incluidas:**
- Seleccion de fuente (`local`, `publico`, `ambos`)
- Selector de etiqueta local (`sano`, `parkinson`); se deshabilita automaticamente cuando se carga un CSV de etiquetas para evitar confusion sobre cual tiene prioridad
- CSV opcional de etiquetas por sesion (formato `sello,etiqueta`, tiene prioridad sobre etiqueta global)
- Presets rapidos (incluye `Reprocesado oficial v1` con carpeta de salida timestamp)
- Validacion previa de rutas e `indice.json` antes de ejecutar
- Ejecucion del pipeline con logs en vivo
- Botones **Limpiar logs** y **Ver historial** (abre carpeta del JSON en el explorador)
- Copiar comando generado para reproducibilidad
- Historial JSON de ejecuciones en `resultados/historial_preprocesamiento.json`
- Controles deshabilitados durante ejecucion para prevenir cambios accidentales

**Nota:** La inferencia sobre una sesion individual (`evaluar_paciente.py`) se perdio y
esta pendiente de reescribirse en la Fase 6, ahora sobre el modelo Keras de la Rama B.

---

## Fase 5 - Entrenamiento de la CNN (Rama B)

### Como ejecutar

```bash
# 1. Generar ventanas GAF con division por persona (balanceada)
python -m src.preprocesamiento.pipeline --fuente publico --rama b

# 2. Entrenar
python -m src.modelos.entrenar_cnn_rama_b datos/procesados/rama_b_gaf_v2.0s_o50_balanceado/datos.npz
```

Argumentos de `entrenar_cnn_rama_b.py`: `--epochs` (50), `--batch-size` (64), `--salida`.
Cada corrida crea una carpeta nueva con marca de tiempo (`..._vAAAAMMDD_HHMMSS`) para no
sobrescribir resultados.

### Que hace el script

1. Carga `X_train/X_val/X_test` (GAF 128x128) y les agrega el canal.
2. CNN: 3 bloques `Conv2D -> BatchNorm -> MaxPool` (32, 64, 128 filtros),
   `GlobalAveragePooling2D`, `Dense(64)` con L2, `Dropout(0.35)`, salida sigmoide.
3. Adam (lr 1e-4) con `BinaryFocalCrossentropy`.
4. `EarlyStopping`, `ModelCheckpoint` y `ReduceLROnPlateau` monitoreando `val_pr_auc`.
5. Busca en validacion el umbral que maximiza la balanced accuracy.
6. Evalua en test con ese umbral y guarda:
   `mejor_modelo.keras`, `modelo_final.keras`, `curvas_aprendizaje.png`,
   `matriz_confusion.png`, `resumen_metricas.json`.

### Reglas para no romper la evaluacion

- **Dividir siempre por persona.** En ds002778 cada paciente con Parkinson tiene dos sesiones
  (`on`/`off`). `etiquetado.id_paciente()` convierte `pd9_off` en `pd9` para agruparlas.
- En sesiones TGAM propias, llenar la columna `sujeto` de `etiquetas_locales.csv` con un ID
  por persona.
- No usar solo accuracy: el test puede estar desbalanceado. Revisar ROC AUC, PR AUC y
  balanced accuracy.

### Usar el modelo

```python
import numpy as np
import tensorflow as tf

modelo = tf.keras.models.load_model("mejor_modelo.keras", compile=False)
prob = modelo.predict(gaf[..., np.newaxis]).ravel()   # gaf: (n, 128, 128)
prediccion = prob >= 0.66                             # umbral calibrado de la v6
```

La entrada debe pasar exactamente por el mismo preprocesamiento: raw a 512 Hz, ventanas de 2 s
(1024 muestras) y GAF `summation` de 128x128.

### Troubleshooting

| Problema | Causa / solucion |
|---|---|
| TensorFlow no usa la GPU | En Windows nativo TensorFlow >= 2.11 no soporta GPU. Usar WSL2 o aceptar CPU (~75 s por epoca). |
| Memoria insuficiente al cargar `datos.npz` | Cada variante GAF ocupa ~450 MB; cerrar otros programas o reducir `--gaf-size`. |
| Accuracy alta pero ROC AUC ~0.5 | El modelo predice siempre una clase; revisar el balance del test. |

