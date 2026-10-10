# Reporte de Fase 1 - Lectura del Dispositivo en Python
## Servicio Social: Analisis de Redes Neuronales para Deteccion Temprana de Parkinson

**Estado:** COMPLETADA

---

## 1. Objetivo de la Fase

Establecer la comunicacion con el dispositivo NeuroSky TGAM via puerto serial desde Python,
implementando un parser del protocolo binario que extraiga los mismos datos que muestra
la aplicacion MindView (proporcionada por el profesor): senal cruda EEG, 8 bandas de
frecuencia, atencion, meditacion, calidad de senal y fuerza de parpadeo.

**Resultado:** Objetivo cumplido al 100%. El programa Python lee y decodifica todos
los datos del dispositivo en tiempo real, validados contra la salida de Realterm y MindView.

---

## 2. Actividades Realizadas

### 2.1 Organizacion del proyecto
- Se organizaron todos los archivos recibidos del profesor en 6 carpetas tematicas
  (Documentacion, Drivers, Codigo Fuente, SDKs, Tests, Hardware)
- Se creo un inventario completo documentando cada archivo y su proposito
- Se creo la carpeta `07_Proyecto_Python/` con la estructura modular del proyecto

### 2.2 Analisis del codigo de referencia
Se revisaron 4 fuentes de referencia para entender el protocolo:
- `Parser.cs` (C#) - Parser binario del profesor
- `COMPortManager.cs` (C#) - Configuracion serial del profesor
- `ThinkGearStreamParser.c` (C) - Parser oficial de NeuroSky
- `thinkgear.h` (C++) - Header con constantes y API oficial

### 2.3 Implementacion en Python
- **serial_manager.py** - Modulo de conexion serial con autodeteccion de puerto COM
- **tgam_parser.py** - Parser del protocolo binario TGAM (maquina de estados)
- **leer_tgam.py** - Programa principal de lectura con visualizacion en consola

### 2.4 Pruebas con dispositivo fisico
- Se conecto el dispositivo TGAM via adaptador BT-USB en COM5
- Se verifico la lectura de datos raw EEG a 512 Hz
- Se verificaron las 8 bandas de frecuencia, atencion y meditacion
- Se confirmo calidad de senal 0/200 (excelente) con contacto correcto

### 2.5 Revision de codigo
Se ejecuto una revision triple (calidad, eficiencia, reuso) y se corrigieron hallazgos.

---

## 3. Hallazgos y Observaciones

### 3.1 Problemas encontrados en el codigo original del profesor (C#)

El parser C# del profesor (`Parser.cs`) presentaba las siguientes deficiencias:

| Problema | Detalle |
|----------|---------|
| **Codigo 0x04 mal nombrado** | Se llama `PARSER_CODE_CONFIGURATION` pero en realidad es el codigo de **Attention** segun la especificacion oficial del protocolo TGAM |
| **Meditation ausente** | No se implementa el codigo 0x05 (Meditation). El parser original no lee este dato |
| **Bandas EEG ignoradas** | El codigo `PARSER_CODE_EEG_POWER` (0x83) tiene un case que solo hace `i += valueBytesLength` sin parsear los 24 bytes de las 8 bandas de frecuencia |
| **Blink Strength ausente** | No se implementa el codigo 0x16 para detectar parpadeos |
| **Logica confusa en code 0x04** | Hay un filtro de "signal values" (29, 54, 55, 56, 80, 81, 82, 107, 200) donde imprime "NoShouldAtt" que no tiene documentacion y no corresponde al protocolo oficial |
| **extendedCodeLevel incrementado incorrectamente** | Se incrementa en cada iteracion del while exterior en lugar de solo cuando se encuentran bytes EXCODE (0x55) |

Estas deficiencias se descubrieron al contrastar el codigo C# con:
- El parser oficial en C (`ThinkGearStreamParser.c`) que usa callbacks genericos
- El header `thinkgear.h` que define los codigos correctos (TG_DATA_ATTENTION=2, TG_DATA_MEDITATION=3)
- La especificacion del protocolo en `mindset_communications_protocol.pdf`

### 3.2 Observaciones del dispositivo

- El dispositivo transmite datos raw EEG a 512 Hz (512 muestras por segundo)
- Las 8 bandas de frecuencia y los valores eSense (atencion/meditacion) llegan aprox 1 vez por segundo
- Cuando no hay contacto con la piel, `poor_signal = 200` y los raw oscilan entre -2048 y 2047
- Con buen contacto, `poor_signal = 0` y los datos se estabilizan
- El puerto COM se autodetecta correctamente como "Dispositivo serie USB (COM5)"
- La consola de Windows (cp1252) no soporta caracteres Unicode como barras de progreso

### 3.3 Datos de prueba obtenidos

Con el dispositivo en contacto correcto:
```
Calidad de senal:    0/200  (EXCELENTE)
Attention:           44/100
Meditation:          83/100
Raw EEG:            -135
Delta:           395,460
Theta:             5,168
Low Alpha:         4,593
High Alpha:        2,159
Low Beta:          1,217
High Beta:         1,630
Low Gamma:           278
High Gamma:          120
```

---

## 4. Decisiones Tecnicas y Justificacion

### 4.1 Conexion directa al COM en lugar de ThinkGear_Connector

**Decision:** Conectar directamente al puerto COM con `pyserial` en vez de usar el servicio
ThinkGear_Connector (TCP puerto 13854).

**Justificacion:**
- Elimina dependencia de software propietario de NeuroSky
- Control total sobre el parsing de datos sin intermediarios
- Menor latencia al leer directamente del puerto serial
- Mas portable (no requiere instalar servicios adicionales)

### 4.2 Autodeteccion de puerto COM en vez de hardcodear COM5

**Decision:** Usar `serial.tools.list_ports` para detectar automaticamente el puerto.

**Justificacion:**
- El puerto COM puede cambiar entre maquinas (COM3, COM5, COM8, etc.)
- Si solo hay un puerto disponible, se usa automaticamente sin preguntar
- Si hay multiples puertos, se muestra una lista para que el usuario elija
- Hace el programa portable a cualquier maquina sin modificar codigo

### 4.3 Reimplementacion completa del parser en vez de port directo del C#

**Decision:** No hacer un port literal del `Parser.cs`, sino reimplementar basandose en el
parser C oficial y el header C++ como fuentes autoritativas.

**Justificacion:**
- El parser C# del profesor tenia errores de nombrado y datos faltantes (ver seccion 3.1)
- El parser C oficial (`ThinkGearStreamParser.c`) tiene la implementacion correcta
- El header `thinkgear.h` define los codigos oficiales del protocolo
- Se implementaron los 6 codigos de datos (el C# solo tenia 3 funcionales)

### 4.4 Uso de `int.from_bytes()` en vez de operaciones de bits manuales

**Decision:** Usar `int.from_bytes(data, byteorder="big", signed=True)` para la
conversion de bytes a enteros.

**Justificacion:**
- Es la forma idiomatica de Python para estas conversiones
- Mas legible que `(high << 8) | low; if raw > 32767: raw -= 65536`
- Maneja automaticamente el signo (signed=True para raw wave)

### 4.5 Callback con copia del objeto en vez de referencia compartida

**Decision:** El parser envia `copy(self.datos)` al callback en vez de `self.datos` directo.

**Justificacion:**
- Si el consumidor guarda la referencia (ej. para CSV o buffer de datos), todas las
  referencias apuntarian al mismo objeto mutable que se sobreescribe con cada paquete
- Enviar una copia previene este bug de correctness que seria dificil de diagnosticar
- El costo de `copy()` es insignificante (dataclass con 13 campos int)

### 4.6 Clase LectorTGAM en vez de variables globales

**Decision:** Encapsular el estado de visualizacion en una clase `LectorTGAM` en vez de
usar variables globales `_raw_count` y `_mostrar_solo_raw`.

**Justificacion:**
- Variables globales mutables dificultan el testing y reutilizacion
- La clase permite tener multiples instancias si fuera necesario
- Mejor separacion de responsabilidades

---

## 5. Problemas Encontrados y Soluciones

| Problema | Causa | Solucion |
|----------|-------|----------|
| `pip install` fallaba por timeout | Conexion a internet inestable | Reintentar con `--timeout 60` |
| Error `UnicodeEncodeError` al imprimir barras | Consola Windows usa cp1252 que no soporta caracteres Unicode (`█`, `░`) | Reemplazar con caracteres ASCII (`#`, `-`) |
| Datos raw oscilan -2048/2047 sin contacto | Comportamiento normal del TGAM sin contacto con la piel | Documentado; `poor_signal=200` indica sin contacto |
| `bandas_dict()` se llamaba a 512 Hz innecesariamente | El callback se invoca por cada paquete (raw llega 512 veces/seg) | Verificar `datos.delta > 0` directamente antes de crear dict |

---

## 6. Archivos Creados

| Archivo | Descripcion |
|---------|-------------|
| `07_Proyecto_Python/leer_tgam.py` | Programa principal de lectura del TGAM |
| `07_Proyecto_Python/src/conexion/serial_manager.py` | Autodeteccion y conexion al puerto COM |
| `07_Proyecto_Python/src/parser/tgam_parser.py` | Parser del protocolo binario TGAM |
| `07_Proyecto_Python/src/__init__.py` | Init del paquete src |
| `07_Proyecto_Python/src/conexion/__init__.py` | Init del paquete conexion |
| `07_Proyecto_Python/src/parser/__init__.py` | Init del paquete parser |
| `07_Proyecto_Python/requirements.txt` | Dependencias: pyserial>=3.5 |
| `07_Proyecto_Python/PLAN_DE_IMPLEMENTACION.md` | Plan detallado de las 6 fases |
| `01_Documentacion/INVENTARIO_PROYECTO.md` | Inventario completo del proyecto |
| `CLAUDE.md` | Contexto del proyecto para sesiones futuras |

## 7. Dependencias Instaladas

| Paquete | Version | Proposito |
|---------|---------|-----------|
| `pyserial` | 3.5 | Comunicacion serial con el puerto COM |

---

## 8. Siguiente Fase

**Fase 2 - Visualizacion en tiempo real:** Implementar graficas con `matplotlib` que
repliquen las dos graficas de MindView (onda cruda + bandas de frecuencia, y
atencion/meditacion/calidad de senal).
