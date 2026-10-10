# Guia NPZ

Esta guia explica que son los archivos `.npz`, que guardan en este proyecto y como inspeccionarlos.

## 1. Que es un `.npz`

Un `.npz` es un contenedor comprimido de NumPy.

Piensalo como un `.zip` que guarda varios arrays con nombre.

En este proyecto se usa para guardar:

- datasets preprocesados listos para entrenar
- normalizadores
- grabaciones publicas procesadas

## 2. Tipos de `.npz` del proyecto

### A. Grabaciones publicas procesadas

Ruta:

```text
datos/openneuro_procesado/sub-...npz
```

Claves tipicas:

- `raw_fp1`
- `etiqueta`
- `sujeto`
- `sesion`
- `fs`

### B. Salida de preprocesamiento Rama A

Ruta:

```text
datos/procesados/rama_a_.../datos.npz
```

Claves tipicas:

- `X_train`
- `y_train`
- `sujetos_train`
- `X_val`
- `y_val`
- `sujetos_val`
- `X_test`
- `y_test`
- `sujetos_test`

Si `repr-temporal=raw`:

- cada muestra suele tener forma `(1024,)` para ventana de 2 segundos a 512 Hz

### C. Salida de preprocesamiento Rama B

Ruta:

```text
datos/procesados/rama_b_.../datos.npz
```

Claves tipicas:

- `X_train`
- `y_train`
- `sujetos_train`
- `X_val`
- `y_val`
- `sujetos_val`
- `X_test`
- `y_test`
- `sujetos_test`

Si `repr-imagen=gaf`:

- cada muestra suele tener forma `(128, 128)`

### D. Normalizadores

Ruta:

```text
datos/procesados/rama_a_.../normalizador.npz
```

Claves tipicas:

- para MinMax: `tipo`, `min`, `max`
- para Z-score: `tipo`, `media`, `std`

## 3. Como abrir un `.npz`

```python
import numpy as np

datos = np.load("datos/procesados/rama_a_raw_v2.0s_o50_minmax/datos.npz", allow_pickle=True)
print(datos.files)
```

## 4. Como interpretar su contenido

- `X_train`, `X_val`, `X_test`: entradas del modelo
- `y_train`, `y_val`, `y_test`: etiquetas, donde `0 = sano` y `1 = parkinson`
- `sujetos_*`: identificador del sujeto origen de cada muestra
- `metadata.json`: configuracion usada para generar ese dataset

## 5. Como visualizar una muestra

### Opcion recomendada: GUI

```bash
python inspeccionar_npz_gui.py
```

La GUI permite:

- cargar un `.npz`
- ver sus claves y shapes
- mostrar serie temporal
- mostrar GAF
- revisar indices y splits

Guia detallada de la interfaz:

- `guias/GUIA_INSPECTOR_NPZ.md`

### Opcion por terminal

```bash
python herramientas/inspeccionar_npz.py datos/openneuro_procesado/sub-hc1_ses-hc.npz
```

### Opcion de visualizacion avanzada

```bash
python herramientas/ver_series_gaf.py datos/openneuro_procesado/sub-hc1_ses-hc.npz --inicio-sec 10 --duracion-sec 2
```

## 6. Ejemplos rapidos con Python

### Serie temporal raw

```python
import numpy as np
import matplotlib.pyplot as plt

datos = np.load("datos/procesados/rama_a_raw_v2.0s_o50_minmax/datos.npz", allow_pickle=True)
x = datos["X_train"][0]

plt.plot(x)
plt.title("Ventana raw normalizada")
plt.show()
```

### GAF

```python
import numpy as np
import matplotlib.pyplot as plt

datos = np.load("datos/procesados/rama_b_gaf_v2.0s_o50/datos.npz", allow_pickle=True)
img = datos["X_train"][0]

plt.imshow(img, cmap="RdBu_r", aspect="auto")
plt.colorbar()
plt.title("GAF")
plt.show()
```

## 7. Resumen mental

- `.npz` = varios arrays guardados juntos
- `datos.npz` = dataset listo para entrenar
- `normalizador.npz` = parametros de normalizacion
- `sub-...npz` = una grabacion publica procesada
- `metadata.json` = explicacion de la configuracion usada
