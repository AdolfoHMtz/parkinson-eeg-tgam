# Flujo Operativo

Guia corta para reproducir el flujo completo del proyecto desde otra maquina.

## Scripts esenciales

Para operar el proyecto en el dia a dia solo se necesitan estos scripts en la raiz:

- `leer_tgam.py`
- `visualizar_eeg.py`
- `preprocesar_datos_gui.py`
- `inspeccionar_npz_gui.py`

Para aprender a usar el inspector:
- `guias/GUIA_INSPECTOR_NPZ.md`

Herramientas auxiliares:
- `herramientas/comparar_tgam_vs_openneuro.py`
- `herramientas/inspeccionar_npz.py`
- `herramientas/ver_series_gaf.py`

## 1. Captura con TGAM

### Paso 1: identificar el puerto

```bash
python leer_tgam.py
```

Que hace:
- si solo hay un puerto util, lo toma automaticamente
- si hay varios, muestra la lista para elegir

Objetivo:
- confirmar que el TGAM se ve en la maquina
- identificar el `COM` correcto

### Paso 2: abrir el visualizador

```bash
python visualizar_eeg.py COM5
```

Tambien se puede usar autodeteccion:

```bash
python visualizar_eeg.py
```

### Paso 3: grabar sesiones

Dentro del visualizador:
- elegir `Duracion`
- elegir `Etiqueta`: `sano` o `parkinson`
- presionar `Grabar`

Cada sesion genera **2 archivos CSV**:
- `sesion_YYYY-MM-DD_HH-MM-SS_raw.csv`
- `sesion_YYYY-MM-DD_HH-MM-SS_bandas.csv`

Ademas, registra la etiqueta en:
- `datos/sesiones/etiquetas_locales.csv`

## 2. Que archivo importa para entrenar

El archivo principal del proyecto es:
- `raw.csv`

Se usa para:
- series temporales raw
- GAF para CNN

El archivo de:
- `bandas.csv`

se conserva para:
- inspeccion
- trazabilidad
- comparacion visual con lo que calcula el TGAM

## 3. Preprocesamiento recomendado

Abrir la GUI:

```bash
python preprocesar_datos_gui.py
```

Configuracion recomendada:
- `Fuente`: `publico` para entrenamiento base
- `Fuente`: `ambos` si quieres mezclar OpenNeuro + sesiones locales
- `Repr temporal`: `raw`
- `Repr imagen`: `gaf`
- `Tamano GAF`: `128`
- `Ventana`: `2.0`
- `Overlap`: `0.5`
- `Normalizacion`: `minmax`
- `Rama`: `ambas`

Si vas a usar sesiones locales mezcladas:
- seleccionar `datos/sesiones/etiquetas_locales.csv`

## 4. Salida del preprocesamiento

El pipeline no modifica las sesiones originales.

Genera carpetas nuevas como:

```text
datos/procesados/
  rama_a_raw_v2.0s_o50_minmax/
  rama_b_gaf_v2.0s_o50/
```

Cada carpeta contiene:
- `datos.npz`
- `metadata.json`
- `normalizador.npz` en Rama A

## 5. Pipeline por terminal

### Dataset publico solamente

```bash
python -m src.preprocesamiento.pipeline --fuente publico --rama ambas --repr-temporal raw --repr-imagen gaf --gaf-size 128 --ventana 2 --overlap 0.5 --norm minmax
```

### Mezcla de OpenNeuro + sesiones locales

```bash
python -m src.preprocesamiento.pipeline --fuente ambos --rama ambas --repr-temporal raw --repr-imagen gaf --gaf-size 128 --ventana 2 --overlap 0.5 --norm minmax --etiquetas-local-csv datos/sesiones/etiquetas_locales.csv
```

## 6. Procesar OpenNeuro desde cero

Si se descarga otro dataset o se reprocesa OpenNeuro, la receta base es:

1. elegir canal frontal equivalente (`Fp1` recomendado)
2. convertir a monocanal
3. remuestrear a `512 Hz`
4. pasar a microvoltios si hace falta
5. remover offset DC
6. guardar como `.npz`

## 7. Reglas utiles

- no borrar `datos/sesiones/`
- no mezclar etiquetas a mano dentro de los CSV
- si mueves sesiones entre computadoras, copia tambien `etiquetas_locales.csv`
- si no existe `etiquetas_locales.csv`, puedes usar etiqueta global en el GUI
- si no existe `bandas.csv` pero si existe `raw.csv`, el preprocesamiento local actual sigue funcionando
- la validacion TGAM vs OpenNeuro es recomendable, pero no es parte del flujo diario de captura
