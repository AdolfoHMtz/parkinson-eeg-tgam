# Comparador TGAM vs OpenNeuro

Esta guia explica como usar el comparador que revisa si las sesiones locales TGAM se parecen lo suficiente al dataset publico OpenNeuro a nivel de `raw` monocanal.

## 1. Para que sirve

El comparador ayuda a responder:

- si una sesion local TGAM es estable
- si una sesion local cae dentro del rango de amplitud razonable del dataset publico
- que sesiones conviene usar para validar el dominio TGAM
- que sesiones conviene descartar por ruido o valores atipicos

## 2. Script

El script es:

```bash
python herramientas/comparar_tgam_vs_openneuro.py
```

## 3. Uso basico

Desde la raiz de `07_Proyecto_Python`:

```bash
python herramientas/comparar_tgam_vs_openneuro.py
```

Esto usa por defecto:
- `datos/sesiones`
- `datos/openneuro_procesado`
- salida en `herramientas/compatibilidad_tgam_vs_openneuro`

## 4. Uso con rutas explicitas

```bash
python herramientas/comparar_tgam_vs_openneuro.py --sesiones datos/sesiones --dataset datos/openneuro_procesado --salida herramientas/compatibilidad_tgam_vs_openneuro
```

## 5. Que genera

### A. Resumen JSON

```text
herramientas/compatibilidad_tgam_vs_openneuro/resumen_compatibilidad.json
```

Contiene:
- resumen por sesion local
- resumen global del dataset publico
- resumen de sanos y Parkinson
- vecinos publicos mas cercanos por sesion local

### B. Tabla CSV

```text
herramientas/compatibilidad_tgam_vs_openneuro/sesiones_locales_resumen.csv
```

Sirve para ver rapido:
- `std`
- `p95_abs`
- `poor_signal_mean`
- `poor_signal_max`
- `pct_abs_ge_1000`
- `compatible`

### C. Reporte Markdown

```text
herramientas/compatibilidad_tgam_vs_openneuro/REPORTE_COMPATIBILIDAD.md
```

Este es el archivo mas util para el reporte del proyecto.

## 6. Como interpreta el comparador

Para cada sesion local revisa tres cosas:

- si `std` cae dentro del rango util del dataset publico
- si `p95_abs` cae dentro del rango util del dataset publico
- si la señal se ve estable:
  - `poor_signal_max < 25`
  - `pct_abs_ge_1000 < 0.05`

Si esas tres condiciones se cumplen, la sesion se marca como:
- `COMPATIBLE`

Si no, se marca como:
- `REVISAR / DESCARTAR`

## 7. Como leer el resultado

### Compatible

Conviene usarla para:
- validacion externa del dominio TGAM
- comparar luego contra el modelo entrenado con OpenNeuro

### Revisar / descartar

Conviene:
- no usarla en el analisis principal
- revisarla como sesion ruidosa o con contacto inestable

## 8. Nota metodologica

El comparador no dice que TGAM y OpenNeuro sean “identicos”.

Lo que si dice es si:
- el `raw` monocanal del TGAM se parece lo suficiente al `Fp1` procesado del dataset publico

Eso es lo que justifica:
- entrenar con el dataset publico
- validar despues con sesiones reales TGAM

## 9. Flujo recomendado

1. procesar OpenNeuro
2. grabar sesiones locales TGAM
3. correr el comparador
4. elegir sesiones locales compatibles
5. preprocesar dataset para entrenamiento
6. entrenar la red
