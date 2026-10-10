# Guias del Proyecto

Esta carpeta concentra la documentacion operativa y de consulta rapida del proyecto.

## Orden recomendado de lectura

1. `FLUJO_OPERATIVO.md`
2. `PLAN_DE_IMPLEMENTACION.md`
3. `GUIA_NPZ.md`
4. `GUIA_INSPECTOR_NPZ.md`
5. `COMPARADOR_TGAM_OPENNEURO.md`
6. `RESUMEN_PROYECTO_PARA_EXPONER.md`

## Scripts principales

Los scripts que se usan en el flujo diario quedaron en la raiz de `07_Proyecto_Python`:

- `leer_tgam.py`
- `visualizar_eeg.py`
- `preprocesar_datos_gui.py`
- `inspeccionar_npz_gui.py`

## Utilidades auxiliares

Las herramientas de apoyo quedaron en `herramientas/`:

- `comparar_tgam_vs_openneuro.py`
- `inspeccionar_npz.py`
- `ver_series_gaf.py`

La inferencia sobre una sesion individual (`evaluar_paciente.py`) esta pendiente
de reescribirse (ver Fase 6 en `PLAN_DE_IMPLEMENTACION.md`).
