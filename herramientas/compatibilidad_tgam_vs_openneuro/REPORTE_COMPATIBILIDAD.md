# Reporte de compatibilidad TGAM vs OpenNeuro

## Decision sugerida

Se recomienda usar **raw EEG de 1 canal** como señal base común entre TGAM y OpenNeuro.
Las sesiones locales compatibles pueden servir para validación externa del dominio TGAM.
Las bandas del TGAM deben quedarse como apoyo/inspección, no como eje principal de entrenamiento cruzado.

## Resumen público

- Grabaciones públicas analizadas: 46
- Mediana/rango útil de amplitud: std q10-q90 = 57.05 a 208.56
- Mediana/rango útil de amplitud robusta: p95 q10-q90 = 99.89 a 355.99
- Controles sanos: 16 | Parkinson: 30

## Sesiones locales

### 2026-04-06_04-36-37 - COMPATIBLE

- std = 85.29
- p95_abs = 179.00
- poor_signal_mean = 0.00
- poor_signal_max = 0
- pct_abs_ge_1000 = 0.0000
- std_en_rango = True
- p95_en_rango = True
- senal_estable = True

Vecinos públicos más cercanos:
- sub-pd11_ses-on | PD | distancia=2.7894
- sub-pd6_ses-on | PD | distancia=3.3484
- sub-pd5_ses-on | PD | distancia=3.5045
- sub-hc10_ses-hc | HC | distancia=4.1547
- sub-pd12_ses-off | PD | distancia=4.5139

### 2026-04-20_12-27-18 - REVISAR / DESCARTAR

- std = 719.01
- p95_abs = 1548.00
- poor_signal_mean = 19.57
- poor_signal_max = 26
- pct_abs_ge_1000 = 0.1582
- std_en_rango = False
- p95_en_rango = False
- senal_estable = False

Vecinos públicos más cercanos:
- sub-pd12_ses-off | PD | distancia=12.7770
- sub-pd9_ses-off | PD | distancia=13.3570
- sub-hc25_ses-hc | HC | distancia=13.6910
- sub-pd11_ses-off | PD | distancia=13.9054
- sub-pd5_ses-off | PD | distancia=14.3554

### 2026-04-21_16-46-26 - COMPATIBLE

- std = 107.87
- p95_abs = 237.00
- poor_signal_mean = 0.00
- poor_signal_max = 0
- pct_abs_ge_1000 = 0.0000
- std_en_rango = True
- p95_en_rango = True
- senal_estable = True

Vecinos públicos más cercanos:
- sub-pd6_ses-on | PD | distancia=2.6686
- sub-pd11_ses-on | PD | distancia=4.7472
- sub-hc10_ses-hc | HC | distancia=4.7538
- sub-hc2_ses-hc | HC | distancia=5.0612
- sub-pd12_ses-off | PD | distancia=5.1868

## Conclusión

- Compatibles recomendadas para validación TGAM: 2026-04-06_04-36-37, 2026-04-21_16-46-26
- Sesiones a revisar o excluir del análisis principal: 2026-04-20_12-27-18
- El dataset público sí es utilizable para preentrenamiento o entrenamiento base en raw EEG monocanal.
- La compatibilidad encontrada justifica migrar la rama de imagen hacia GAF derivado del raw, manteniendo el entrenamiento cruzado sobre una señal común.
