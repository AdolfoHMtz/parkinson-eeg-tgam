"""
Modulo de preprocesamiento de datos EEG para deteccion de Parkinson.

Contiene:
  - cargador_csv: Lectura de CSVs propios (sesiones TGAM)
  - dataset_publico: Descarga y adaptacion del dataset OpenNeuro ds002778
  - normalizacion: MinMax y Z-score
  - ventanas: Ventanas deslizantes con overlap
  - features: Extraccion de features estadisticos por ventana (ruta legacy/opcional)
  - gaf: Generacion de Gramian Angular Fields desde raw EEG
  - espectrogramas: Generacion de STFT del raw EEG (ruta legacy/opcional)
  - pipeline: Script principal que ejecuta todo el preprocesamiento
  - gui_pipeline: Interfaz grafica para configurar/ejecutar el pipeline
"""
