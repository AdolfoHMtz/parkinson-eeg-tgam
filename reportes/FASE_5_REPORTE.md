# Reporte de Fase 5 - Entrenamiento y Evaluación de Redes Neuronales (Rama B)

## Servicio Social: Análisis de Redes Neuronales para Detección Temprana de Parkinson

**Estado:** COMPLETADO

---

## 1. Objetivo de la Fase

Implementar, entrenar y evaluar el modelo de Clasificación de Imágenes basado en las representaciones GAF (Gramian Angular Fields) generadas en la Fase 4 (Rama B). El objetivo es que la red neuronal aprenda a extraer patrones invisibles en las ondas EEG en 2D para diagnosticar el Parkinson con un alto porcentaje de exactitud.

**Resultado Actual:** El modelo de referencia es la **v6** (seccion 8): misma CNN que la v5, pero entrenada y evaluada con una division **por persona** que corrige una fuga de datos detectada en las versiones v1-v5. En test (5 personas nunca vistas) obtuvo ROC AUC 0.899 y balanced accuracy 0.831, y ordeno correctamente a las 9 personas de validacion y test. Con tan pocas personas en test, el resultado se considera prometedor pero no concluyente.

---

## 2. Desarrollo del Modelo CNN (versión 1.0)

Se diseñó un script de entrenamiento `entrenar_cnn_rama_b.py` utilizando **TensorFlow y Keras**.

### 2.1 Arquitectura de la Red CNN (Inspiración VGG)

Se optó por una red convolucional robusta y regularizada para combatir el ruido típico en estudios EEG:

- **Entrada:** Imagen GAF (128x128x1).
- **Extracción de Características:** 3 Bloques Convolucionales.
  - Conv2D -> BatchNormalization -> MaxPooling2D (Filtros: 32, 64, 128)
- **Clasificador (Capas Densas):**
  - Flatten() a vector 1D (25,088 parámetros).
  - Capa Densa (128 neuronas) + Dropout al 50%.
  - Salida Sigmoidal (1 neurona para clasificación binaria: Sano=0, Parkinson=1).

### 2.2 Hiperparámetros (v1.0)

- **Learning Rate:** 0.001 (Adam)
- **Batch Size:** 32
- **Épocas:** 50
- **Regularización Adicional:** Early Stopping (patience=10) y Model Checkpoint.

### 2.3 Resultados del Primer Entrenamiento (v1.0)

Al alimentar la red con las más de 6,000 imágenes de entrenamiento generadas en la Fase 4, los resultados fueron increíblemente positivos, pero mostraron síntomas clásicos de memoria temprana (overfitting).

- **Época de mejor rendimiento:** 1 (La red absorbió el conocimiento críticamente rápido)
- **Activación Early Stopping:** Época 11. El Validation Loss comenzó a dispararse indicando pérdida de generalización.
- **Exactitud en Test Final (Muestras independientes):** **86.13%**
- **Pérdida en Test Final:** 0.4526

Superar el umbral del 85% en las primeras ejecuciones en ingeniería biomédica pura es un resultado sobresaliente para una arquitectura base, sin embargo, el acierto en la Época 1 indicó un factor de "suerte estadística" más que un aprendizaje profundo real.

---

## 3. Mejoras Implementadas y Resultados (versión 2.0)

Dado que la red v1.0 tuvo problemas de sobreajuste rápido, se aplicaron ajustes defensivos directamente en el código fuente para obligar a la CNN a aprender más lentamente.

### 3.1 Hiperparámetros ajustados (v2.0):

1. **Ratio de Aprendizaje (Learning Rate):** Se redujo a `0.0001`.
2. **Tamaño del Lote (Batch Size):** Se aumentó a `64`.
3. **Dropout Espacial (Doble capa):** Se agregó un nuevo `Dropout(0.25)` después del último bloque de `MaxPooling2D`.

### 3.2 Resultados de la versión 2.0 (Caída por sobre-regularización)

Al aplicar medidas sumamente defensivas (doble dropout y un learning rate sumamente bajo), la red sufrió un síndrome severo de _Subajuste_. La red avanzó a tientas y aunque internamente creía mejorar, en los datos de validación no logró pasar del 50%.

El error de diseño fue fijar el Early Stopping en la estabilización del `val_loss`. El modelo rescató la Época 1 arrojando una **Exactitud real en test de 21.60%**.

---

## 4. Mejoras Implementadas para v3.0 (El Punto de Equilibrio)

Para solucionar los extremos de las versiones anteriores, se recalibró la red buscando un balance:

1. **Corrección de la Métrica de Guardado (Crucial):**
   - Ahora la red no se fija en la "pérdida matemática" (`val_loss`), sino puramente en su % de aciertos `val_accuracy` (Mode MAX), y la paciencia del Early Stopping subió de 10 a 15 épocas.
2. **Learning Rate (Punto Intermedio):**
   - Establecido en `0.0005`. Un ritmo ni tan rápido que memorice (v1), ni tan lento que no aprenda (v2).
3. **Capacidad Densa (Densidad Reducida):**
   - La capa final se redujo de 128 a `64` neuronas. Menos neuronas reducen la capacidad bruta de "memorizar" a la fuerza.
4. **Restauración del Dropout:**
   - Se removió el dropout doble que entorpeció el aprendizaje en la v2, y el Dropout final bajó ligeramente a `0.4` (40%).

---

## 5. Resultados Definitivos (versión 3.0)

Tras ejecutar la versión 3.0 con el versionamiento por tiempo activado, el modelo logró un entrenamiento estable y orgánico. El Early Stopping detuvo la red en la Época 20, pero rescató exitosamente los pesos óptimos descubiertos en la **Época 5**.

**Métricas Finales de Evaluación (Conjunto de Test):**

- **Exactitud (Accuracy):** 77.89% (0.7789)
- **Pérdida (Loss):** 0.5523

### 5.1 Analisis Critico: ¿Por que la v3.0 es mejor base que la v1.0?

Aunque la v1.0 tuvo un accuracy mas alto en test, la v3.0 es mejor como modelo de trabajo porque su comportamiento es mas **estable, controlado y defendible**:

1. **Mejor criterio de seleccion del modelo:** La v1.0 llego rapido a un pico fuerte en la primera epoca, pero despues mostro sintomas claros de sobreajuste. En cambio, la v3.0 trabajo con `val_accuracy` como referencia de guardado y con `restore_best_weights`, asi que no solo busco "bajar la perdida", sino conservar el punto que realmente generalizaba mejor.
2. **Balance entre aprendizaje y regularizacion:** La v3.0 usa un `learning rate` intermedio, menos neuronas densas y dropout moderado. Eso reduce la probabilidad de memorizar ruido y obliga a la red a aprender patrones mas generales de las imagenes GAF.
3. **Resultado mas interpretable para uso clinico:** Aunque el 77.89% de test no supera numericamente al 86.13% de la v1.0, la v3.0 ofrece una base mas seria para continuar iterando porque su entrenamiento fue mas consistente y su mejor epoca se pudo justificar con mayor claridad.

### 5.2 Artefactos Generados

El modelo final ha sido empaquetado y asegurado para su implementación en la aplicación diagnóstica.

- **Directorio:** `datos/modelos_entrenados/cnn_rama_b_v20260618_181718`
- **Archivos clave guardados:**
  - `mejor_modelo.keras` (Pesos limpios de la Época 5).
  - `modelo_final.keras` (Pesos de cierre de la Época 20).
  - `curvas_aprendizaje.png` (Evidencia gráfica de la estabilidad lograda).

---

## 6. Version Candidata v4.0 - Mejora Final Evaluada

Con base en el comportamiento de la v3.0 y en la sospecha de que la exactitud estaba siendo engañosa por el desbalance de clases, se implementó una nueva corrida de entrenamiento con ajustes más orientados a clasificación médica real:

1. **Función de pérdida enfocada en casos difíciles:** Se sustituyó `binary_crossentropy` por `BinaryFocalCrossentropy` para obligar al modelo a prestar más atención a errores difíciles y no quedarse con la clase dominante.
2. **Métricas más adecuadas para desbalance:** Se añadió `roc_auc`, `pr_auc`, `precision` y `recall` para dejar de depender solo de `accuracy`.
3. **Selección de umbral basada en validación:** Se buscó el umbral óptimo en validación para reportar resultados más honestos que el corte fijo de `0.5`.
4. **Diagnóstico del desbalance real:** El split final mostró una distribución muy sesgada en test, con **188** muestras de clase 0 y **1196** de clase 1, por lo que la exactitud por sí sola dejó de ser una métrica confiable.

### 6.1 Resultados de la v4 candidata

La corrida produjo un comportamiento estable en entrenamiento, pero el modelo terminó colapsando hacia una predicción casi constante de la clase positiva en test.

- **Accuracy en test:** 86.42%
- **ROC AUC en test:** 0.4999
- **PR AUC en test:** 0.8622
- **Balanced Accuracy con umbral calibrado:** 0.5000
- **Matriz de confusión:** `[[0, 188], [0, 1196]]`

### 6.2 Interpretación

La v4 candidata no debe presentarse como una mejora real del clasificador, porque el accuracy alto quedó explicado por el sesgo del conjunto de prueba. Lo valioso de esta corrida es que dejó claro que el siguiente salto no depende tanto de “hacer la red más fuerte”, sino de corregir el particionado o el balance de datos para que el modelo aprenda a distinguir ambas clases y no solo a acertar la clase mayoritaria.

---

## 7. Version Final v5.0 - Split Balanceado por Sujeto

A partir del diagnostico anterior, se genero una nueva particion balanceada por sujeto para la Rama B. El objetivo fue conservar la independencia por sujeto, pero evitando que el conjunto de prueba quedara dominado por una sola clase.

### 7.1 Ajuste aplicado

Se construyo un nuevo `datos.npz` con una distribucion mas pareja por clase:

- **Train:** clase 0 = 2092, clase 1 = 4140
- **Val:** clase 0 = 389, clase 1 = 856
- **Test:** clase 0 = 568, clase 1 = 973

Esto corrigio el sesgo extremo de la version anterior, donde el test tenia 188 muestras de clase 0 frente a 1196 de clase 1.

### 7.2 Resultados de la v5.0

Con el mismo modelo mejorado y sobre el split balanceado, la corrida que mejor equilibro separacion, precision y sensibilidad fue la guardada en `cnn_rama_b_v20260623_203449`. Esa es la version que se toma como final defendible para entrega:

- **Accuracy en test:** 74.17%
- **Loss en test:** 0.1426
- **ROC AUC en test:** 0.8144
- **PR AUC en test:** 0.8898
- **Precision en test:** 0.7762
- **Recall en test:** 0.8304
- **Umbral seleccionado en validacion:** `0.55`
- **Accuracy con umbral calibrado:** 71.71%
- **Balanced Accuracy con umbral calibrado:** 0.7532
- **F1 con umbral calibrado:** 0.7332
- **Matriz de confusion:** `[[506, 62], [374, 599]]`

### 7.3 Interpretacion final

Esta corrida representa la mejor combinacion entre separacion global y comportamiento por umbral. El `ROC AUC` de `0.8144` y el `PR AUC` de `0.8898` muestran una capacidad de discriminacion mas solida que las versiones anteriores, y la `Balanced Accuracy` calibrada de `0.7532` indica que el modelo ya no depende de una sola clase.

Ademas, el umbral de `0.55` produjo una matriz de confusion util y clinicamente mas interpretable que las corridas previas. Por eso, para fines de entrega, esta es la version mas defendible: conserva buen rendimiento global, mantiene un equilibrio razonable entre falsos positivos y falsos negativos, y ya no presenta el colapso de umbral observado en iteraciones anteriores.

### 7.4 Artefactos finales

- **Directorio:** `datos/modelos_entrenados/cnn_rama_b_v20260623_203449`
- **Archivos clave:** `mejor_modelo.keras`, `modelo_final.keras`, `curvas_aprendizaje.png`, `matriz_confusion.png`, `resumen_metricas.json`

### 7.5 Cierre de la v5

> **Nota:** esta conclusion quedo superada por la seccion 8. Las versiones v1-v5 se evaluaron con una division que repetia personas entre train y test.

En su momento, la version `203449` se adopto como el modelo final de la Fase 5 porque ofrece el mejor equilibrio entre capacidad discriminativa, interpretabilidad clinica y estabilidad de la decision binaria. Las corridas posteriores fueron utiles como verificacion metodologica, pero no superaron el desempeño integral de esta version final. Por ello, este resultado queda como la base mas solida para continuar hacia la siguiente fase del proyecto.

---

## 8. Version v6 - Correccion de Fuga de Datos por Paciente

### 8.1 Problema detectado

En OpenNeuro ds002778 cada paciente con Parkinson tiene **dos grabaciones**: con medicacion (`on`) y sin medicacion (`off`). El pipeline identificaba a cada ventana con `sujeto_sesion` (por ejemplo `pd9_on` y `pd9_off`), asi que trataba las dos grabaciones de una misma persona como si fueran personas distintas. Por eso hablaba de 46 "sujetos" cuando en realidad son **31 personas**.

Consecuencia en el split balanceado de la v5:

- Los **5 pacientes con Parkinson del test** (pd9, pd14, pd19, pd26, pd28) tambien estaban en train con su otra sesion.
- Los 4 pacientes con Parkinson de validacion tambien estaban en train.
- Los controles sanos (una sola sesion) si eran personas nuevas.

Es decir, el modelo podia "reconocer" a personas con Parkinson que ya habia visto, pero no a los controles. Las metricas de v1-v5 no miden generalizacion a personas nuevas.

### 8.2 Correccion

- `pipeline.py`: el identificador de grupo pasa a ser solo el sujeto (`pd9`), no `sujeto_sesion`.
- `etiquetado.py`: funcion `id_paciente()` para normalizar IDs y division con `GroupShuffleSplit` agrupada por persona.
- `rebalancear_splits_rama_b.py`: agrupa por persona al reconstruir el split balanceado.
- Sesiones locales TGAM: nueva columna `sujeto` en `etiquetas_locales.csv`; si falta, todas las sesiones se asumen de la misma persona (`local_01`).

Nuevo split (`rama_b_gaf_v2.0s_o50_por_paciente`):

| Conjunto | Personas | Ventanas sano | Ventanas Parkinson |
|---|---|---|---|
| Train | 22 | 2092 | 4331 |
| Val | 4 (hc10, hc33, pd6, pd16) | 389 | 851 |
| Test | 5 (hc4, hc8, hc29, pd3, pd9) | 568 | 787 |

Ninguna persona aparece en dos conjuntos.

### 8.3 Entrenamiento

Mismo script (`entrenar_cnn_rama_b.py`), misma arquitectura e hiperparametros que la v5. El early stopping se detuvo en la epoca 13 y restauro la epoca 3 (mejor `val_pr_auc` = 0.803).

### 8.4 Resultados en test (5 personas no vistas)

| Metrica | v5 (con fuga) | **v6 (por persona)** |
|---|---|---|
| ROC AUC (por ventana) | 0.814 | **0.899** |
| PR AUC | 0.890 | **0.924** |
| Umbral calibrado en val | 0.55 | 0.66 |
| Accuracy calibrada | 0.717 | **0.827** |
| Balanced accuracy calibrada | 0.753 | **0.831** |
| Precision | 0.906 | 0.887 |
| Recall | 0.616 | **0.806** |
| F1 | 0.733 | **0.844** |
| Matriz de confusion | [[506, 62], [374, 599]] | [[487, 81], [153, 634]] |

Probabilidad media por persona:

| Conjunto | Persona | Real | Prob. media | % ventanas >= umbral |
|---|---|---|---|---|
| Val | hc10 | Sano | 0.651 | 22 % |
| Val | hc33 | Sano | 0.651 | 33 % |
| Val | pd16 | Parkinson | 0.659 | 46 % |
| Val | pd6 | Parkinson | 0.661 | 49 % |
| Test | hc4 | Sano | 0.635 | 4 % |
| Test | hc8 | Sano | 0.639 | 10 % |
| Test | hc29 | Sano | 0.652 | 27 % |
| Test | pd3 | Parkinson | 0.677 | 76 % |
| Test | pd9 | Parkinson | 0.678 | 85 % |

### 8.5 Interpretacion

1. **Corregir la fuga no bajo las metricas.** Lo que mas mueve el resultado es *que personas* caen en test: con 5-8 personas en test, cambiar a una sola puede mover el AUC varias decimas. Por eso v5 y v6 no se pueden comparar como "mejor" o "peor"; la v6 es simplemente la unica evaluacion valida.
2. **Hay senal, pero el margen es pequeno.** Todas las probabilidades estan entre 0.63 y 0.68 y la mejor epoca fue la 3. Aun asi, en las 9 personas no vistas (val + test) todos los pacientes con Parkinson quedaron por encima de todos los controles.
3. **No es concluyente.** Se necesitan mas personas (o repetir el experimento con distintas particiones) antes de afirmar que el modelo generaliza. Tampoco se ha validado con grabaciones TGAM de pacientes con Parkinson.

### 8.6 Artefactos

- **Directorio:** `datos/modelos_entrenados/cnn_rama_b_por_paciente_v20261009_200737`
- **Archivos:** `mejor_modelo.keras`, `modelo_final.keras`, `curvas_aprendizaje.png`, `matriz_confusion.png`, `resumen_metricas.json`
- Las carpetas `cnn_rama_b_v2026*` (v1-v5) se conservan como historial; fueron evaluadas con la division con fuga.
