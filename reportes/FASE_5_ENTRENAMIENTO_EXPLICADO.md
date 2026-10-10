# Fase 5 - Explicacion Tecnica del Entrenamiento

## Proyecto: Deteccion Temprana de Parkinson con EEG

**Modelo final de referencia:** `cnn_rama_b_por_paciente_v20261009_200737` (v6)  
**Artefacto principal generado:** `modelo_final.keras`

---

## 1. Que se hizo en esta fase

En esta fase entrenamos un modelo de inteligencia artificial para que aprendiera a distinguir entre dos tipos de imagenes generadas a partir de señales EEG:

- `0` = Sano
- `1` = Parkinson

La idea no fue solo sacar un porcentaje de acierto. Lo importante era construir un modelo que pudiera aprender patrones utiles en las imagenes y que ademas fuera suficientemente estable y defendible para un contexto medico.

En palabras sencillas: tomamos una señal biologica, la convertimos en una imagen, y luego entrenamos a una red neuronal para reconocer si esa imagen se parece mas a un caso sano o a un caso con Parkinson.

---

## 2. Lenguaje, librerias y herramientas

### 2.1 Lenguaje principal

- **Python**

### 2.2 Librerias principales

- **TensorFlow / Keras**: para construir y entrenar la red neuronal.
- **NumPy**: para cargar y manipular matrices de datos.
- **Matplotlib**: para generar graficas del entrenamiento y de la matriz de confusion.
- **scikit-learn**: para calcular metricas como `accuracy`, `precision`, `recall`, `F1`, `ROC AUC`, `PR AUC` y `balanced_accuracy`.

### 2.3 Entorno de trabajo

- **Windows**
- **Python 3.13**
- Estructura modular dentro de `07_Proyecto_Python/src`

---

## 3. Como funciona el entrenamiento

El script `src/modelos/entrenar_cnn_rama_b.py` sigue esta ruta:

1. Carga el archivo `datos.npz`.
2. Prepara las imagenes para que Keras las entienda con la forma correcta.
3. Construye la red neuronal convolucional.
4. La entrena con los datos de entrenamiento.
5. Revisa el comportamiento con validacion.
6. Ajusta el umbral de decision.
7. Evalua el resultado final en test.
8. Guarda resultados, graficas y el modelo final.

Eso permite que el entrenamiento no solo produzca un modelo, sino tambien evidencia clara de como se comporto durante el aprendizaje.

---

## 4. Datos utilizados

La version final trabajo con el dataset dividido **por persona**:

- `datos/procesados/rama_b_gaf_v2.0s_o50_por_paciente/datos.npz`

Distribucion:

- **Train:** 22 personas | clase 0 = 2092, clase 1 = 4331 ventanas
- **Val:** 4 personas | clase 0 = 389, clase 1 = 851 ventanas
- **Test:** 5 personas | clase 0 = 568, clase 1 = 787 ventanas

Dividir por persona es importante porque cada paciente con Parkinson tiene dos grabaciones
(con y sin medicacion). Si se divide por grabacion, la misma persona puede quedar en train y en
test, y el modelo aprende a reconocer a la persona en lugar de la enfermedad. Las versiones
v1-v5 tenian ese problema (ver `FASE_5_REPORTE.md`, seccion 8).

## 5. Arquitectura del modelo

La red usada fue una CNN compacta con tres bloques convolucionales:

- `Conv2D(32) -> BatchNormalization -> MaxPooling2D`
- `Conv2D(64) -> BatchNormalization -> MaxPooling2D`
- `Conv2D(128) -> BatchNormalization -> MaxPooling2D`
- `GlobalAveragePooling2D`
- `Dense(64, relu)`
- `Dropout(0.35)`
- `Dense(1, sigmoid)`

### 5.1 Que significa cada parte

#### `Conv2D`

Es la parte que “mira” la imagen buscando patrones. En lugar de leer toda la imagen de golpe, va localizando formas, bordes y combinaciones de pixeles que puedan ser importantes.

#### `BatchNormalization`

Sirve para estabilizar el aprendizaje. En palabras simples, ayuda a que la red no se descontrole tanto durante el entrenamiento.

#### `MaxPooling2D`

Reduce el tamaño de la informacion y se queda con lo mas importante. Eso hace que la red trabaje con un resumen de la imagen y no se distraiga con detalles innecesarios.

#### `GlobalAveragePooling2D`

Convierte toda la informacion aprendida en un vector mas pequeño. Esto ayuda a que el modelo tenga menos parametros y no memorice de mas.

#### `Dense(64, relu)`

Es la parte que toma lo aprendido por los bloques convolucionales y lo combina para tomar una decision.

#### `Dropout(0.35)`

Durante el entrenamiento apaga parte de las neuronas de forma aleatoria. Esto obliga a la red a no depender siempre de los mismos caminos y reduce el sobreajuste.

#### `Dense(1, sigmoid)`

Es la salida final. Da una probabilidad entre 0 y 1:

- cerca de 0, la red piensa en `Sano`
- cerca de 1, la red piensa en `Parkinson`

### 5.2 Por que usar tres bloques

Se eligieron tres bloques porque era suficiente para extraer patrones sin volver el modelo demasiado pesado.

Eso importa por tres razones:

- Las imagenes GAF ya concentran bastante informacion.
- Una red enorme podria memorizar ruido en lugar de aprender patrones reales.
- Una arquitectura intermedia es mas facil de entrenar, revisar y justificar.

En otras palabras: no se busco un modelo exageradamente complejo, sino uno equilibrado.

---

## 6. Parametros de entrenamiento

La corrida final se entreno con:

- **Optimizer:** `Adam`
- **Learning rate:** `0.0001`
- **Loss:** `BinaryFocalCrossentropy(gamma=2.0, alpha=0.25)`
- **Batch size:** `64`
- **Epochs maximas:** `50`

### 6.1 Que significa cada parametro

#### Optimizer `Adam`

Es el mecanismo que va corrigiendo los errores de la red paso a paso. Se puede imaginar como un piloto automatico que ajusta los pesos de la red para que aprenda mejor.

#### Learning rate `0.0001`

Es el tamaño de cada ajuste. Si es demasiado grande, el entrenamiento puede volverse brusco; si es demasiado pequeño, puede aprender muy lento. Aqui se eligio un valor bajo para que el aprendizaje fuera mas estable.

#### Loss `BinaryFocalCrossentropy`

Es la funcion que mide el error. Se uso esta version porque le presta mas atencion a los casos dificiles y evita que el modelo se quede “comodamente” aprendiendo solo la clase mas facil.

#### Batch size `64`

Significa que la red procesa 64 ejemplos a la vez antes de actualizarse. Eso hace que el aprendizaje sea mas ordenado que hacerlo ejemplo por ejemplo.

#### Epochs maximas `50`

Es el limite de vueltas de entrenamiento. No quiere decir que siempre se usen todas, porque el sistema puede parar antes si ya no mejora.

### 6.2 Por que se cambio respecto a modelos anteriores

Los cambios principales se hicieron para mejorar la generalizacion:

- el `learning rate` bajo ayudó a aprender de forma mas cuidadosa;
- la `BinaryFocalCrossentropy` ayudó a enfocarse en errores importantes;
- `Dropout` y `BatchNormalization` ayudaron a reducir el sobreajuste;
- `EarlyStopping` y `ModelCheckpoint` evitaron quedarse con una epoca mala.

---

## 7. Como se evaluo el modelo

No se confio solo en `accuracy`, porque en este proyecto esa metrica por si sola podia ser engañosa.

Se usaron estas metricas:

- `accuracy`
- `precision`
- `recall`
- `F1`
- `ROC AUC`
- `PR AUC`
- `balanced_accuracy`

### 7.1 Por que no bastaba con accuracy

Si una clase domina o el conjunto de prueba queda desbalanceado, la red puede parecer buena aunque en realidad este acertando siempre la misma clase.

Por eso se revisaron otras metricas:

- `ROC AUC`: muestra que tan bien separa el modelo ambas clases.
- `PR AUC`: es muy util cuando las clases no estan perfectamente equilibradas.
- `recall`: dice cuantos casos positivos logra detectar.
- `balanced_accuracy`: evita que la clase mayoritaria esconda errores en la otra clase.

---

## 8. Version final defendida

La corrida que se toma como resultado final es la v6:

- **Carpeta:** `datos/modelos_entrenados/cnn_rama_b_por_paciente_v20261009_200737`

### 8.1 Resultados principales (test: 5 personas no vistas)

- **ROC AUC:** 0.899
- **PR AUC:** 0.924
- **Umbral seleccionado en validacion:** `0.66`
- **Accuracy calibrada:** 82.73%
- **Balanced Accuracy calibrada:** 0.831
- **Precision:** 0.887
- **Recall:** 0.806
- **F1:** 0.844

### 8.2 Interpretacion

El modelo separa ambas clases en personas que nunca vio: en validacion y test (9 personas),
todos los pacientes con Parkinson recibieron una probabilidad media mayor que todos los controles.

Pero hay que leerlo con cuidado:

- el test tiene solo 5 personas, asi que el resultado depende mucho de quienes caen ahi;
- las probabilidades son muy parecidas entre si (0.63 a 0.68), el margen es pequeno;
- todavia no se ha probado con grabaciones TGAM de pacientes con Parkinson.

Es una senal prometedora, no una prueba de que el modelo funciona clinicamente.

## 9. Graficas generadas y su significado

### 9.1 `curvas_aprendizaje.png`

Esta grafica muestra dos cosas:

- **Exactitud por epoca:** entrenamiento vs validacion.
- **Perdida por epoca:** entrenamiento vs validacion.

#### Como leerla

- Si entrenamiento y validacion suben juntos, el modelo esta aprendiendo bien.
- Si entrenamiento mejora pero validacion empeora, hay sobreajuste.
- Si validacion sube y baja demasiado, puede haber inestabilidad por el split o por el tamaño del conjunto.

En la version final, la grafica muestra que el modelo si aprendio, aunque la validacion sigue teniendo cierta variacion. Eso es normal cuando se trabaja con datos biologicos y particion por sujeto.

### 9.2 `matriz_confusion.png`

La matriz de confusion final (por ventana de 2 s) fue:

- **Verdadero Sano predicho Sano:** 487
- **Verdadero Sano predicho Parkinson:** 81
- **Verdadero Parkinson predicho Sano:** 153
- **Verdadero Parkinson predicho Parkinson:** 634

#### Como leerla

- Los `81` falsos positivos son ventanas de personas sanas que el modelo marcaria como Parkinson.
- Los `153` falsos negativos son ventanas de pacientes con Parkinson que el modelo no detecto.
- La matriz permite ver el tipo de error, no solo el porcentaje total.

En palabras simples, esta grafica sirve para entender en que se equivoca el modelo y no solo cuantas veces acierta.

---

## 10. Que produce el entrenamiento

Al terminar, el script guarda varios archivos:

- `modelo_final.keras`
- `mejor_modelo.keras`
- `curvas_aprendizaje.png`
- `matriz_confusion.png`
- `resumen_metricas.json`

### 10.1 Para que sirve `modelo_final.keras`

Este archivo es el modelo entrenado ya listo para reutilizarse.

Sirve para:

- cargarlo despues en otra computadora;
- hacer predicciones nuevas sobre otros pacientes;
- integrarlo a una aplicacion de inferencia;
- repetir resultados sin volver a entrenar desde cero.

En palabras simples, `modelo_final.keras` es la “memoria” de la red: ahi queda guardado todo lo que aprendio durante el entrenamiento.

### 10.2 Como se usa despues

El uso normal seria cargarlo con Keras y pasarle una nueva muestra ya preprocesada:

```python
import tensorflow as tf

modelo = tf.keras.models.load_model("modelo_final.keras")
probabilidad = modelo.predict(x_nuevo)
```

Despues esa probabilidad se compara con el umbral de decision elegido, que en la version final fue `0.66`.

Eso significa que no basta con tener el archivo guardado: primero hay que transformar la nueva señal EEG al mismo formato que se uso en entrenamiento, y luego la red devuelve su prediccion.

### 10.3 Como se visualiza

El modelo puede revisarse de varias maneras:

- con `model.summary()` para ver la arquitectura;
- con `plot_model()` para generar un diagrama de la red;
- abriendo las graficas de entrenamiento y la matriz de confusion que genera el script.

En resumen, el modelo no se “ve” como una sola imagen fija, sino como una combinacion de estructura, pesos aprendidos y resultados graficos.

---

## 11. Que cambios ayudaron a mejorar el resultado

Los cambios mas importantes para obtener una mejor version fueron:

- usar un split balanceado y agrupado por persona (sin fuga de datos);
- cambiar la funcion de perdida a `BinaryFocalCrossentropy`;
- usar metricas mas adecuadas para clasificacion biomédica;
- calibrar el umbral de decision sobre validacion;
- guardar el mejor modelo por `val_pr_auc`.

Eso fue lo que hizo que la version final fuera mas defendible que las corridas previas, donde el accuracy podia verse bien pero la clasificacion real era pobre.

---

## 12. Cierre

La Fase 5 entrega un modelo CNN entrenado y evaluado de forma completa para la Rama B. El resultado final no debe leerse solo como una cifra de exactitud, sino como un sistema que aprendio patrones utiles, produjo evidencia visual de su entrenamiento y quedo listo para reutilizarse en la siguiente fase del proyecto.

