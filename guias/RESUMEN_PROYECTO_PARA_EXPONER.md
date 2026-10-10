# Resumen del Proyecto para Exponer

Este documento resume, en lenguaje claro, lo que se ha hecho hasta ahora en el proyecto de deteccion temprana de Parkinson con EEG, por que se tomaron ciertas decisiones y que significan las salidas que vemos en las graficas y archivos procesados.

## 1. De que trata el proyecto

La idea general del proyecto es analizar señales de actividad cerebral, conocidas como EEG, para entrenar modelos de inteligencia artificial que ayuden a distinguir entre personas sanas y personas con enfermedad de Parkinson.

El proyecto usa dos fuentes de datos:

- un dispositivo real de bajo costo llamado **NeuroSky TGAM**, que captura EEG de un solo electrodo frontal
- un **dataset publico** con pacientes sanos y con Parkinson, obtenido de OpenNeuro

La meta es que el entrenamiento fuerte se haga con el dataset publico, y que las sesiones capturadas con el TGAM sirvan para validar si ese entrenamiento puede acercarse a una situacion real.

## 2. Que se ha hecho hasta ahora

Hasta este punto se han completado cuatro partes importantes:

### 1. Lectura del dispositivo TGAM

Se logro conectar el TGAM en Python, leer sus paquetes binarios y extraer correctamente:

- señal cruda o `raw EEG`
- 8 bandas de frecuencia
- `attention`
- `meditation`
- `poor signal`

Esto fue importante porque nos dio una base propia de captura y una forma de entender exactamente que datos entrega el dispositivo.

### 2. Visualizacion en tiempo real

Se desarrollo una interfaz que permite ver la señal en vivo mientras el TGAM esta conectado.

Eso permite:

- revisar si el sensor esta funcionando
- observar la calidad de la señal
- grabar sesiones etiquetadas como `sano` o `parkinson`

### 3. Exportacion de sesiones a CSV

Cada sesion local grabada se guarda en dos archivos:

- `raw.csv`: contiene la señal cruda del EEG
- `bandas.csv`: contiene las 8 bandas y variables lentas como `attention` y `meditation`

Esto permite conservar los datos para analizarlos despues y usarlos en el preprocesamiento.

### 4. Preprocesamiento para aprendizaje automatico

Se construyo un pipeline para transformar los datos en formatos utiles para redes neuronales.

Ese pipeline toma señales EEG y produce dos salidas principales:

- **Rama A**: series temporales listas para modelos secuenciales
- **Rama B**: imagenes GAF listas para modelos tipo CNN

Ademas, se integro y adapto un dataset publico de OpenNeuro para que sea comparable con el TGAM.

## 3. Que decision importante se tomo

Al inicio se pensaba usar varias representaciones, incluyendo espectrogramas y algunas features estadisticas.

Despues de revisar mejor el problema, se tomo una decision clave:

- usar como señal base comun el **raw EEG monocanal**
- dejar las **bandas del TGAM** como apoyo e inspeccion
- usar **series temporales** como representacion principal de la señal
- usar **GAF** como representacion en imagen para la CNN

La razon principal fue que el dataset publico y el TGAM no producen exactamente las mismas salidas avanzadas, pero si comparten algo muy importante:

- ambos pueden representarse como una señal cruda de un canal frontal

Eso hace mas justo el entrenamiento y la comparacion.

## 4. Que es el raw EEG

El `raw EEG` es la señal electrica cruda capturada desde la cabeza.

Se ve como una onda que sube y baja constantemente. A simple vista puede parecer ruido, pero en realidad contiene patrones de actividad cerebral.

Esa señal cruda es valiosa porque:

- no depende tanto de algoritmos propietarios del dispositivo
- existe tanto en el TGAM como en el dataset publico
- a partir de ella se pueden generar otras representaciones

En este proyecto, el `raw EEG` es la materia prima principal.

## 5. Que son las bandas

Las bandas son una forma resumida de la señal EEG, separando la energia de la señal en distintos rangos de frecuencia:

- delta
- theta
- alpha
- beta
- gamma

El TGAM calcula esas bandas internamente y las entrega ya procesadas.

Son utiles para:

- inspeccion rapida
- visualizacion
- trazabilidad

Pero en este proyecto ya no son la entrada principal del entrenamiento, porque no tenemos garantia de que el dataset publico calcule esas bandas exactamente igual que el TGAM.

## 6. Que es una serie temporal

Una serie temporal es simplemente una secuencia de valores ordenados en el tiempo.

En este caso, una ventana de EEG de 2 segundos a 512 Hz se convierte en una secuencia de 1024 muestras.

Eso significa que el modelo puede aprender patrones como:

- cambios de amplitud
- variabilidad
- ritmo
- forma general de la señal

La ventaja de esta representacion es que conserva la señal de forma directa, sin resumirla demasiado.

## 7. Que es GAF

GAF significa **Gramian Angular Field**.

Es una tecnica que toma una serie temporal y la transforma en una imagen.

La idea sencilla es esta:

- primero se normaliza la señal
- luego cada punto de la serie se representa de forma angular
- despues se construye una matriz que relaciona cada instante con los demas

El resultado es una imagen donde aparecen patrones visuales de la dinamica de la señal.

No es una foto del cerebro ni un mapa medico directo. Es una forma matematica de convertir una señal en una imagen para que una CNN pueda analizarla.

## 8. Por que usar GAF

Se eligio GAF porque:

- nace directamente de la serie temporal
- funciona bien con una sola señal
- permite usar CNN 2D sin perder por completo la estructura temporal
- es mas coherente con el TGAM de un solo electrodo que otras representaciones mas complejas

En otras palabras:

- la **Rama A** ve la señal como secuencia
- la **Rama B** ve la misma señal como imagen GAF

## 9. Que significa lo que se ve en las salidas del procesamiento

Despues del preprocesamiento aparecen varias carpetas y archivos que pueden parecer confusos.

Lo importante es esto:

### `datos/openneuro_procesado`

Aqui se guardan las grabaciones individuales del dataset publico, ya adaptadas al proyecto.

Cada archivo `.npz` contiene una grabacion por sujeto o sesion, con:

- `raw_fp1`
- etiqueta sana o Parkinson
- sujeto
- sesion
- frecuencia de muestreo

Este nivel todavia es la base del dataset, no el dataset final de entrenamiento.

### `datos/procesados/rama_a_.../datos.npz`

Aqui ya no hay grabaciones completas. Aqui hay miles de **ventanas cortas** de señal, listas para entrenar un modelo temporal.

Sus salidas principales son:

- `X_train`, `X_val`, `X_test`: ventanas de EEG
- `y_train`, `y_val`, `y_test`: etiquetas

Esto sirve para modelos que leen secuencias.

### `datos/procesados/rama_b_.../datos.npz`

Aqui esas mismas ventanas ya se transformaron en **imagenes GAF**.

Sirve para modelos que trabajan con imagenes, como una CNN.

## 10. Que significa ver una serie temporal o un GAF

Cuando una persona ve una grafica de EEG o una imagen GAF, normalmente no puede decir de inmediato:

- "esto es sano"
- "esto es Parkinson"

Y eso es normal.

Estas salidas no son diagnosticos visuales directos. Son representaciones de la señal para que un modelo aprenda patrones sutiles.

### Lo importante de una serie temporal

En la serie temporal se pueden observar:

- amplitud
- variabilidad
- cambios bruscos
- estabilidad o ruido

Eso ayuda a verificar si la señal tiene calidad suficiente para entrenar.

### Lo importante de un GAF

En la imagen GAF no se busca una figura conocida, como si fuera una radiografia.

Lo que se busca es:

- estructura
- regularidad
- contraste
- repeticion de patrones

La CNN aprende a encontrar diferencias entre esos patrones cuando compara muchas muestras sanas y muchas muestras con Parkinson.

## 11. En que se nota el Parkinson

Esta es una parte muy importante: el Parkinson **no aparece como una marca unica y obvia** en una sola grafica.

Lo que la literatura suele reportar es que puede haber diferencias en:

- ritmos lentos y rapidos del EEG
- distribucion de potencia en ciertas bandas
- variabilidad de la señal
- dinamica temporal

Pero esas diferencias suelen ser:

- sutiles
- variables entre pacientes
- dependientes del estado del paciente, la medicacion y el tipo de registro

Por eso este proyecto no busca que una persona vea una grafica y diagnostique a ojo.

Lo que busca es:

- convertir la señal en representaciones utiles
- entrenar modelos con muchas muestras
- dejar que el modelo aprenda patrones que no siempre son evidentes a simple vista

## 12. Que resultado importante ya obtuvimos

Uno de los resultados mas importantes hasta ahora fue comparar sesiones reales del TGAM contra el dataset publico procesado.

La pregunta era:

> si el TGAM solo tiene un electrodo, realmente se parece lo suficiente al dataset publico como para entrenar con ese dataset?

La conclusion fue:

- **si**, a nivel de `raw EEG` monocanal, hay compatibilidad suficiente para usar el dataset publico como base de entrenamiento
- **no**, no conviene asumir que las bandas del TGAM son exactamente equivalentes al dataset publico

Tambien se vio que:

- dos sesiones locales resultaron compatibles y estables
- una sesion local presento demasiado ruido y conviene descartarla o revisarla

Eso ayudo a justificar el cambio de estrategia hacia:

- `raw EEG`
- series temporales
- GAF

## 13. Por que esto es util para Parkinson

El objetivo final no es solo hacer graficas bonitas.

Lo que se busca es construir un flujo donde:

1. se capture o cargue una señal EEG
2. se limpie y se prepare
3. se convierta en una representacion adecuada
4. se entrene una red neuronal
5. el modelo aprenda a diferenciar patrones asociados con personas sanas y con Parkinson

Si el modelo logra generalizar bien, podria servir como apoyo inicial para evaluaciones neurologicas futuras.

No sustituye un diagnostico medico, pero si podria ser una herramienta de apoyo.

## 14. En que punto va el proyecto

Hasta este momento:

- ya se captura EEG con TGAM
- ya se visualiza en tiempo real
- ya se guardan sesiones etiquetadas
- ya existe el pipeline de preprocesamiento
- ya se proceso el dataset publico
- ya se validó que el dataset publico es compatible con el enfoque de un solo canal
- ya se definio que el entrenamiento seguira principalmente la ruta:
  - series temporales
  - GAF

El siguiente paso es:

- comenzar el entrenamiento de las redes neuronales

## 15. Mensaje final para una exposicion

Una forma simple de explicar el proyecto seria:

> Estamos desarrollando un sistema que toma señales EEG de una persona, las transforma en formatos que una red neuronal pueda entender y busca aprender diferencias entre patrones cerebrales de personas sanas y personas con Parkinson. Para ello combinamos un dispositivo real de bajo costo con un dataset publico mas grande, y decidimos trabajar principalmente con la señal cruda de un canal, tanto en forma de serie temporal como en forma de imagen GAF.

## 16. Idea clave para cerrar

Las graficas raras no son el resultado final, sino la materia prima del analisis.

- La serie temporal muestra como cambia la señal cerebral en el tiempo
- El GAF convierte esa señal en una imagen matematica para que una CNN la pueda estudiar
- Las etiquetas sano/Parkinson permiten entrenar modelos que aprendan diferencias sutiles entre grupos

En resumen: el proyecto ya no esta en la etapa de capturar datos solamente; ya entro a una etapa donde la señal fue organizada y preparada para comenzar el entrenamiento real de los modelos.
