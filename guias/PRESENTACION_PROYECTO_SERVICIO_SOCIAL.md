# Presentacion del proyecto de servicio social

## Deteccion experimental de Parkinson mediante EEG de bajo costo

**Duracion objetivo:** 10 minutos  
**Presenta:** Adolfo Huerta Martinez, estudiante de octavo semestre

> La diapositiva debe mostrar ideas y cifras clave. El texto de la seccion
> `Guion oral` es una guia para explicar, no para copiar completo en pantalla.

---

## Diapositiva 1 - Presentacion y pregunta central

### En pantalla

**EEG de bajo costo + inteligencia artificial**  
Una plataforma experimental para estudiar patrones relacionados con Parkinson

- Adolfo Huerta Martinez
- Estudiante de octavo semestre
- Proyecto de servicio social

### Guion oral

Mi nombre es Adolfo Huerta Martinez, soy estudiante de octavo semestre y estoy
terminando mis materias. Durante mi servicio social he trabajado en un proyecto
que combina EEG, programacion e inteligencia artificial.

La pregunta central es: ¿podemos usar un dispositivo EEG de bajo costo para
capturar una senal cerebral, procesarla y estudiar si existen patrones que
ayuden a diferenciar entre personas sanas y personas con Parkinson?

La palabra importante es **estudiar**. El sistema todavia no diagnostica y no
pretende sustituir a un neurologo.

**Tiempo:** 40 segundos

---

## Diapositiva 2 - El problema y la oportunidad

### En pantalla

- El Parkinson requiere evaluacion especializada.
- El EEG puede aportar informacion sobre la actividad cerebral.
- Los equipos clinicos completos pueden ser costosos y poco accesibles.
- Una plataforma de bajo costo podria apoyar investigacion y cribado futuro.

### Guion oral

El Parkinson es una enfermedad compleja. No existe una unica marca visible en
una grafica de EEG que permita identificarla de manera directa. Los patrones
pueden depender de la edad, la medicacion, el estado de la persona, el ruido y
el protocolo de captura.

Por eso el proyecto no intenta reemplazar una evaluacion clinica. La
oportunidad es construir una herramienta accesible que permita capturar datos,
organizarlos y buscar patrones con metodos reproducibles de aprendizaje
automatico.

**Tiempo:** 50 segundos

---

## Diapositiva 3 - Objetivo y alcance actual

### En pantalla

**Objetivo:** construir y evaluar un flujo completo para analizar EEG de un
canal.

```text
capturar -> visualizar -> guardar -> preparar -> entrenar -> evaluar
```

**Alcance actual:** plataforma experimental, no dispositivo medico.

### Guion oral

El objetivo no era comenzar directamente con una red neuronal. Primero habia
que resolver toda la cadena de datos: comunicarnos con el sensor, entender su
protocolo, verificar la calidad de la senal, visualizarla, guardarla y despues
prepararla para los modelos.

Actualmente contamos con esa cadena y con un primer modelo experimental. Lo que
todavia falta es demostrar que el resultado generaliza a personas nuevas y a
sesiones capturadas directamente con el TGAM.

**Tiempo:** 40 segundos

---

## Diapositiva 4 - Que construimos con Python

### En pantalla

- `pyserial`: comunicacion con el puerto COM.
- Parser propio del protocolo binario TGAM.
- `PyQt6` y `PyQtGraph`: interfaz y graficas en tiempo real.
- `NumPy` y `Pandas`: buffers, arreglos y archivos CSV.
- `MNE-Python`: lectura y adaptacion de EEG publico.
- `scikit-learn` y `TensorFlow/Keras`: preparacion y CNN.

### Guion oral

La parte de programacion se dividio en modulos. `pyserial` permite leer el
puerto del dispositivo. El parser interpreta los paquetes binarios y revisa el
checksum para descartar informacion corrupta.

Para visualizar usamos PyQt6 y PyQtGraph, porque necesitamos actualizar una
senal de aproximadamente 512 muestras por segundo sin que la interfaz se
vuelva lenta. NumPy ayuda con los buffers circulares y el procesamiento de
senales.

Para el aprendizaje automatico usamos datos publicos procesados con
herramientas de EEG y una CNN implementada con TensorFlow/Keras.

**Tiempo:** 55 segundos

---

## Diapositiva 5 - Flujo tecnico del sistema

### En pantalla

```text
NeuroSky TGAM
      |
      v
Puerto serial -> parser y checksum -> EEG raw
                                      |
                 +--------------------+--------------------+
                 v                                         v
          visualizacion                              grabacion CSV
                                                        |
                                                        v
                         ventanas de 2 s / 1024 muestras
                                      |
                         +------------+------------+
                         v                         v
                 Rama A: serie raw          Rama B: imagen GAF
                                                       |
                                                       v
                                                   CNN 2D
```

### Guion oral

El TGAM envia paquetes por el puerto serial. El parser identifica la senal
raw, las bandas, la atencion, la meditacion y la calidad de senal.

La senal puede verse en tiempo real y tambien almacenarse. Despues se divide
en ventanas de dos segundos, que equivalen a 1,024 muestras cuando trabajamos
a 512 Hz.

Con cada ventana tenemos dos rutas. La Rama A conserva la senal como una serie
temporal para modelos secuenciales. La Rama B convierte la misma senal en una
imagen GAF para analizarla con una red convolucional.

**Tiempo:** 1 minuto 5 segundos

---

## Diapositiva 6 - Datos y decisiones de investigacion

### En pantalla

- TGAM: un canal frontal, raw EEG y variables de apoyo.
- OpenNeuro `ds002778`: 31 sujetos y 46 grabaciones.
- Se usa Fp1 como aproximacion al canal frontal del TGAM.
- Ventanas de 2 s con 50% de solapamiento.
- Division por persona: nadie aparece en entrenamiento y prueba a la vez.

### Guion oral

El sensor TGAM tiene un solo electrodo frontal. El dataset publico tiene mas
canales, pero para hacer una comparacion mas justa seleccionamos Fp1, que es una
ubicacion frontal.

Una decision importante fue trabajar principalmente con el raw EEG. Las bandas
que calcula el TGAM son utiles para visualizar, pero no podemos asumir que se
calcularon exactamente igual en el dataset publico. En cambio, ambos conjuntos
pueden compararse como senales raw de un canal.

Tambien dividimos los datos por persona. Cada paciente con Parkinson tiene dos
grabaciones, con y sin medicacion. Al principio las tratabamos como personas
distintas y una misma persona podia quedar en entrenamiento y en prueba. Lo
detectamos y lo corregimos: si el modelo ya vio a alguien, el resultado puede
parecer bueno sin que realmente aprenda a generalizar.

**Tiempo:** 1 minuto

---

## Diapositiva 7 - Que significa GAF y que modelo usamos

### En pantalla

**GAF:** transforma una serie temporal en una imagen matematica.

```text
EEG raw -> normalizacion -> GAF 128 x 128 -> CNN 2D
```

**CNN compacta:**

- Conv2D: 32 -> 64 -> 128 filtros
- Batch normalization y pooling
- Global average pooling
- Dense + dropout
- Salida sigmoid: probabilidad de clase

### Guion oral

GAF significa Gramian Angular Field. No es una imagen del cerebro ni una
radiografia. Es una representacion matematica que relaciona los distintos
instantes de una serie temporal.

Esto permite aplicar una CNN, que es una arquitectura muy utilizada para
encontrar estructuras locales en imagenes. La red recibe las imagenes GAF y
aprende patrones asociados con las etiquetas sano y Parkinson.

La salida debe interpretarse como una probabilidad o puntuacion experimental,
no como una conclusion medica.

**Tiempo:** 55 segundos

---

## Diapositiva 8 - Resultados obtenidos hasta ahora

### En pantalla

**Ingenieria**

- Lectura estable del TGAM a aproximadamente 512 Hz.
- Parser con deteccion de paquetes corruptos.
- Visualizacion cercana a 30 FPS.
- Grabacion separada de raw y bandas.
- Dos sesiones locales compatibles; una requiere revision por ruido.

**CNN GAF, evaluada en 5 personas que nunca vio**

- ROC AUC: **0.899**
- Balanced accuracy: **83.1%**
- Recall: **80.6%**
- Precision: **88.7%**
- 9 de 9 personas no vistas ordenadas correctamente (val + test)

### Guion oral

Los primeros resultados no son solamente una cifra de accuracy. Primero
demostramos que el sistema captura y procesa datos reales de manera estable.
Despues comparamos sesiones locales con el dataset publico y encontramos que
el raw monocanal es suficientemente compatible para continuar con este
enfoque.

Con la division por persona, la CNN obtuvo un ROC AUC de 0.899 y una
balanced accuracy de 83 por ciento en cinco personas que nunca vio. En las
nueve personas que el modelo no vio durante el entrenamiento, todos los
pacientes con Parkinson recibieron una probabilidad mayor que todos los
controles.

Pero el margen es pequeno y el conjunto de prueba es de solo cinco personas:
si cambian esas personas, las cifras pueden cambiar bastante. Es una senal
prometedora, no una precision clinica.

**Tiempo:** 1 minuto 15 segundos

---

## Diapositiva 9 - Que falta y que se puede mejorar

### En pantalla

**Siguiente etapa tecnica**

1. Repetir el entrenamiento con distintas particiones de personas.
2. Conseguir mas personas, en especial pacientes con Parkinson grabados con TGAM.
3. Comparar CNN 2D, 1D-CNN, LSTM y GRU (Rama A).
4. Construir inferencia reproducible desde una sesion TGAM.
5. Probar con personas nuevas y mas datos.

### Guion oral

El siguiente reto no es agregar mas capas a la red sin control. Es saber si el
resultado se mantiene cuando cambiamos de sujeto, de sesion o de dispositivo.

Ya evaluamos agrupando por persona; ahora necesitamos repetirlo con distintas
particiones para saber cuanto depende de quienes caen en prueba. Despues podemos comparar arquitecturas: una CNN sobre GAF, una CNN sobre raw,
modelos secuenciales como LSTM o GRU y tambien modelos clasicos como SVM o
Random Forest.

Finalmente debemos conectar el preprocesamiento y la inferencia con nuevas
sesiones del TGAM, manteniendo exactamente las mismas transformaciones usadas
durante el entrenamiento.

**Tiempo:** 1 minuto

---

## Diapositiva 10 - Impacto y convocatoria

### En pantalla

**Posible impacto**

- Investigacion accesible sobre EEG y Parkinson.
- Apoyo a cribado o seguimiento, si se valida correctamente.
- Herramientas abiertas para capturar y analizar senales.
- Formacion en Python, senales, IA y trabajo interdisciplinario.

**Busco personas que quieran continuar el proyecto**

- Programacion y GUI
- Procesamiento de senales
- Machine learning
- Documentacion y pruebas
- Investigacion sobre Parkinson

### Guion oral

El impacto posible esta en hacer mas accesible la experimentacion con EEG y en
crear una base tecnica que otros estudiantes puedan continuar. A largo plazo,
si se reunen mas datos y se realiza una validacion clinica rigurosa, podria
convertirse en una herramienta de apoyo para investigacion, cribado o
seguimiento.

Pero el proyecto necesita mas trabajo antes de llegar a ese punto: mas sujetos,
mejores controles, validacion externa y colaboracion con personal medico.

Como estoy terminando mis materias, quiero invitar a estudiantes que deseen
sumarse al servicio social. Hay trabajo tanto para quienes programan como para
quienes quieran aprender procesamiento de senales, redes neuronales,
documentacion o investigacion sobre Parkinson.

Mi propuesta es que quien tenga interes se acerque al final para explicarle con
mas detalle el repositorio, el flujo actual y una posible tarea de entrada.

**Tiempo:** 1 minuto 10 segundos

---

## Frase de cierre

> Ya construimos la base: podemos capturar EEG de bajo costo, visualizarlo,
> almacenarlo, procesarlo y entrenar un primer modelo. El futuro del proyecto
> depende de demostrar que ese modelo funciona con personas nuevas y de
> convertir la plataforma experimental en una herramienta validada, siempre
> como apoyo y nunca como sustituto del diagnostico medico.

---

## Recomendaciones visuales

- Usar una captura de la interfaz de visualizacion en la diapositiva 4 o 5.
- Mostrar un ejemplo de raw EEG y un GAF lado a lado en la diapositiva 7.
- Mostrar la matriz de confusion junto a las metricas de la diapositiva 8.
- No llenar las diapositivas con el guion oral.
- Mantener visibles las palabras `experimental`, `un canal` y `no diagnostico`.
- Reservar aproximadamente 30 segundos finales para recibir interesados.

## Preguntas que pueden hacerte

### ¿El sistema ya detecta Parkinson?

No. Ya existe un primer clasificador experimental, pero todavia no hay
validacion clinica ni evidencia suficiente para usarlo como diagnostico.

### ¿Por que usar un dispositivo de un solo canal?

Porque es economico, accesible y permite estudiar si una señal frontal puede
aportar informacion util. La desventaja es que contiene menos informacion
espacial que un EEG clinico de multiples canales.

### ¿Por que usar datos publicos si tenemos el TGAM?

El dataset publico ofrece mas grabaciones etiquetadas para entrenar. El TGAM
permite comprobar si el flujo es aplicable a un dispositivo de bajo costo. La
compatibilidad debe verificarse y no darse por sentada.

### ¿Que necesitan los siguientes colaboradores?

Necesitamos personas que puedan ayudar a repetir experimentos, mejorar la
validacion, conectar la inferencia en tiempo real, documentar el sistema y
buscar mejores formas de relacionar la senal con el problema neurologico.

### ¿Que pasa si una medicion tiene mucho ruido?

Se revisa la calidad de senal y puede descartarse o marcarse para revision. Un
modelo no debe recibir datos de mala calidad y producir una respuesta como si
fuera confiable.
