# Guia del Inspector NPZ

Esta guia explica como usar la interfaz grafica `inspeccionar_npz_gui.py`.

## 1. Para que sirve

El inspector permite revisar archivos `.npz` del proyecto sin escribir codigo.

Sirve para:

- abrir una grabacion publica procesada de OpenNeuro
- abrir un `datos.npz` de la Rama A
- abrir un `datos.npz` de la Rama B
- ver la serie temporal
- ver el GAF
- revisar claves, shapes, tipos y rangos

## 2. Como abrirlo

Desde la raiz de `07_Proyecto_Python`:

```bash
python inspeccionar_npz_gui.py
```

## 3. Que partes tiene la ventana

### Archivos

- `Archivo principal`: aqui cargas el `.npz` que quieres inspeccionar
- `NPZ de serie opcional`: se usa cuando el archivo principal es de Rama B y quieres ver tambien la serie temporal correspondiente

### Controles

- `Tipo detectado`: la app detecta si el archivo es `publico`, `Rama A` o `Rama B`
- `Split`: elige `train`, `val` o `test` cuando cargas un dataset procesado
- `Indice`: elige que muestra quieres ver dentro del split
- `Inicio (s)`: solo aplica a grabaciones publicas
- `Duracion (s)`: solo aplica a grabaciones publicas
- `Tamano GAF`: resolucion de la imagen GAF que se genera para `publico` y `Rama A`
- `Metodo GAF`: `summation` o `difference`

### Acciones

- `Procesar`: actualiza la vista con la configuracion actual
- `Guardar figura`: exporta la grafica como `png`, `jpg` o `pdf`

### Resumen

En la parte izquierda inferior aparece un resumen con:

- claves del `.npz`
- `shape`
- `dtype`
- `min`
- `max`
- `mean`

### Vista principal

En la derecha aparece:

- serie temporal + GAF si el archivo tiene o puede generar serie
- solo GAF si el archivo principal es de Rama B y no cargaste un NPZ de serie

## 4. Caso 1: ver una grabacion publica procesada

Usa un archivo de:

```text
datos/openneuro_procesado/sub-...npz
```

Pasos:

1. Abrir `Archivo principal`
2. Elegir un archivo de `datos/openneuro_procesado/`
3. Ajustar `Inicio (s)` y `Duracion (s)`
4. Ajustar `Tamano GAF` si quieres
5. Presionar `Procesar`

Que veras:

- a la izquierda de la figura: una ventana de `raw_fp1`
- a la derecha: el GAF calculado con esa ventana

Configuracion recomendada:

- `Inicio (s)`: `10`
- `Duracion (s)`: `2.0`
- `Tamano GAF`: `128`
- `Metodo GAF`: `summation`

## 5. Caso 2: ver una muestra de la Rama A

Usa un archivo como:

```text
datos/procesados/rama_a_raw_v2.0s_o50_minmax/datos.npz
```

Pasos:

1. Abrir `Archivo principal`
2. Elegir el `datos.npz` de Rama A
3. Elegir `Split`
4. Elegir `Indice`
5. Dejar `Tamano GAF = 128`
6. Presionar `Procesar`

Que veras:

- serie temporal ya preprocesada
- GAF generado desde esa misma serie

Este caso es muy util para comprobar si las ventanas realmente se ven limpias antes de entrenar.

## 6. Caso 3: ver una muestra de la Rama B

Usa un archivo como:

```text
datos/procesados/rama_b_gaf_v2.0s_o50/datos.npz
```

Pasos:

1. Abrir `Archivo principal`
2. Elegir el `datos.npz` de Rama B
3. Elegir `Split`
4. Elegir `Indice`
5. Presionar `Procesar`

Que veras:

- el GAF guardado en el dataset

En este caso no se recalcula el GAF. Solo se muestra la imagen ya almacenada.

## 7. Caso 4: ver Rama B junto con su serie temporal

Este es el caso mas util cuando quieres validar si el GAF corresponde a la señal correcta.

Pasos:

1. Cargar en `Archivo principal` el `datos.npz` de Rama B
2. Cargar en `NPZ de serie opcional` el `datos.npz` de Rama A correspondiente
3. Elegir el mismo `Split`
4. Elegir el mismo `Indice`
5. Presionar `Procesar`

Que veras:

- serie temporal desde Rama A
- GAF almacenado desde Rama B

Importante:

- ambos archivos deben venir del mismo preprocesamiento
- deben tener la misma particion y el mismo orden de muestras

## 8. Como interpretar lo que ves

### Serie temporal

Busca:

- amplitud razonable
- forma continua
- ausencia de saturaciones raras
- ausencia de bloques planos

### GAF

Busca:

- patrones definidos, no ruido completamente caotico
- contraste visible
- consistencia entre muestras del mismo tipo

El GAF no se interpreta igual que una señal EEG cruda. Su valor principal es dar una representacion visual consistente para CNN 2D.

## 9. Problemas comunes

### No detecta el tipo de archivo

Pasa si el `.npz` no tiene la estructura esperada del proyecto.

Revisa que sea uno de estos:

- `datos/openneuro_procesado/sub-...npz`
- `datos/procesados/rama_a_.../datos.npz`
- `datos/procesados/rama_b_.../datos.npz`

### La ventana sale vacia o da error

En grabaciones publicas:

- reduce `Inicio (s)`
- verifica que `Duracion (s)` no rebase el final de la grabacion

### Rama B no muestra serie temporal

Es normal si no cargas `NPZ de serie opcional`.

### El indice da error o no muestra lo esperado

Verifica:

- el `Split`
- el rango del `Indice`
- que Rama A y Rama B correspondan al mismo dataset si los estas emparejando

## 10. Flujo recomendado de uso

1. Abrir una grabacion publica y revisar una ventana cruda
2. Abrir Rama A y revisar varias series preprocesadas
3. Abrir Rama B y revisar varios GAF
4. Emparejar Rama A + Rama B para validar consistencia
5. Guardar figuras utiles para reporte o presentacion

## 11. Relacion con otras guias

- `GUIA_NPZ.md`: explica que guarda cada `.npz`
- `FLUJO_OPERATIVO.md`: explica en que momento del proyecto se usa el inspector
