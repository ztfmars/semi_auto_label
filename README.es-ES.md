# Instrucción
Esta es una herramienta de etiquetado semiautomático que te ayuda a etiquetar datos PASCAL VOC automáticamente.
Todo lo que necesitas hacer es seguir 3 pasos:
S1: coloca tus materiales como video, imágenes raw, en el directorio correspondiente
S2: cambia `label_config.cfg` para elegir el modelo y ejecuta `main.py`
S2: obtén el archivo xml con formato VOC automáticamente

PS: si quieres obtener `txt formato yolo`, puedes ejecutar `train_yolo/voc_annotation.py` para cambiar los formatos.

# 1. Información del entorno
## Entorno PC
```bash
python >= 3.6.5
win10, mac o linux esta bien
```
## pip list
```bash
tensorflow-gpu=1.14.0
keras==2.2.5
opencv==4.2.0
lxml==4.4.2
numpy, matplotlib
```
**consejos:**
tf2 esta bien, si usas tu propio modelo tf2 como sigue:

(1)reemplaza `modelo definido por el usuario` y cambia la función `auto_label_picdir` en la línea 72 en 'main.py';

(2)git clone desde [keras_retina](https://github.com/fizyr/keras-retinanet) rama master y reemplaza el directorio `keras-retinanet-disable-tf2-behavior` con reconstruir el entorno (o puedes usar `pip install keras_retina`);


# construir el entorno

`pip install tensorflow-gpu==1.14.0`

`pip install keras==2.2.5`

`pip install opencv-python==4.2.0.32 opencv-contrib-python==4.2.0.32 lxml numpy matplotlib tqdm`

`cd ./keras-retinanet-disable-tf2-behavior`

`pip install . --user`

`python setup.py build_ext --inplace`

# 2.Función & Uso

## (1) etiquetado automático para detección multi-objeto 
**Aplicado para : Video con objetos en movimiento continuo**

![pic](./info/mot.gif)

## (2) etiquetado semiautomático para modelo definido por el usuario
**Aplicado para: Imágenes con objetos específicos o ciertos tipos de objetos**

![pic](./info/selfdefine.gif)


## (3) etiquetado automático para tipos COCO
**Aplicado para: Imágenes con algunos tipos en COCO**

![pic](./info/retina.gif)

## Video tutorial (haga clic en la imagen de abajo para ver el detalle)

[![Watch the video](https://github.com/ztfmars/semi_auto_label/blob/main/info/cover.JPG)](https://www.bilibili.com/video/BV1ih411C7j1/)

# 3. Configuración de parámetros
puedes cambiar los parámetros en `label_config.cfg`

```bash
[FILE]  -> para la ubicación y estado a elegir
	xml_dir -> el directorio donde colocar el archivo xml generado
	pic_dir -> el directorio donde pones tu imagen a etiquetar
	model_name -> puedes usar 3 palabras clave para elegir el estado deseado: `MOT`, `SELF_DEFINE`, `RETINA`
	[SELF_DEFINE]  -> para el yolov3 preentrenado etiquetar tipos específicos en la imagen
	score_threshold -> puntuaciones para yolov3
	involved_classes -> los tipos del modelo yolov3 preentrenado que posees


[MOT]  -> para el proceso de etiquetado de video
	interval -> intervalo de guardado de archivos xml
	set_width -> el ancho deseado de la imagen
	set_height -> la altura deseada de la imagen
	video -> ruta del video
	tracker -> puedes elegir uno de los modelos MOT para seguir el objeto: `csrt/kcf/boosting/mil/tld/medianflow/mosse`

[RETINA] -> para el modelo retina etiquetar tipos deseados de 80 tipos COCO
	retina_weight -> la ruta completa para almacenar el modelo preentrenado
	coco_classes -> los tipos que quieres etiquetar, si quieres etiquetar todos los tipos, rellénalo con palabras `all`
	retina_threshold -> el umbral para la detección del modelo
```


# 5. Entrenamiento e iteración - definido por el usuario

- ## Modelo definido por el usuario : YOLOv3
Puedes usar entrenamiento & iteración para acortar el proceso de etiquetado, si usas el estado `definido por el usuario` para etiquetar muchas imágenes raw de tipos específicos en tu propio

**caso 1:**
Ofrezco todos los archivos de entrenamiento de yolov3 en el directorio `/train_yolo`. Después de tu proceso de entrenamiento, el nuevo modelo entrenado puede usarse para reemplazar el peso en el directorio `/model/models/best_weight_711.h5` para que puedas etiquetar tus imágenes de tipos específicos con las herramientas. Los detalles puedes ver `README` en el directorio `/train_yolo`

descarga el peso preentrenado `yolo_weights.h5` y almacénalo en `train_yolo/model_data/` para ajustar finamente tus propios pesos de tipo con YOLOv3


**consejos:**

	recuerda descargar el peso preentrenado `best_weight_711.h5` y almacenarlo en el directorio `model/models/` si quieres etiquetar semiautomáticamente para los siguientes 4 tipos:
	(tipos: phone/cigar/person/hat)

**caso 2:**
También puedes usar tu propio modelo para detectar objetos y generar xml VOC. ¿Pero cómo? puedes cambiar el código en `main.py` desde la línea 70 hasta la 73;
asegúrate de que tu modelo esté encapsulado como sigue:

```python
out_boxes, out_scores, out_classes = xxxx_model (image)
```

**Discusión sobre el etiquetado manual Vs etiquetado semiautomático**
`en cuanto a gtx 1060 se refiere. El rendimiento puede ser mucho mejor con una mejor GPU.`

```python
Mi experiencia:
    tomará 1 día de trabajo para etiquetar manualmente unas 1000 imágenes raw,
    tomará unos 3-6 horas entrenar yolov3 con las 1000 imágenes etiquetadas
    luego puedes usar este peso para etiquetar automáticamente unas 5000 imágenes raw en unos 20 minutos mientras puedes usar 1-3 horas para refinar los datos xml
    tomará un día de trabajo entrenar nuevamente yolov3 con 6k datos etiquetados
    Después de 6k datos, puede usarse para etiquetar automáticamente 10k-20k imágenes en unas 1-2 horas, con medio día de trabajo para refinar las imágenes.
    El proceso final de entrenamiento puede tomar unos 3 días, por lo que el modelo será muy robusto con más de 20k datos.
```

**Sobre 20k imágenes raw a etiquetar, para 1 persona, en Nvidia GTX 1060 bajo 8 horas de tiempo de trabajo**

|Etiquetado manual|etiquetado semiautomático de tipos específicos|etiquetado automático de video o algunos datos de COCO|
|:--|:--|:--|
|15 días de trabajo para etiquetar; 3-4 días para entrenar| 2 días de etiquetado & refinar y 4 días para entrenar| 2-3 horas |

# 6 Operación - MOT & RETINA

- ## MOT
puedes escribir palabras en tu teclado para elegir la función:
**consejos:**
antes de comenzar, recuerda cambiar los nombres de los tipos y el número de teclas correspondientes en `mot.py` desde la línea 13-17

|tecla en teclado|función|
|:--|:--|
|`s`|empezar a entrar en el proceso de selección ROI, ver el recordatorio en la parte superior, y escribir el tipo que quieres etiquetar|
|`botón izquierdo del ratón`| después de estar en el proceso ROI, puedes elegir ROI para seguir, si has elegido uno, entonces debes presionar `enter` para confirmar, de lo contrario se cancelará. Puedes elegir muchos ROIs. Después de haber elegido todos, presiona `Esc` para entrar en el proceso principal|
|`r`|escribe `r` para limpiar todos los seguidores y volver a elegir el ROI|
|`q`|escribe `q` para salir de todo|

- ## RETINA
`coco_classes=all` -> quieres etiquetar 80 tipos

`coco_classes=person/car` -> solo quieres etiquetar persona y coche, otros se ignoran

los nombres de los tipos son los siguientes:
	
		       0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
	               7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
	               13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
	               21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
	               28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
	               34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
	               39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
	               46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
	               53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
	               60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
	               67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
	               73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
	               79: 'toothbrush'

# Modelo Preentrenado

- YOLOV3: 
		
	**¿función?**
		
		best_weight_711.h5 -> yolov3 preentrenado para detectar `phone/cigar/person/hat`
		
		yolo_weights.h5 -> puede usarse para ajustar finamente YOLOv3 para tus propios tipos
		
	**¿en qué directorio almacenarlos?**
		
		best_weight_711.h5 -> `/model/models/`
		
		yolo_weights.h5 -> `train_yolo/model_data/`
		
- RetinaNet:  
		
	**función**
		
		resnet50_coco_best_v2.1.0.h5 -> peso del modelo retina para detectar tipos COCO
		
	**¿en qué directorio almacenarlos?**
		
		resnet50_coco_best_v2.1.0.h5 -> `/load_weight/`

ver el 
[release](https://github.com/ztfmars/semi_auto_label/releases/tag/keras_tf1_version)

# Lista de tareas pendientes

- [ ] Selección ROI para recortar imágenes con diferentes tipos en video
- [ ] Más modelos de detección eficientes como EfficientNet, o scaled-yolov4 para reemplazar la parte del modelo definido por el usuario
- [ ] Interfaz gráfica de usuario
- [ ] Series GAN para generar nuevas imágenes / transferencia de estilo / cambios de rostros / superresolución
- [ ] Aprendizaje con pocos datos (WSOD) para anotaciones o visualización de mapas de calor con pocas anotaciones manuales
- [ ] Vision Transformer para generar imágenes específicas
- [ ] Algoritmos o métodos de aprendizaje incremental y autoaprendizaje durante el proceso de iteración y optimización
