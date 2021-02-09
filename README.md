# Instruction
This is a semiauto label tool helps you to label the PASCAL VOC data automatically.
All you need to do is to follow 3 steps:
S1: put your material such as video, raw pictures,  in the certain directory 
S2: change the `label_config.cfg` to choose the model and run the `main.py`
S2: get the xml file with voc format automatically

PS: if you want to get `yolo-format txt`, you can run `train_yolo/voc_annotation.py` to change the formats.

# 1.Env info
## PC env
python >= 3.6.5
win10, mac or linux is ok

## pip list
tensorflow-gpu=1.14.0
keras==2.2.5
opencv==4.2.0
lxml==4.4.2
numpy, matplotlib

**tips:**
			tf2 is ok, if you user your own tf2 model as following: 
			(1)replace of `self-define model` and change the function `auto_label_picdir` at line 72 in 'main.py';
			(2)git clone from [keras_retina](https://github.com/fizyr/keras-retinanet) master branch and replace the dir `keras-retinanet-disable-tf2-behavior` with rebuild the env (or you can use `pip install keras_retina`);


# build the env

`pip install tensorflow-gpu==1.14.0`

`pip install keras==2.2.5`

`pip install opencv-python==4.2.0.32 opencv-contrib-python==4.2.0.32 lxml numpy matplotlib tqdm`

`cd ./keras-retinanet-disable-tf2-behavior`

`pip install . --user`

`python setup.py build_ext --inplace`

# 2.Function & Usage

## (1) auto-label for multi-object- detect 
**Applied for : Video with continous moving objects**

![pic](./info/mot.gif)

## (2) semiauto-label for self-define model
**Applied for: Pictures with specific objects or certain object types**

![pic](./info/selfdefine.gif)


## (3) auto-label for coco types
**Applied for: Pictures with some types in coco**

![pic](./info/retina.gif)

# 3. Parameters Settings
you can change the parameters in `label_config.cfg`

```bash
[FILE]  -> for the location and mood to choose
	xml_dir -> the dir where to place the generated xml file
	pic_dir -> the dir where you put your picture to label
	model_name -> you can use 3 key words to choose the wanted mood: `MOT`, `SELF_DEFINE`, `RETINA`
	[SELF_DEFINE]  -> for the pretrained yolov3 to label specific types in picture
	score_threshold -> scores for yolov3
	involved_classes -> the pretrained yolov3 model's type of your own


[MOT]  -> for the video label process
	interval -> xml file save interval
	set_width -> the width size of wanted picture
	set_height ->the height size of wanted picture
	video -> video path
	tracker -> you can choose one of the MOT model to track the object: `csrt/kcf/boosting/mil/tld/medianflow/mosse`

[RETINA] -> for the retina model to label wanted types of 80 coco types
	retina_weight -> the full path to store the pretrained model 
	coco_classes -> the types you want to label, if you want label all the types, fill it with words `all`
	retina_threshold -> the threshold for model to detect
```


# 5. Train and iteration - self-define

- ## self-define model : YOLOv3
You can use train & iteration to short the label process, if you use choose `self-define` mood to label many  specific types raw picture on your own

**case 1:**
I offer all the `yolov3`'s training py-file in dir `/train_yolo`. The trained model can replace the weight and types in dir `/model` so that you can label your specific-types' picture with the tools. The details you can see `README` in dir `/train_yolo`
**tips:**
remember to download pretrained weight `best_weight_711.h5` to sore in dir `model/models/` for semiauto label; (**types: phone/cigar/person/hat**)
download pretrained weight `yolo_weights.h5` to store in `train_yolo/model_data/` for fine-turning YOLOv3 in training process

**case 2:**
You can also use your own model to detect objects to generate VOC xml. But how? you can change the code in `main.py` from line 70 to line 73;
make sure your model are encapsulated  as followings:

```python
out_boxes, out_scores, out_classes = xxxx_model (image)
```

**Discussion on the Manual label Vs semiauto label**
`as far as gtx 1060 is concerned. Performance may be much better with better GPU.`

```python
My experience:
    it will take 1 day's work for manual label about 1000 raw pictures, 
    it will take about 3-6 hours to train the yolov3 with the labeled 1000 picture
    then you can use this weight to auto label about 5000 raw pic with about 20 minutes while you can use 1-3 hours to refine the xml data
    it will take about 1 day's time to retrained the yolov3 with 6k labeled data
    After 6k's data , it can be used to auto label 1w-2w picture about 1-2 hours, with half day's work to refine the picture.
    The final retrain process may take about 3 days, so the model will be very robost with more than 2w data.
```

**About 2w raw picture to label, for 1 person,  on Nvidia GTX 1060 under 8 hours' working time**

|Manual label|semi-label specific types|auto-label video or some of coco data|
|:--|:--|:--|
|15 days work for label; 3-4 day for train| 2 day's label&refine and 4 days' for train| 2-3 hours |

# 6 Operation - MOT & RETINA

- ## MOT
you can enter words in your keyboard to choose the function:
**tips:**
before beginning, remember to change the type names and corresponding key number at `mot.py` from line 13-17

|key in keyboard|function|
|:--|:--|
|`s`|begin to go into ROI choose process, see the reminder on top , and enter the type you wan to label|
|`Left mouse button`| after in the ROI process, you can choose ROI to track, if you have chose one, then you must enter `enter` to confirm otherwise it will cancel. You can choose many ROIs .After you have chosen all, enter `Esc` to come to the main process|
|`r`|enter `r` to clean all the trackers and re-choose the ROI again|
|`q`|enter `q` to quit all|

- ## RETINA
`coco_classes=all` -> you want to label 80 types

`coco_classes=person/car` -> you only want to label person and car, others are ignored

the type names are showing as followings:
	
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

# Pretrained Model

- YOLOV3: best_weight_711.h5 / yolo_weights.h5
- RetinaNet:  resnet50_coco_best_v2.1.0.h5

see the 
[release](https://github.com/ztfmars/semi_auto_label/releases/tag/keras_tf1_version)

# TO-DO list

- [ ] ROI select to crop image with differ types in video
- [ ] more efficient object-detect model such as EfficientNet, or scaled-yolov4 to replace self-define model part
- [ ] operation GUI
- [ ] GAN series in generate new images/style transfer/faces changes/ super resolution
- [ ] WSOD in annotation or heat-map visual with few manual annotations
- [ ] Vision Transformer to generate specific images
- [ ] incremental learning and self-learning algorithms or methods during iteration and optimazing process 



