from xml_save import annotation_single_img
from other_utils import read_cfg
import os
import cv2
from pathlib import Path
import sys,re

import numpy as np
from model.yolo import YOLO
from PIL import Image

from mot import mot_detect

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.gpu import setup_gpu

LABEL_CONFIG = "label_config.cfg"
TRAIN_CONFIG = "train_config.cfg"


# load label to names mapping for visualization purposes
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
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
                   79: 'toothbrush'}


class Auto_Detect_Label():
    def __init__(self, json_info):
        self.pic_dir = json_info["pic_dir"]
        self.xml_dir = json_info["xml_dir"]
        self.involed_class_list = json_info["involved_classes"]
        self.score_threshold = json_info["score_threshold"]
        self.model_name = json_info["model_name"]
        # print("[INFO]class list:",  self.involed_class_list)
        self.interval=json_info["interval"]
        self.set_width=json_info["set_width"]
        self.set_height=json_info["set_height"]
        self.video=json_info["video"]
        self.tracker=json_info["tracker"]

        self.retina_weight = json_info["retina_weight"]
        self.coco_classes = json_info["coco_classes"]
        self.retina_threshold = json_info["retina_threshold"]

    def auto_label_picdir(self):
        pic_list = os.listdir(self.pic_dir)

        full_yolo = YOLO()
        no_label_img_list = []

        for pic_name in pic_list:
            mpos = []
            mclass = []
            mscore = []
            pic_fullpath = Path(self.pic_dir).joinpath(pic_name)
            print("[INFO]picfullpath:", pic_fullpath, type(pic_fullpath))

            ########change the model here to your own by following the paradim########
            image = Image.open(pic_fullpath)
            out_boxes, out_scores, out_classes = full_yolo.detect_yolo(image)
            ##########################################################

            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = full_yolo.class_names[c]

                box = out_boxes[i]
                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                y = top
                x = left
                w = right - left
                h = bottom - top
                score = out_scores[i]

                mpos.append((x, y, x + w, y + h))
                mclass.append(predicted_class)
                mscore.append(score)
            no_label_img = annotation_single_img(self.pic_dir, pic_name,
                                                 self.xml_dir, mclass, mpos)

            if no_label_img != None:
                no_label_img_list.append(no_label_img)
        if no_label_img_list != []:
            print("[WARNING] There are some picture which have no label, Please remove them:", \
                  str(no_label_img_list))

    def auto_label_video(self):
        args = {}
        args["interval"] = self.interval
        args["pic_dir"]=self.pic_dir
        args["xml_dir"] = self.xml_dir
        args["set_width"]= self.set_width
        args["set_height"] = self.set_height
        args["video"] = self.video
        args["tracker"] = self.tracker
        print("--" * 20)
        print("MOT detect begin")
        print("--" * 20)
        mot_detect(args)

    def auto_label_coco(self):
        # use this to change which GPU to use
        gpu = 0
        # set the modified tf session as backend in keras
        setup_gpu(gpu)
        model_path = self.retina_weight
        model = models.load_model(model_path, backbone_name='resnet50')

        pic_list = os.listdir(self.pic_dir)
        no_label_img_list = []
        for pic_name in pic_list:
            pic_start = time.time()
            mpos = []
            mclass = []
            mscore = []

            pic_fullpath = Path(self.pic_dir).joinpath(pic_name)
            print("[INFO]picfullpath:", pic_fullpath, type(pic_fullpath))

            image = read_image_bgr(pic_fullpath)
            image = preprocess_image(image)
            image, scale = resize_image(image)
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            boxes /= scale
            class_dict = {}
            if self.coco_classes != "all":
                class_dict = self.coco_classes
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                # if score < 0.5:
                if score < float(self.retina_threshold):
                    break
                involed_class = labels_to_names[label]
                if self.coco_classes != "all" and involed_class not in class_dict:
                    continue
                mpos.append(box)
                mclass.append(involed_class)
                mscore.append(score)
            no_label_img = annotation_single_img(self.pic_dir, pic_name,
                                                 self.xml_dir, mclass, mpos)
            pic_end = time.time()
            print("[INFO]single pic time:", str(pic_end-pic_start))
            if no_label_img != None:
                no_label_img_list.append(no_label_img)
        if no_label_img_list != []:
            print("[WARNING] There are some picture which have no label, Please remove them:", \
                  str(no_label_img_list))


def auto_gen_VOC():
    print("**"*20)
    print("READ CFG PARAMETERS")
    print("**" * 20)

    json_info = read_cfg(flag="label", config_file=LABEL_CONFIG)
    json_info["involved_classes"] = re.split("/", json_info["involved_classes"])
    if json_info["coco_classes"] != "all":
        json_info["coco_classes"] = re.split("/", json_info["coco_classes"])

    json_info["score_threshold"] = float(json_info["score_threshold"])
    model_name = json_info["model_name"]
    print("--"*20)
    print("[INFO] Read cfg parameters:", json_info)
    print("--" * 20)
    if json_info == None:
        sys.exit(0)

    print("**" * 20)
    print("DETECT MODEL LOAD")
    print("**" * 20)
    auto_delab = Auto_Detect_Label(json_info)

    if model_name == "MOT":
        # used for obj detected in video
        auto_delab.auto_label_video()
    elif model_name == "SELF_DEFINE":
        # yolov3 or self-defined models to detect own types
        # auto_delab.auto_label_pic()
        auto_delab.auto_label_picdir()
    elif model_name == "RETINA":
        # mmdet to detect some of coco obj types
        auto_delab.auto_label_coco()




if __name__ == '__main__':
    import time
    start_t = time.time()
    auto_gen_VOC()
    end_t = time.time()

    print("*"*20)
    print("total time:", end_t-start_t)
    print("*"*20)
