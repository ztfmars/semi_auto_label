import os
import configparser
import sys
import sys

sys.path.insert(0, '../')

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

LABEL_CONFIG = "label_config.cfg"
TRAIN_CONFIG="train_config.cfg"

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


def retina_detect_img(model_path="load_weight/resnet50_coco_best_v2.1.0.h5",
                                          img_path="voc/JPEGImages/1.jpg"):
    # use this to change which GPU to use
    gpu = 0
    # set the modified tf session as backend in keras
    setup_gpu(gpu)
    # ## Load RetinaNet model
    # adjust this to point to your downloaded/trained model
    # models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')
    # if the model is not converted to an inference model, use the line below
    # see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
    # model = models.convert_model(model)
    # print(model.summary())
    # ## Run detection on example
    # load image
    image = read_image_bgr(img_path)
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    # process image
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    # correct for image scale
    boxes /= scale
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        # if score < 0.5:
        if score < 0.38:
            break
        color = label_color(label)
        b = box.astype(int)
        draw_box(draw, b, color=color)
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()

def read_cfg(flag="label", config_file=LABEL_CONFIG):
    print("[INFO] cfg path:", os.path.join(os.getcwd(), config_file))
    if os.path.exists(os.path.join(os.getcwd(), config_file)):
        config_dict = {}
        config = configparser.ConfigParser()
        config.read(config_file)

        if flag == "label":
            config_dict["xml_dir"] = config.get("FILE", "xml_dir")
            config_dict["pic_dir"] = config.get("FILE", "pic_dir")
            config_dict["model_name"] = config.get("FILE", "model_name")

            config_dict["score_threshold"] = config.get("SELF_DEFINE", "score_threshold")
            config_dict["involved_classes"] = config.get("SELF_DEFINE", "involved_classes")

            config_dict["interval"]=config.get("MOT", "interval")
            config_dict["set_width"] = config.get("MOT", "set_width")
            config_dict["set_height"] = config.get("MOT", "set_height")
            config_dict["video"] = config.get("MOT", "video")
            config_dict["tracker"] = config.get("MOT", "tracker")

            config_dict["retina_weight"] = config.get("RETINA", "retina_weight")
            config_dict["coco_classes"] = config.get("RETINA", "coco_classes")
            config_dict["retina_threshold"] = config.get("RETINA", "retina_threshold")

        elif flag == "train":
            print("[ERROR] to be cotinue")
            import sys
            sys.exit(0)
            # config_dict["model_weight"] = config.get("TRAIN", "model_weight")
            # config_dict["Freeze_epoch"] = config.get("TRAIN", "load_weight")
            # config_dict["Epoch"] = config.get("TRAIN", "epochs")
        else:
            print("[ERROR] Wrong Flag, only support 'label' or 'train' ")
            import sys
            sys.exit(0)

        return config_dict
    else:
        print("[ERROR] cfg path is not existed!")
        return None