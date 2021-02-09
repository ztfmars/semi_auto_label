# -------------------------------------#
#       调用摄像头检测
# -------------------------------------#
from keras.layers import Input
from .yolo import YOLO, tinyYOLO
from PIL import Image
import numpy as np
import cv2
import time
import os
from .font.hanzi import put_chinese_text
from pathlib import Path
from xml_save import annotation_single_img


# 基本路径和加载字库
base_path = os.path.abspath(os.path.dirname(__file__))
base_path = base_path + '/'
ft = put_chinese_text(base_path + 'font/yahei.ttf')

mfilter = ["person"]


def tiny_object_detect(tiny_yolo, current_frame):
    # 图像推理(yolo3检测)
    mpos = []
    mclass = []
    mscore = []

    frame0 = current_frame[:, :, ::-1].copy()
    image = Image.fromarray(frame0)
    out_boxes, out_scores, out_classes = tiny_yolo.detect_yolo(image)

    # print("yolo detect box:%s"%str(out_boxes))
    # print("yolo detect score:%s" % str(out_scores))
    # print("yolo detect class:%s" % str(out_classes))

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = tiny_yolo.class_names[c]
        # print(predicted_class)
        # 只是识别person，其余类别忽略
        if not (predicted_class in mfilter):
            continue
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

    return  mclass, mpos, mscore


def yolo_object_detect(current_frame, myyolo, crop_pos):
    # 图像推理(yolo3检测)
    mpos = []
    mclass = []
    mscore = []

    frame0 = current_frame[:, :, ::-1].copy()
    image = Image.fromarray(frame0)
    print("origin pic shape:(%f,%f)"%(image.size[0], image.size[1]))
    for crop_num in range(len(crop_pos)):
        boxt = crop_pos[crop_num]
        topt, leftt, bottomt, rightt = boxt
        print("#"*10)
        print("(x0,y0,x1,y1)",(topt, leftt, bottomt, rightt))

        topt = max(0, np.floor(topt + 0.5).astype('int32'))
        leftt = max(0, np.floor(leftt + 0.5).astype('int32'))
        bottomt = min(image.size[1], np.floor(bottomt + 0.5).astype('int32'))
        rightt = min(image.size[0], np.floor(rightt + 0.5).astype('int32'))

        new_image = image.crop((leftt, topt, rightt, bottomt))

        out_boxes, out_scores, out_classes = myyolo.detect_yolo(new_image)

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = myyolo.class_names[c]

            box = out_boxes[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32')) + topt
            left = max(0, np.floor(left + 0.5).astype('int32')) + leftt
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32')) + topt
            right = min(image.size[0], np.floor(right + 0.5).astype('int32')) + leftt
            y = top
            x = left
            w = right - left
            h = bottom - top
            score = out_scores[i]

            mpos.append((x, y, x + w, y + h))
            mclass.append(predicted_class)
            mscore.append(score)

    return mclass, mpos, mscore


def main():
    full_yolo = YOLO()
    tiny_yolo = tinyYOLO()
    # 调用摄像头
    #vs = 0
    vs = 'rtsp://admin:admin123@192.168.9.200:554/Streaming/Channels/1/'
    capture = cv2.VideoCapture(vs)  # capture=cv2.VideoCapture("1.mp4")

    num = 0
    while capture.isOpened():

        num = num + 1

        # 读取某一帧
        ret, frame = capture.read()
        if ret is False:
            capture = cv2.VideoCapture(vs)
            ret, frame = capture.read()
        #cv2.namedWindow("frame", 0)
        #cv2.setWindowProperty("frame", cv2.WND_PROD_FULLSCREEN, \
                #cv2.WINDOW_FULLSCREEN)
        try:
            # 跳帧法显示相关的图像
            if num%2:
                continue

            else:
                tclass, tpos, tscore = tiny_object_detect(tiny_yolo=tiny_yolo,
                                                          current_frame=frame)
                if tpos:
                    mclass, mpos, mscore = yolo_object_detect(current_frame=frame,
                                                              crop_pos=tpos,
                                                              myyolo=full_yolo)

                    # 统一显示box和标签,判断边界
                    for item in range(len(mclass)):
                        if mclass[item] == 'cigar':
                            mColor = (0, 0, 255)
                        else:
                            mColor = (0, 255, 0)
                        (x0, y0, x1, y1) = mpos[item]
                        frame = ft.draw_text(frame, (x0, y0 - 24), '类型:%s，分数：%s' %(mclass[item],mscore[item]), 14, mColor)
                        cv2.rectangle(frame, (x0, y0), (x1, y1), mColor, 2)

                cv2.imshow("frame", frame)
        except Exception as e:
            print("[ERROR]",e)
            cv2.waitKey(1)
            time.sleep(1)

            capture = cv2.VideoCapture(vs)
            continue

        c = cv2.waitKey(30) & 0xff
        if c == 27:
            capture.release()
            break
    tiny_yolo.close_session()
    full_yolo.close_session()




def yolo3_detect_picdir(pic_dir, xml_dir, score_threshold, involed_class_list, pic_list):
    full_yolo = YOLO()
    no_label_img_list = []

    for pic_name in pic_dir:
        mpos = []
        mclass = []
        mscore = []
        pic_fullpath = Path(pic_dir).joinpath(pic_name)
        print("picfullpath", pic_fullpath, type(pic_fullpath))
        img = cv2.imread(str(pic_fullpath))
        pic_size = img.shape
        new_size = (pic_size[1], pic_size[0], pic_size[2])
        image = Image.open(pic_fullpath)
        out_boxes, out_scores, out_classes = full_yolo.detect_yolo(image)

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
        no_label_img = annotation_single_img(pic_dir, pic_name,
                                                                                xml_dir, mclass, mpos)
        if no_label_img != None:
                no_label_img_list.append(no_label_img)
    if no_label_img_list != []:
        print("[WARNING] There are some picture which have no label, Please remove them:", \
        str(no_label_img_list))




if __name__ == '__main__':
    main()
