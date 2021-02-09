from lxml import etree
from pathlib import Path
import re
import cv2

class Gen_Annotations:
    def __init__(self, json_info):
        self.root = etree.Element("annotation")

        child1 = etree.SubElement(self.root, "folder")
        child1.text = str(json_info["pic_dirname"])

        child2 = etree.SubElement(self.root, "filename")
        child2.text = str(json_info["filename"])

        child3 = etree.SubElement(self.root, "path")
        # print(json_info["pic_path"], type(json_info["pic_path"]))
        child3.text = str(json_info["pic_path"])

        child4 = etree.SubElement(self.root, "source")
        # child2.set("database", "The VOC2007 Database")
        child5 = etree.SubElement(child4, "database")
        child5.text = "The VOC2007 Database"

    def set_size(self, witdh, height, channel):
        size = etree.SubElement(self.root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(witdh)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        channeln = etree.SubElement(size, "depth")
        channeln.text = str(channel)
        segmented = etree.SubElement(self.root, "segmented")
        segmented.text = "0"

    def savefile(self, filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')

    def add_pic_attr(self, label, x0, y0, x1, y1):
        object = etree.SubElement(self.root, "object")
        namen = etree.SubElement(object, "name")
        namen.text = label
        pose = etree.SubElement(object, "pose")
        pose.text = "Unspecified"
        truncated = etree.SubElement(object, "truncated")
        truncated.text = "0"
        difficult = etree.SubElement(object, "difficult")
        difficult.text = "0"
        bndbox = etree.SubElement(object, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(x0)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(y0)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(x1)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(y1)


def annotation_single_img(pic_dir, pic_name,xml_dir,
                                                   tclass, tpos):
    # tclass=["head", "head", "person"]
    # tpos = [(10,11,10,12), (13,14,10,12),(103,11,102,12)]
    # pic_dirpath = r"D:\2WORK\Project\Self_Label\semi_auto_label\voc\JPEG"
    # pic_dirname="JPEG"
    # pic_name = "2.jpg"
    # # w,h, c
    # pic_size = (416,416,3)
    # xml_path=r"D:\2WORK\Project\Self_Label\semi_auto_label\voc\ann\2.xml"

    print("**" * 20)
    print("XML LABEL MODEL ")
    print("**" * 20)
    # full path for pic
    pic_fullpath = Path(pic_dir).joinpath(pic_name)
    # no suffix(.jgp,png) for pic
    pic_boldname = Path(pic_fullpath).stem
    # pic dir name
    pic_dirname=Path(pic_fullpath).parts[-2]

    img = cv2.imread(str(pic_fullpath))
    pic_size = img.shape

    json_info = {}
    json_info["pic_dirname"]=pic_dirname
    json_info["pic_path"] = pic_fullpath
    json_info["filename"] = pic_name

    anno= Gen_Annotations(json_info)
    anno.set_size(pic_size[1], pic_size[0],pic_size[2],)
    label_flag = False
    if tpos:
        for item in  range(len(tclass)):
            label =  tclass[item]
            (x0, y0, x1, y1) = tpos[item]
            anno.add_pic_attr( label, x0, y0, x1, y1)
            label_flag=True
    if label_flag:
        xml_path = Path(xml_dir).joinpath(pic_boldname+".xml")
        anno.savefile(str(xml_path))
        return None
    return pic_name


if __name__ == '__main__':
    pass

