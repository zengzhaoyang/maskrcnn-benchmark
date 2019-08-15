import os

import torch
import torch.utils.data
from PIL import Image
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from maskrcnn_benchmark.structures.bounding_box import BoxList


class VGDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, split, use_difficule=False, transforms=None):
        self.root = data_dir
        self.image_set = split

        self._imgsetpath = os.path.join(self.root, "genome", "%s_real.txt")
        self._objectpath = os.path.join(self.root, "genome", "1600-400-20", "objects_vocab.txt")
        self._attributepath = os.path.join(self.root, "genome", "1600-400-20", "attributes_vocab.txt")

        self._ann_prefix = os.path.join(self.root, "genome")
        self._img_prefix = os.path.join(self.root, "vg")

        self.cls = ['__background__']
        self.class_to_ind = {'__background__': 0}
        f = open(self._objectpath)
        count = 1
        for line in f:
            names = [n.lower().strip() for n in line.split(',')]
            self.cls.append(names[0])
            for n in names:
                self.class_to_ind[n] = count
            count += 1
        f.close()

        aliaspath = open(os.path.join(self.root, "genome", "object_alias.txt"))
        for line in aliaspath:
            tmp = line.strip().split(',')
            if tmp[0] in self.class_to_ind:
                for item in tmp[1:]:
                    self.class_to_ind[item] = self.class_to_ind[tmp[0]]
                print([self.class_to_ind[i] for i in tmp])


        self.att = ['__no_attribute__']
        self.attribute_to_ind = {'__no_attribute__': 0}
        f = open(self._attributepath)
        count = 1
        for line in f:
            names = [n.lower().strip() for n in line.split(',')]
            self.att.append(names[0])
            for n in names:
                self.attribute_to_ind[n] = count
            count += 1
        f.close()

        self.transforms = transforms


        f = open(self._imgsetpath % split)
        self._annopaths = []
        self._imgpaths = []
        self._height_width = []
        #idx = 0
        #for line in f:
        #    if idx % 1000 == 1:
        #        print('load', idx)
        #    idx += 1
        #    tmp = line.strip().split()
        #    filename = self._ann_prefix + '/' + tmp[1]
        #    if os.path.exists(filename):
        #        tree = ET.parse(filename)
        #        size = tree.find("size")
        #        height = int(size.find("height").text)
        #        width = int(size.find("width").text)

        #        for obj in tree.findall('object'):
        #            obj_name = obj.find('name').text.lower().strip()
        #            if obj_name in self.class_to_ind:
        #                self._annopaths.append(tmp[1])
        #                self._imgpaths.append(tmp[0])
        #                self._height_width.append((height, width))
        #                break

        for line in f:
            tmp = line.strip().split()
            self._annopaths.append(tmp[1])
            self._imgpaths.append(tmp[0])
            self._height_width.append((int(tmp[2]), int(tmp[3])))
        f.close()

        #f2 = open('objects365_real.txt', 'w')
        #for i in range(len(self._annopaths)):
        #    f2.write('%s %s %d %d\n'%(self._imgpaths[i], self._annopaths[i], self._height_width[i][0], self._height_width[i][1]))
        #f2.close()

    def __getitem__(self, index):
        img = Image.open(self._img_prefix + '/' + self._imgpaths[index]).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self._imgpaths)

    def get_groundtruth(self, index):
        anno = ET.parse(self._ann_prefix + '/' + self._annopaths[index]).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("attr_labels", anno["attr_labels"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        gt_attributes = []

        size = target.find("size")
        height = int(size.find("height").text)
        width = int(size.find("width").text)
        im_info = (height, width)

        for obj in target.iter("object"):
            name = obj.find("name").text.lower().strip()
            if name in self.class_to_ind:
                bb = obj.find("bndbox")
                bndbox = [
                    max(0, float(bb.find("xmin").text)),
                    max(0, float(bb.find("ymin").text)),
                    min(width-1, float(bb.find("xmax").text)),
                    min(height-1, float(bb.find("ymax").text)),
                ]

                if bndbox[2] < bndbox[0] or bndbox[3] < bndbox[1]:
                    bndbox = [0, 0, width-1, height-1]

                boxes.append(bndbox)
                gt_classes.append(self.class_to_ind[name])

                atts = obj.findall("attribute")
                n = 0
                gt_att = [0 for _ in range(401)]
                for att in atts:
                    att = att.text.lower().strip()
                    if att in self.attribute_to_ind:
                        gt_att[self.attribute_to_ind[att]] = 1
                        n += 1
                    if n >= 16:
                        break
                gt_attributes.append(gt_att)

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "attr_labels": torch.tensor(gt_attributes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        h, w = self._height_width[index]
        return {"height": h, "width": w}

    def map_class_id_to_class_name(self, class_id):
        return self.cls[class_id]
