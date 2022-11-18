import torch
import numpy as np
import cv2 as cv
import os
from yolov4 import *
from utils.utils import *
from yolov4_layer import *
from CSPDarknet import *
# ----------------------------------------------------
# #===================================================
# 推理部分
# ===================================================#
# ----------------------------------------------------
class Inference(object):
    # 初始化模型参数，导入仪训练好的权重
    def __init__(self, **kwargs):
        self.model_path = kwargs['model_path']
        self.anchors_path = kwargs['anchors_path']
        self.classes_path = kwargs['classes_path']
        self.model_image_size = kwargs['model_image_size']
        self.confidence = kwargs['confidence']
        self.cuda = kwargs['cuda']

        self.classes_names = self.get_class()
        self.anchors = self.get_anchors()
        print(self.anchors)

        # [1.] backbone+neck
        self.net = YoloBody(3, len(self.classes_names)).eval()
        load_model_pth(self.net, self.model_path)

        # 判断是否有gpu
        if self.cuda:
            self.net = self.net.cuda()
            self.net.eval()

        print("Finished!")

        # [2.] Head
        self.yolo_decodes = []
        # anchors_masks = [[0,1,2],[3,4,5],[6,7,8]]
        anchors_masks = [[6,7,8],[3,4,5],[0,1,2]]
        stride = [32,16,8]

        for i in range(3):
            head = Yolo_Layer(anchors_masks[i], len(self.classes_names), self.anchors,
                                                        len(self.anchors)//2, stride[i]).eval()
            self.yolo_decodes.append(head)

        print("{} model anchors and classes loaded!".format(self.model_path))


    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   获得所有的先验框
    # ---------------------------------------------------#
    def get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return anchors
        # return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

    # ---------------------------------------------------#
    #   导入图片
    # ---------------------------------------------------#
    def detect_image(self, img_path):
        img = cv.imread(img_path)
        h, w, _ = img.shape
        img = cv.resize(img, (608, 608))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.float32)
        # 归一化，且通道调换为（channel，h，w）
        img = np.transpose(img/255.0, (2,0,1))
        # 增加维度（1，channel，h，w）
        img = np.asarray([img])

        with torch.no_grad():
            img = torch.from_numpy(img)
            if self.cuda:
                img = img.cuda()
            outputs = self.net(img)

        output_list = []
        for i in range(3):
            output_list.append(self.yolo_decodes[i](outputs[i]))
        output = torch.cat(output_list, dim=1)
        print(output.shape)
        # 非极大值抑制（没有编写）
        batch_detections = non_max_suppression(output, len(self.classes_names),conf_thres=self.confidence, nms_thres=0.1)

        boxes = [box.cpu().numpy() for box in batch_detections]
        print(boxes[0])
        return boxes[0]


if __name__ == "__main__":
    params = {
        "model_path": '../pth/yolo4_weights_my.pth',
        "anchors_path": '../work_dir/yolo_anchors_coco.txt',
        "classes_path": '../work_dir/coco_classes.txt',
        "model_image_size": (608, 608,3),
        "confidence": 0.4,
        "cuda": False
    }

    model = Inference(**params)
    class_names = load_class_names(params['classes_path'])
    img_path = '../data/dog.jpg'
    boxes = model.detect_image(img_path)
    # 画框
    plot_boxes_cv2(img_path, boxes, savename='../results/output3.jpg', class_names=class_names)
























