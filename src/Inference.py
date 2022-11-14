import torch
from yolov4 import *

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
        self.anchors = self.get_anchor()
        print(self.anchors)

        # [1.] backbone+neck
        self.net = YoloBody(3, len(self.classes_names)).eval()
        self.load_model_pth(self.net, self.model_path)

        # 判断是否有gpu
        if self.cuda:
            self.net = self.net.cuda()
            self.net.eval()

        print("Finished!")

        # [2.] Head
        self.yolo_decodes = []
        anchors_masks = [[6,7,8],[3,4,5],[0,1,2]]
        stride = [32,16,8]

        for i in range(3):
            head = YoloLayer(anchors_masks[i], len(self.classes_names), self.anchors,
                                                        len(self.anchors)//2, stride[i]).eval()
            self.yolo_decodes.append(head)

        print("{} model anchors and classes loaded!".format(self.model_path))





if __name__ == "__main__":
    params = {
        "model_path": '../pth/yolo4_weights_my.pth',
        "anchors_path": '../work_dir/yolo_anchors_coco.txt',
        "classes_path": '../work_dir/coco_classes.txt',
        "model_image_size": (608, 608,3),
        "confidence": 0.4,
        "cuda": True
    }






















