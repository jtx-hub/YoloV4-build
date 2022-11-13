import torch.nn as nn


def yolo_decode(output):
    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()

        

# ----------------------------------------------------
# #===================================================
# YoloHead
# ===================================================#
# ----------------------------------------------------
class Yolo_Layer(nn.Module):
    """
    anchor_mask: 不同网格大小anchor的索引
    anchor: 先验框
    stride: 原图上网格的宽高尺寸
    scale_x_y: 缩放因子，默认为1
    """
    def __init__(self, anchor_mask=[], num_classes=80, anchors=[], num_anchors=9, stride=32, scale_x_y=1):
        super(Yolo_Layer, self).__init__()
        # 注意：现在以[6, 7, 8]为例
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchor = num_anchors
        # 表示anchors中每组数是一对
        self.anchor_step = len(anchors) // num_anchors
        self.stride = stride
        self.scale_x_y = scale_x_y

    def foward(self, output):
        if self.training:
            return output

        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step: (m+1) * self.anchor_step]

        # anchors以像素为单位表示原图先验框的宽高，现在换算成在网格里的宽高（anchor/stride）
        masked_anchors = [anchor/self.stride for anchor in masked_anchors]

        # decode
        # output->(B, A*n_ch, H, W)->(1, 3*(5+80), 19, 19)
        data = yolo_decode(self, output, self.num_classes, masked_anchors, len(self.anchor_mask), scale_x_y=self.scale_x_y)

        return data