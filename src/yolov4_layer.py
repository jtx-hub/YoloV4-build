import torch.nn as nn
import torch
import numpy as np


# output(B,A*n_ch,H,W) -> (B,A,H,W,n_ch)
def yolo_decode(output, num_classes, anchors, num_anchors, scale_x_y):
    device = None
    cuda_check = output.is_cuda
    # 选择gpu
    if cuda_check:
        device = output.get_device()

    # output顺序转换
    A = num_anchors
    n_ch = 4+1+num_classes
    B = output.size(0)
    H = output.size(2)
    W = output.size(3)
    # permute()交换顺序，contiguous()保证内存连续
    output = output.view(B, A, n_ch, H, W).permute(0, 1, 3, 4, 2).contiguous()

    # 取数
    tx, ty = output[..., 0], output[..., 1]
    tw, th = output[..., 2], output[..., 3]

    det_conf = output[..., 4]
    cls_conf = output[..., 5:]

    # 计算bx，by，bh，bw, conf, cls
    bx = torch.sigmoid(tx)
    by = torch.sigmoid(ty)
    # scale_x_y=0或1，检测目标包含大小物体=0，小物体较多=1(相当于没有)
    bw = torch.exp(tw)*scale_x_y-0.5*(scale_x_y-1)
    bh = torch.exp(th)*scale_x_y-0.5*(scale_x_y-1)
    # obj，cls也要sigmod()
    det_conf = torch.sigmoid(det_conf)
    cls_conf = torch.sigmoid(cls_conf)

    # 构造网格，网格序号表示, 例：[1,3,19,19]
    grid_x = torch.arange(W, dtype=torch.float).repeat(1,3,W,1).to(device)
    # 行列互换
    grid_y = torch.arange(H, dtype=torch.float).repeat(1,3,H,1).permute(0,1,3,2).to(device)

    bx += grid_x
    by += grid_y

    for i in range(num_anchors): # i:[0,1,2]
        # i表示anchor索引
        bw[:,i,:,:] *= anchors[i*2]
        # anchors有6个元素，两两一对儿
        bh[:,i,:,:] *= anchors[i*2+1]

    # 相对位置,增加一个维度1(1,3,19,19,1)
    bx = (bx/W).unsqueeze(-1)
    by = (by/H).unsqueeze(-1)
    bw = (bw/W).unsqueeze(-1)
    bh = (bh/H).unsqueeze(-1)

    # (b,a,h,w,1)->(b,a,h,w,4)->(b,a*h*w,4)
    boxes = torch.cat((bx,by,bw,bh),dim=-1).reshape(B, A*H*W, 4)
    det_conf = det_conf.unsqueeze(-1).reshape(B, A*H*W, 1)
    cls_conf = cls_conf.reshape(B, A*H*W, num_classes)
    # cat
    outputs = torch.cat([boxes, det_conf, cls_conf], dim=-1)

    return outputs


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

    def forward(self, output):
        if self.training:
            return output

        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step: (m+1) * self.anchor_step]

        # anchors以像素为单位表示原图先验框的宽高，现在换算成在网格里的宽高（anchor/stride）
        masked_anchors = [anchor/self.stride for anchor in masked_anchors]

        # decode
        # output->(B, A*n_ch, H, W)->(1, 3*(5+80), 19, 19)
        data = yolo_decode(output, self.num_classes, masked_anchors, len(self.anchor_mask), scale_x_y=self.scale_x_y)

        return data
