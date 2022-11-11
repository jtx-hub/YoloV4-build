import torch
import torch.nn as nn
from collections import OrderedDict

from CSPDarknet import *


'''
#===================================================
CBL模块:CONV+BN+LeakyRelu
===================================================# 
'''
def conv2d(filter_in, filter_out, kernek_size, stride=1):
    padding = (kernek_size-1//2 if kernek_size else 0 )
    return nn.Sequential(OrderedDict([
                             ('conv', nn.Conv2d(filter_in, filter_out, kernek_size, stride, padding)),
                             ('bn', nn.BatchNorm2d(filter_out)),
                             ('relu', nn.LeakyReLU(0.1))
                         ]))


'''
#===================================================
SPP模块:不同尺寸的池化后堆叠
===================================================# 
'''
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5,9,13]):
        super(SpatialPyramidPooling, self).__init__()
        self.maxpools = nn.ModuleList([nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size//2) for pool_size in pool_sizes])

    def foward(self, x):
        features = [maxpool(x) for maxpool in maxpools[::-1]]
        features = torch.cat(features+[x], dim=1)

        return features


'''
#===================================================
卷积 + 上采样
===================================================# 
'''
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            # 拉伸原来两倍
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def foward(self, x):
        x = self.upsample(x)
        return x


'''
#===================================================
三次卷积块
[512, 1024]
===================================================# 
'''
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], kernek_size=1),
        conv2d(filters_list[0], filters_list[1], kernek_size=3),
        conv2d(filters_list[1], filters_list[0], kernek_size=1)
    )
    return m


'''
#===================================================
五次卷积块
===================================================# 
'''
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], kernek_size=1),
        conv2d(filters_list[0], filters_list[1], kernek_size=3),
        conv2d(filters_list[1], filters_list[0], kernek_size=1),
        conv2d(filters_list[0], filters_list[1], kernek_size=3),
        conv2d(filters_list[1], filters_list[0], kernek_size=1)
    )
    return m


'''
#===================================================
yolo最后的输出
===================================================# 
'''
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], kernek_size=3),
        nn.Conv2d(filters_list[0], filters_list[1], 1)
    )
    return m


'''
----------------------------------------------------
#===================================================
YoloBody：
    backbone + neck
===================================================# 
----------------------------------------------------
'''
class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        # backbone
        self.backbone = darknet53(None)

        # neck(五个部分)
        # 1
        self.conv1 = make_three_conv([512, 1024] ,1024)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512, 1024] ,2048)

        # 2
        self.upsample1 = Upsample(512, 256)
        self.conv_fo_P4 = conv2d(512, 256 ,1)
        self.make_five_conv = make_five_conv([256, 512] ,512)

        # 3
        self.upsample2 = Upsample(256, 128)
        self.conv_fo_P3 = conv2d(512, 256, 1)
        self.make_five_conv2 = make_five_conv([128, 256], 256)
        # neck最后的输出
        final_out_filter2 = num_anchors*(5+num_classes)
        self.yolo_head3 = yolo_head([256, final_out_filter2], 128)

        # 4
        # 这里cbl包含一次下采样
        self.dowmsample1 = conv2d(128, 256, 3, stride=2)
        self.make_five_conv3 = make_five_conv([256, 512] ,512)
        self.yolo_head2 = yolo_head([512, final_out_filter2], 256)

        # 5
        self.dowmsample2 = conv2d(256, 512, 3, stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024], 1024)
        self.yolo_head1 = yolo_head([1024, final_out_filter2], 512)

    def foward(self, x):
        x2, x1, x0 = backbone(x)
        # 1
        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        P5 = self.conv2(P5)
        # 2
        P5_upsample = self.upsample1(P5)
        P4 = self.conv_fo_P4(x1)
        P4 = torch.cat([P4, P5_upsample], dim=1)
        P4 = self.make_five_conv(P4)
        # 3
        P4_upsample = self.upsample2(P4)
        P3 = self.conv_fo_P3(x2)
        P3 = torch.cat([P3, P4_upsample], dim=1)
        P3 = self.make_five_conv2(P3)
        # 4
        P3_downsample = self.dowmsample1(P3)
        P4 = torch.cat([P4, P3_downsample], dim=1)
        P4 = self.make_five_conv3(P4)
        # 5
        P4_downsample = self.dowmsample2(P4)
        P5 = torch.cat([P5, P4_downsample], dim=1)
        P5 = self.make_five_conv3(P5)

        out2 = self.yolo_head3(P3)
        out1 = self.yolo_head2(P4)
        out0 = self.yolo_head1(P5)

        return out0, out1, out2


if __name__ == "__main__":
    coco_weights_path = 'D:/Python_Myscript/Git/Yolov4-build/pth/yolo4_weights_my.pth'
    model = YoloBody(3, 80)
    load_model_pth(model, coco_weights_path)




