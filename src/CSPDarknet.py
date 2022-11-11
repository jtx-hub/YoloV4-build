import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
#===================================================
CBM模块:CONV+BN+MISH
===================================================# 
'''
<<<<<<< HEAD
=======


>>>>>>> f6c31e8ee2c79f6c7901bf2677e430a7cd056cfd
# Mish激活函数
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
        # return x * torch.tanh(torch.log(1 + torch.exp(x)))


# 卷积块
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()
        # yolov4的卷积核尺寸就1和3，所以1时padding=0，2时padding=1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


'''
#===================================================
CSP模块（CSP1和CSP2结构有区别！）
===================================================# 
'''
<<<<<<< HEAD
=======


>>>>>>> f6c31e8ee2c79f6c7901bf2677e430a7cd056cfd
# 残差模块
class ResBlock(nn.Module):
    def __init__(self, channels, hidden_channels = None):
        super(ResBlock, self).__init__()

        # 降维参数选择
        if hidden_channels == None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)


# CSPNet模块
'''
num_blocks: CSP模块数量
first: 是否第一个部分的CSP
'''
<<<<<<< HEAD
=======


>>>>>>> f6c31e8ee2c79f6c7901bf2677e430a7cd056cfd
class ResBlock_Body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(ResBlock_Body, self).__init__()

        # 下采样
        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)
        if first:
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
<<<<<<< HEAD
                ResBlock(channels=out_channels, hidden_channels=out_channels // 2),
=======
                ResBlock(channels=out_channels, hidden_channels=out_channels//2),
>>>>>>> f6c31e8ee2c79f6c7901bf2677e430a7cd056cfd
                BasicConv(out_channels, out_channels, 1)
            )
            self.concat_conv = BasicConv(out_channels*2, out_channels, 1)
        else:
            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)
            self.blocks_conv = nn.Sequential(
<<<<<<< HEAD
                # *列表解耦成参数
                *[ResBlock(channels=out_channels//2) for _ in range(num_blocks)],
=======
                *[ResBlock(channels=out_channels, hidden_channels=out_channels // 2)],
>>>>>>> f6c31e8ee2c79f6c7901bf2677e430a7cd056cfd
                BasicConv(out_channels//2, out_channels//2, 1)
            )
            self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)


'''
----------------------------------------------------
#===================================================
BackBone结构
CSP[1,2,8,8,4]
CSPX = CBM + CSP + CBM(ResBlock_Body就已经写好了)
===================================================# 
----------------------------------------------------
'''
class CSPDarkNet(nn.Module):
    def __init__(self, layers):
        super(CSPDarkNet, self).__init__()
        # 输入通道数
        self.inplanes = 32
        self.conv1 = BasicConv(3, self.inplanes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]
        self.stages = nn.ModuleList([
            ResBlock_Body(self.inplanes, self.feature_channels[0], layers[0], first=True),
            ResBlock_Body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            ResBlock_Body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            ResBlock_Body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            ResBlock_Body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False),
        ])

        # 初始化
        for m in self.modules():
            # 卷积初始化
            if isinstance(m, nn.Conv2d):
                # kernel宽高*通道数
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                # 正态
                m.weight.data.normal_(0, math.sqrt(2./n))
            # BN参数初始化
            elif isinstance(m, nn.BatchNorm2d):
                # 伽马
                m.weight.data.fill_(1)
                # 贝塔
                m.bias.data.zero_()

    # 前向传播
    def foward(self, x):
        x = self.conv1(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)

        return out3, out4, out5


'''
#===================================================
BackBone结构+初始化预训练权重
===================================================# 
'''
def load_model_pth(model, pth):
    print("Load weights into state dict, name: %s" % (pth))
    devcie = torch.device("gpu" if torch.cuda.is_available() else "cpu")
    # 加载自建的初始化模型为字典格式
    model_dict = model.state_dict()
    # 加载预训练权重字典
    pretrained_dict = torch.load(pth, map_location=devcie)
    # 匹配
    matched_dict = {}
    for k, v in model_dict.items():
        if k.find("backbone") == -1:
            key = "backbone." + k
            if np.shape(pretrained_dict[key]) == np.shape(v):
                matched_dict[k] = v

    for key in matched_dict:
        print('pretrained items: ', key)

    # 打印匹配情况
    print("%d layers matched, %d layers miss"%(len(matched_dict.keys()), len(model_dict)-len(matched_dict.keys())))
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)
    return model


def darknet53(pretrained):
    model = CSPDarkNet([1, 2, 8, 8, 4])
    load_model_pth(model, pretrained)


'''
#===================================================
Main函数
===================================================# 
'''
if __name__ == "__main__":
    # CoCo数据集的预训练权重
    coco_weights_path = '../pth/yolo4_weights_my.pth'
    backbone = darknet53(coco_weights_path)



