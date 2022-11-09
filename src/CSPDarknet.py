import torch
import torch.nn as nn
import torch.nn.functional as F


'''
#===================================================
CBM模块:CONV+BN+MISH
====================================================# 
'''

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
CSPdarknet结构模块
====================================================# 
'''

# 残差模块
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels = None):
        super(Resblock, self).__init__()

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
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()

        # 下采样
        self.downsample = BasicConv(in_channels, out_channels, 3, stride=2)
        if first:
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels//2),
                BasicConv(out_channels, out_channels, 1)
            )
            self.concat_conv = BasicConv(out_channels*2, out_channels, 1)
        else:
            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)
            self.blocks_conv = nn.Sequential(
                *[Resblock(channels=out_channels, hidden_channels=out_channels // 2)],
                BasicConv(out_channels//2, out_channels//2, 1)
            )

