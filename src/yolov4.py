import torch
import torch.nn as nn

from CSPDarknet import darknet53


'''
#===================================================
CBL模块:CONV+BN+LeakyRelu
===================================================# 
'''
def conv2d(filter_in, filter_out, kernek_size, stride=1):
    padding = (kernek_size-1//2 if kernek_size else 0 )
    return nn.Sequential(OrderedDict[
                             ('conv', nn.Conv2d(filter_in, filter_out, kernek_size, stride, padding)),
                             ('bn', nn.BatchNorm2d(filter_out)),
                             ('relu', nn.LeakyReLU(0.1))
                         ])


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
    pass






'''
#===================================================
五次卷积块
===================================================# 
'''
def make_five_conv(filters_list, in_filters):
    pass

