import torch
import torch.nn as nn
import numpy as np


# 范围切割函数，不高于t+m，不低于t-m
def clip_by_tensor(t, t_min, t_max):
    t = t.float()
    results = (t >= t_min).float() * t + (t < t_min).float() * t_min
    results = (t > t_max).float() * t_max + (t <= t_max).float() * results
    return results


# 平滑标签(目的是让标签不绝对的=1或=0，不过分依赖于标注数据，若数据标注很好也可不用)
def smooth_labels(y_true, label_smoothing, num_classes):
    return y_true * (1.0 - label_smoothing) + label_smoothing/num_classes


# 均值损失
def MESLoss(pred, target):
    return  (pred - target) ** 2


# 二分类交叉熵
def BCELoss(pred,target):
    epsilon = 1e-7
    # 防止log出现问题
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output


# 实现细节以后还需仔细研究？？？？？？？？？？
def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area, min=1e-6)

    # 计算中心的差距
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)

    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    # 计算对角线距离
    enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
    ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)

    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
        b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
    alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
    ciou = ciou - alpha * v
    return ciou



class YoloLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, label_smooth=0, cuda=True):
        super(YoloLoss,self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size
        # 特征图尺寸[76,38,19]
        self.feature_length = [img_size[0]//8, img_size[0]//16, img_size[0]//32]

        self.label_smooth = label_smooth

        self.ignore_threshold = 0.7
        # 损失函数影响因子
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.lambda_loc = 1.0
        self.cuda = cuda


    def get_target(self, target, anchors, in_w, in_h):
        bs = len(target)

        # 获得先验框
        anchor_index = [[0,1,2],[3,4,5],[6,7,8]][self.feature_length.index(in_w)]
        # 相对位置
        subtract_index = [0,3,6][self.feature_length.index(in_w)]

        # 掩码初始化
        mask = torch.zeros(bs, int(self.num_anchors/3), in_w, in_h, requires_grad=False)
        noobj_mask = torch.ones(bs, int(self.num_anchors/3), in_w, in_h, requires_grad=False)
        tx = torch.zeros(bs, int(self.num_anchors/3), in_w, in_h, requires_grad=False)
        ty =torch.zeros(bs, int(self.num_anchors/3), in_w, in_h, requires_grad=False)
        tw = torch.zeros(bs, int(self.num_anchors/3), in_w, in_h, requires_grad=False)
        th =torch.zeros(bs, int(self.num_anchors/3), in_w, in_h, requires_grad=False)
        t_box = torch.zeros(bs, int(self.num_anchors/3), in_w, in_h, 4, requires_grad=False)
        tconf = torch.zeros(bs, int(self.num_anchors/3), in_w, in_h, requires_grad=False)
        tcls = torch.zeros(bs, int(self.num_anchors/3), in_w, in_h, self.num_classes, requires_grad=False)

        box_loss_scale_x = torch.zeros(bs, int(self.num_anchors/3), in_w, in_h, requires_grad=False)
        box_loss_scale_y = torch.zeros(bs, int(self.num_anchors/3), in_w, in_h, requires_grad=False)

        for b in range(bs):
            for t in range(target[b].shape[0]):
                # 将gt的whxy换算成网格为单位的数值（原本相对于1的）
                gx = target[b][t, 0]*in_w
                gy = target[b][t, 1]*in_h
                gh = target[b][t, 2]*in_w
                gw = target[b][t, 3]*in_h

                # 计算网格位置
                gi = int(gx)
                gj = int(gy)

                # 计算IOU
                # 先将gt_box和anchors移动到0坐标处
                gt_box = torch.FloatTensor(np.array([0,0,gw,gh])).unsqueeze(0)
                anchor_shapes = torch.FloatTensor(np.concatenate(np.zeros(self.num_anchors, 2), np.array(anchors),1))
                anchor_iou = bbox_iou(gt_box, anchor_shapes)

                # 找到对应的head
                best_iou = np.argmax(anchor_iou)
                if best_iou not in anchor_index:
                    continue

                # 填充掩码
                if (gi<in_w) and (gj<in_h):
                    # 相对位置
                    best_iou = best_iou - subtract_index
                    # 给网格赋值
                    noobj_mask[b, best_iou, gj, gi] = 0
                    mask[b, best_iou, gj, gi] = 1
                    # label的框坐标填充
                    tx[b, best_iou, gj, gi] = gx
                    ty[b, best_iou, gj, gi] = gy
                    tw[b, best_iou, gj, gi] = gw
                    th[b, best_iou, gj, gi] = gh

                    # 用于xywh的比例anchor的相对位置？？？？？？？？？？
                    box_loss_scale_x[b, best_iou, gj, gi] = target[b][t, 2]
                    box_loss_scale_y[b, best_iou, gj, gi] = target[b][t, 3]

                    # 物体置信度
                    tconf[b, best_iou, gj, gi] = 1
                    # 种类
                    tcls[b, best_iou, gj, int(target)[b][t, 4]] = 1
                else:
                    print("Step {0} out of bound".format(b))

        t_box[..., 0] = tx
        t_box[..., 1] = ty
        t_box[..., 2] = tw
        t_box[..., 3] = th

        return mask, noobj_mask, t_box, tconf, tcls, box_loss_scale_x, box_loss_scale_y


    def get_ignore(self,prediction, scaled_anchors, target, in_w, in_h, noobj_mask):
        # target=[1, m ,5], m指多少个gt框，5表示gx gy gw gh cls
        bs = len(target)

        # 索引哪一个head的anchor
        anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]][self.feature_length.index(in_w)]
        scaled_anchors = np.array(scaled_anchors)[anchor_index]

        # decode
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = torch.sigmoid(prediction[..., 2])
        h = torch.sigmoid(prediction[..., 3])

        # 获取torch的类型
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格(另一种方式，和yololayer效果一样)
        grid_x = torch.linspace(0 ,in_w-1, in_w).repeat(in_w, 1).repeat(int(bs*self.num_anchors/3), 1, 1)\
                                                                        .view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_h, 1).t().repeat(int(bs*self.num_anchors/3), 1, 1)\
                                                                        .view(y.shape).type(FloatTensor)
        # [[w0,h0],...,[w2,h2]] ———> [[w0,...,2]]和[[h0,...,h2]]???????????????????????????
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor[0])
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor[1])

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_w*in_h).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_w*in_h).view(h.shape)

        # DECODE操作
        # 构造[1,3,19,19,4]
        pred_boxes = FloatTensor(prediction[..., 4].shape)
        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h

        # 筛选负样本
        for i in range(bs):
            # [3,19,19]
            pred_boxes_for_ignore = pred_boxes[i]
            # [3*19*19,4]
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)

            if len(target[i]) > 0:
                # 0:1和0的区别??????
                gx = target[i][:, 0:1] * in_w
                gy = target[i][:, 1:2] * in_h
                gw = target[i][:, 2:3] * in_w
                gh = target[i][:, 3:4] * in_h
                gt_box = torch.FloatTensor(np.concatenate([gx, gy, gw, gh])).type(FloatTensor)
                # [m, 3 * 19 * 19]
                anch_ious = bbox_iou(gt_box, pred_boxes_for_ignore)

                # 遍历每一个框
                for t in range(target[i].shape[0]):
                    # [3*19*19,]->[3,19,19]
                    anch_iou = anch_ious.view(pred_boxes[i].size()[:3])
                    noobj_mask[i][anch_iou > self.ignore_threshold] = 0

        return noobj_mask, pred_boxes


    def forward(self, input, targets=None):
        # input:3*(5+num_classes),w,h
        # 多少张图片
        bs = input.size(0)

        in_h = input.size(2)
        in_w = input.size(3)

        # 特征图的一个特征点对应原图多少个像素
        stride_h = self.img_size[1]/in_h
        stride_w = self.img_size[0]/in_w

        # 计算先验框在特征图上的宽高
        scaled_anchors = [(a_w/stride_w, a_h/stride_h) for a_w, a_h in self.anchors]

        # 调整模型预测输出格式
        prediction = input.view(bs, int(self.num_anchors/3), self.bbox_attrs, in_h, in_w).permute(0,1,3,4,2).contiguous()

        # build_target1构建流程（填充掩码+正样本）
        
        # build_target2构建流程（填充负样本+decode）










