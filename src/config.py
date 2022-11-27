from easydict import EasyDict

# 以属性的方式访问字典的值
Cfg = EasyDict()

# 一张图片占显存2.5个g左右
# batch之前在积累梯度，之后才更新权重
Cfg.batch = 64
# 划分batch
Cfg.subdivisions = 8
# 可以改尺寸，尺寸大精度高，计算量大，反则反之
Cfg.width = 608
Cfg.height = 608

# 以下参数尽量不动，影响不大
# ---------------------
# 动量参数，权重衰减正则项
Cfg.momentum = 0.949
Cfg.decay = 0.0005
# 数据增强
Cfg.angle = 0
Cfg.saturation = 1.5
Cfg.exposure = 1.5
Cfg.hue = .1
Cfg.jitter = 0.3
Cfg.mosaic = True
# ----------------------

# 学习率（策略）
# 若数据集和初始权重差异小，可适当调小，否则就用0.001
Cfg.learning_rate = 0.001
# 保护作用
Cfg.burn_in = 1000
# 类别较多，可是适当调大
Cfg.max_batches = 8000
# 缩放10倍的点，多类别也可尝试提升
Cfg.steps = [4000, 6000]
# 不动
# ---------------------
Cfg.policy = Cfg.steps
Cfg.scales = .1, .1
Cfg.classes = 2
Cfg.track = 0
Cfg.w = Cfg.width
Cfg.h = Cfg.height
# ---------------------

# 是否使用余弦退火等
Cfg.cosine_lr = False
# 可以打开，防止漏标或错标，防止过拟合
Cfg.smooth_label = False
Cfg.TRAIN_OPTIMIZER = 'adam'