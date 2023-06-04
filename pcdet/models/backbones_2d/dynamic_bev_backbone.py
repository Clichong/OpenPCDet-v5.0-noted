import numpy as np
import torch
import torch.nn as nn

# 构建基本的卷积层
class CBS(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act='SiLU'):  # ch_in, ch_out, kernel, stride, padding, groups
        super(CBS, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.01)       # 参数保留
        self.act = nn.SiLU() if act == 'SiLU' else nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# 采样yolov7的下采样方式，同时采样卷积下采样以及最大池化下采样
class DownSampleMP(nn.Module):
    def __init__(self, c_in, c_out, k, s, act='SiLU'):
        super(DownSampleMP, self).__init__()
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)     # maxpool下采样
        self.cbs1 = CBS(c_in, c_out // 2, 1, 1, act=act)

        self.cbs2 = CBS(c_in, c_out // 2, k, s, p=1, act=act)        # 卷积下采样
        self.cbs3 = CBS(c_out // 2, c_out // 2, 1, 1, act=act)

    def forward(self, x):
        p1 = self.cbs1(self.mp(x))
        p2 = self.cbs3(self.cbs2(x))
        return torch.cat([p1, p2], dim=1)   # 两个分支进行拼接处理


# SplitAttention注意力机制
class SplitAttention(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.k = k
        self.mlp1 = nn.Linear(c, c, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(c, c * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        b, k, c, h, w = x.shape  # (b, k, c, h, w): (8, 3, 64, 246, 216 )
        x = x.reshape(b, k, c, -1)                  # b,k,c,n
        a = torch.sum(torch.sum(x, 1), -1)          # b,c
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  # b,kc
        hat_a = hat_a.reshape(b, self.k, c)         # b,k,c
        bar_a = self.softmax(hat_a)                 # b,k,c
        attention = bar_a.unsqueeze(-1)             # b,k,c,1
        out = attention * x                         # b,k,c,n
        out = torch.sum(out, 1).reshape(b, c, h, w) # b,c,n -> b,c,h,w
        return out


# 动态模块的核心部分，设计为即插即用
class DynamicMultiscaleModule(nn.Module):
    def __init__(self, n, weight=True, op_mode='cat', c_in=128):
        super(DynamicMultiscaleModule, self).__init__()
        self.weight = weight        # 默认开启权重的重分布
        self.iter = range(n)        # iter object
        self.op_mode = op_mode      # 聚合方式
        # 只能取限定模式模式
        assert op_mode in ['cat', 'add', 'max', 'att'], \
            "{} Choose Error. Just ['cat', 'add', 'max', 'att'] can choose.".format(op_mode)

        # 针对n层特征层构建n-1个可学习参数
        if weight:
            self.w = nn.Parameter(torch.arange(1., n+1) / n, requires_grad=True)  # 这里可以将除以2换成除以3

        if self.op_mode == 'att':
            self.split_attention = SplitAttention(c_in, k=3)    # 分为3组

    def forward(self, x):
        """
        x: list3 [[B,C,Y,X], [B,C,Y,X], [B,C,Y,X]]
        """
        # 1)确定是否使用权重
        if self.weight:
            w = torch.sigmoid(self.w)   # * 2 取消乘2，将权重控制在1以内
            y = [x[i] * w[i] for i in self.iter]
        else:
            y = [x[i] for i in self.iter]

        # 2)确定是否进行拼接还是直接相加
        if self.op_mode == 'cat':       # 特征直接拼接
            y = torch.cat(y, dim=1)
        elif self.op_mode == 'add':
            y = torch.stack(y, dim=1).sum(dim=1)      # 特征相加聚合
        elif self.op_mode == 'max':
            y = torch.stack(y, dim=1).max(dim=1)[0]   # 特征取最值聚合
        elif self.op_mode == 'att':
            y = torch.stack(y, dim=1)     # 在channels维度进行拼接
            y = self.split_attention(y)   # 是否启用split attention
        return y


# 功能：PointPillar的Backbone_2d部分具体网络实现
class DynamicBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        """
        Args:
            model_cfg:        BACKBONE_2D部分的配置参数
                NAME: BaseBEVBackbone
                LAYER_NUMS: [3, 5, 5]
                LAYER_STRIDES: [2, 2, 2]
                NUM_FILTERS: [64, 128, 256]
                UPSAMPLE_STRIDES: [1, 2, 4]
                NUM_UPSAMPLE_FILTERS: [128, 128, 128]
            input_channels:   map2bev模块的输出特征(64)
        """
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS          # [3,5,5]
            layer_strides = self.model_cfg.LAYER_STRIDES    # [2,2,2]
            num_filters = self.model_cfg.NUM_FILTERS        # [64,128,256]
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS  # [128,128,128]
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES          # [1,2,4]
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)    # 3
        c_in_list = [input_channels, *num_filters[:-1]]    # in:[64, 64, 128]  out:[64, 128, 256]

        self.blocks = nn.ModuleList()       # 存储下采样特征提取模块
        self.deblocks = nn.ModuleList()     # 存储上采样反卷积模块
        for idx in range(num_levels):
            # 特征下采样的第一层将尺度减半
            cur_layers = [
                # DownSampleMP(c_in_list[idx], num_filters[idx], k=3, s=layer_strides[idx]),
                nn.ZeroPad2d(1),  # 对上下左右进行填补0
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False    # 第一层的stride=2，进行下采样
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ] if self.model_cfg.get('USE_NORM_DOWNSAMPLE', True) else [
                DownSampleMP(c_in_list[idx], num_filters[idx], k=3, s=layer_strides[idx],
                             act=self.model_cfg.get('MP_ACTIVATION', 'SiLU')),  # 进行MP结构下采样, 默认不启用
            ]

            # 重复堆叠多层卷积层
            for k in range(layer_nums[idx]):    # 每层的重复次数
                cur_layers.append(
                    CBS(num_filters[idx], num_filters[idx], k=3, s=1, p=1,
                        act=self.model_cfg.get('MP_ACTIVATION', 'SiLU'))
                )
                # cur_layers.extend([
                #     nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False), # 其余层保持维度不变
                #     nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                #     nn.ReLU()
                # ])
            self.blocks.append(nn.Sequential(*cur_layers))

            # 上采样操作：将每个尺度的特征输出统一卷积到128的维度（64->128; 128->128; 256->128）
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]  # upsample_strides: [1,2,4]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(     # 反卷积，对特征图尺寸进行上采样操作
                            num_filters[idx],
                            num_upsample_filters[idx],
                            upsample_strides[idx],     # 卷积核大小和步长大小是一致的，相当于是进行patch卷积
                            stride=upsample_strides[idx],
                            bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.SiLU() if self.model_cfg.get('MP_ACTIVATION', 'SiLU') == 'SiLU' else nn.ReLU()
                    ))

        # 动态多尺度模块的初始化，这里需要确定动态模块的多尺度聚合方式以及确定是否使用SplitAttention
        dynamic_mode = self.model_cfg.get('MODE_CHOOSE', 'cat')    # 默认是进行直接拼接的操作
        self.dym = DynamicMultiscaleModule(n=num_levels, op_mode=dynamic_mode, c_in=num_upsample_filters[0])

        # 统计上采样后的总特征维度,只有concat是需要统计全部
        c_in = sum(num_upsample_filters) if dynamic_mode == 'cat' else sum(num_upsample_filters) // num_levels
        self.num_bev_features = c_in    # 处理后的bev特征384，存储在dict中在后续传入dense_head来构建分类头和回归头

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']    # 开始的特征: (16, 64, 496, 432)
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)   # 下采样操作
            stride = int(spatial_features.shape[2] / x.shape[2])    # 采样倍率依次为:2/4/8

            # 三层下采样后的维度分别为: (16, 64, 246, 216) / (16, 128, 124, 108) / (16, 256, 62, 54)
            ret_dict['spatial_features_%dx' % stride] = x   # 保留下采样结果
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))     # 同步channels操作(上采样操作) 不同尺度统一成 (16, 128, 246, 216)
            else:
                ups.append(x)

        # 这里是对上采样的尺度进行了拼接操作，打算用三个尺度的特征层进行融合来处理
        if len(ups) > 1:
            # x = torch.cat(ups, dim=1)   # 在第2个维度拼接在一起 (16, 384, 246, 216)
            x = self.dym(ups)           # 采用动态多尺度权重模块
        elif len(ups) == 1:
            x = ups[0]

        # if len(self.deblocks) > len(self.blocks):   # False
        #     x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x    # backbone_2d提取的特征

        return data_dict


if __name__ == '__main__':

    # mp = DownSampleMP(64, 64, 3, 2)
    #
    # x = torch.rand([16, 64, 246, 216])
    # print(mp(x).shape)

    plot = SplitAttention(64, k=3)
    x = torch.rand([16, 3, 64, 246, 216])
    print(plot(x).shape)
