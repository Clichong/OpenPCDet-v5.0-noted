import numpy as np
import torch
import torch.nn as nn

# 功能：PointPillar的Backbone_2d部分具体网络实现
class BaseBEVBackbone(nn.Module):
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
                nn.ZeroPad2d(1),    # 对上下左右进行填补0
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False    # 第一层的stride=2，进行下采样
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]

            # 重复堆叠多层卷积层
            for k in range(layer_nums[idx]):    # 每层的重复次数
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False), # 其余层保持维度不变
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
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
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)    # 统计上采样后的总特征维度
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in    # 处理后的bev特征384

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

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)   # 在第2个维度拼接在一起 (16, 384, 246, 216)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):   # False
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x    # backbone_2d提取的特征

        return data_dict


class BaseBEVBackboneV1(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    num_filters[idx], num_filters[idx], kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['multi_scale_2d_features']

        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']

        ups = [self.deblocks[0](x_conv4)]

        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[1](x))

        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)

        data_dict['spatial_features_2d'] = x

        return data_dict
