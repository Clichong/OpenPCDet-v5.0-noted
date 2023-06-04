import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate

# 功能：构建PointPillar的dense head模块部分
class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        """
        Args:
            model_cfg:      DENSE_HEAD的配置文件
            input_channels: backbone_2d三层特征输出的总和(128 + 128 + 128 = 384)
            num_class:      类别数目(3类)
            class_names:    类别名称: ['Car', 'Pedestrian', 'Cyclist']
            grid_size:      网格大小
            point_cloud_range:  点云范围：[-x, -y, -z, x, y, z]
            predict_boxes_when_training:    布尔变量：False(因为pointpillar是oneStage模型，没有roi head)
            voxel_size:     设置的体素大小
        """
        super().__init__(   # 基类没有传参input_channels和voxel_size
            model_cfg=model_cfg, num_class=num_class,
            class_names=class_names, grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)  # 6 每个点的anchor数量(3类 * 2方向 = 6种anchor)
        # 384 -> 6*3 -> 6种anchor * 3个类别
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        # 384 -> 6*7 -> 6种anchor * 每个anchor的7个信息表示[x, y, z, dx, dy, dz, heading]
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )
        # 384 -> 6*2 -> 6种anchor * 两个方向
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:    # 方向损失(设置方向卷积层)
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,    # 6*2
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()  # 参数初始化

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))   # 初始化分类的bias
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)        # 初始化回归的weight

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']  # (16, 384, 246, 216)

        # 对特征分别进行cls、box、dir预测，将预测结果存储在forward_ret_dict字典中，来进行后续损失的构建
        # 1.cls/box信息预测
        cls_preds = self.conv_cls(spatial_features_2d)  # 每个anchor的类别预测 (16, 18, 246, 216)
        box_preds = self.conv_box(spatial_features_2d)  # 每个anchor的回归预测dim=7 (16, 42, 246, 216)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds   # 保存所有head的预测结构，进行之后的损失计算
        self.forward_ret_dict['box_preds'] = box_preds

        # 2.dir信息预测
        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)  # 每个anchor的方向 (16, 12, 246, 216)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        # 训练过程
        if self.training:
            targets_dict = self.assign_targets(     # 获取gt信息
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)   # 此时记录gt信息以及预测信息,来进行后续的loss计算

        # 测试过程
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(   # 根据各类预测矩阵生成预测box
                batch_size=data_dict['batch_size'],     # 设置的batch_size为16
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds   # 各预测特征map
            )
            data_dict['batch_cls_preds'] = batch_cls_preds  # (16, 321408, 7)
            data_dict['batch_box_preds'] = batch_box_preds  # (16, 321408, 3)
            data_dict['cls_preds_normalized'] = False

        return data_dict    # 返回更新后的data_dict
