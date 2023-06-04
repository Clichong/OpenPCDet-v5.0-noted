import numpy as np
import torch
import torch.nn as nn

from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner
from .target_assigner.axis_aligned_target_assigner_cornor_loss import AxisAlignedTargetAssignerCornorLoss

# 功能：dense head模块的基类
class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        """
        Args:
            model_cfg:      DENSE_HEAD的配置文件
            num_class:      类别数目(3类)
            class_names:    类别名称: ['Car', 'Pedestrian', 'Cyclist']
            grid_size:      网格大小
            point_cloud_range:  点云范围：[-x, -y, -z, x, y, z]
            predict_boxes_when_training:    布尔变量：False (twoStage模型才会设置为True)
        """
        super().__init__()    # 初始化nn.Module
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training  # False (twoStage模型才会设置为True)
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False) # False (多尺度head的设置)

        # Dense Head模块包含三大子部分：
        # 1)对生成的anchor和gt进行编码和解码
        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG   # anchor分配文件
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)( # 在box_coder_utils文件中调用ResidualCoder类
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),  # 如果没有设置，默认为6
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        # 2)anchor生成配置
        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG   # list:存储每个类别的anchor生成设置
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )

        # 3)gt匹配
        self.anchors = [x.cuda() for x in anchors]  # 放在GPU上
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        # 4)保存前向传播结果
        self.forward_ret_dict = {}    # 根据forward_ret_dict内容来计算loss
        self.build_losses(self.model_cfg.LOSS_CONFIG)       # 分类损失、回归损失、方向损失的构建

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        """
        Args:
            anchor_generator_cfg:   每个类别的anchor配置
            grid_size:              网格大小 [432 496   1]
            point_cloud_range:      点云范围 [  0.   -39.68  -3.    69.12  39.68   1.  ]
            anchor_ndim:            anchor维度: 7 位置 + 大小 + 方向 [x,y,z,dx,dy,dz,rot]
        """
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )

        # 对每个类别生成anchor的feature map [array([216, 248]), array([216, 248]), array([216, 248])]
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]   # 这里的feature size和前向传播处理的特征矩阵尺寸是一样的
        # 返回每个类别构建好的anchor[(1,248,216,1,2,7), ...] 和 每个位置anchor的数量[2, 2, 2]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:    # 默认情况是为7, 如果anchor的维度不等于7，则补0
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,       # dense_head配置
                class_names=self.class_names,   # 类别名称
                box_coder=self.box_coder,       # 编码方式
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssignerCornorLoss':
            target_assigner = AxisAlignedTargetAssignerCornorLoss(
                model_cfg=self.model_cfg,  # dense_head配置
                class_names=self.class_names,  # 类别名称
                box_coder=self.box_coder,  # 编码方式
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        # 添加loss模块，包括分类损失，回归损失和方向损失并初始化(可自行选择设置)
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE    # 如果没有指定则默认SmoothL1
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])    # 各回归指标权重
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes      # anchors-->list:3 [(1，248，216，1，2，7），(1，248，216，1，2，7），(1，248，216，1，2，7)]
        )
        return targets_dict

    def get_cls_layer_loss(self):
        """
        构建分类损失
        """
        cls_preds = self.forward_ret_dict['cls_preds']      # (16, 248, 216, 18) 网络类别预测
        box_cls_labels = self.forward_ret_dict['box_cls_labels']    # (16,321408) 前景anchor类别
        batch_size = int(cls_preds.shape[0])    # 16
        cared = box_cls_labels >= 0         # 关心的anchor (16,321408)
        positives = box_cls_labels > 0      # 前景anchor (16,321408)
        negatives = box_cls_labels == 0     # 背景anchor (16,321408)
        negative_cls_weights = negatives * 1.0      # 背景anchor赋予权重
        cls_weights = (negative_cls_weights + 1.0 * positives).float()  # 背景 + 前景权重=分类损失权重，这里其实前景背景anchor的分类权重都设置为1 (在阈值之间的anchor会被忽略)
        reg_weights = positives.float()  # 回归损失权重
        if self.num_class == 1:
            # class agnostic 类别不可知
            box_cls_labels[positives] = 1

        # 构建正负样本的分类权重
        pos_normalizer = positives.sum(1, keepdim=True).float() # 统计每个点云帧的正样本数量（截断为1，避免无法计算weight） eg:[[162.],[166.],[155.],[108.]]
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)     # 正则化回归损失
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)     # 正则化分类损失

        # cls_targets = cls_targets.unsqueeze(dim=-1)
        # cls_targets = cls_targets.squeeze(dim=-1)

        # 构建target的独热编码
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)  # (16,321408)
        one_hot_targets = torch.zeros(      # (16,321408,4) 零矩阵
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)     # 将目标标签转换为one-hot编码形式
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)      # (16, 248, 216, 18) --> (16, 321408, 3)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds,            # (16, 321408, 3)
                                          one_hot_targets,      # (16, 321408, 3)
                                          weights=cls_weights)  # [N, M] 分类损失的计算
        cls_loss = cls_loss_src.sum() / batch_size   # 归一化操作

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        """
        针对角度添加sin损失，有效防止-pi和pi方向相反时损失过大,这里只是分别构建了sina * cosb与cosa * sinb部分
        但是sin(a - b) = sina * cosb - cosa * sinb公式中还存在相减的部分，这个相减的部分在WeightedSmoothL1Loss中与其他参数一同处理了
        以至于不需要单独对方向进行处理
        """
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])   # sina * cosb (16, 321408, 1)
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])     # cosa * sinb (16, 321408, 1)
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)   # 将sina * cosb部分替换预测的heading列 (16, 321408, 7)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)     # 将cosa * sinb替换真实编码的heading列 (16, 321408, 7)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        """
        Args:
            anchors:     (16, 321408, 7)
            reg_targets: (16, 321408, 7)
            one_hot:     True
            dir_offset:  方向偏移量 0.78539 = π/4
            num_bins:    BINS的方向数 = 2
        """
        batch_size = reg_targets.shape[0]   # 16
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])   # (16, 321408, 7)
        rot_gt = reg_targets[..., 6] + anchors[..., 6]      # gt的转向角度 为什么直接相加，因为编码reg_target的处理是直接相减的：rts = [rg - ra]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)   # 将角度限制在0到2*pi之间,来确定是否反向
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()   # 取值为0和1，num_bins=2 (16, 321408)
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)     # (16, 321408)

        if one_hot:     # 对目标构建成one-hot编码形式
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)    # (16, 321408，2)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        """
        构建回归损失以及角度预测损失
        """
        box_preds = self.forward_ret_dict['box_preds']      # 回归信息预测 (16, 248, 216, 42)
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)    # 方向信息预测 (16, 248, 216, 12)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']      # 目标回归信息 (16, 321408, 7) 这里只对前景anchor进行编码
        box_cls_labels = self.forward_ret_dict['box_cls_labels']        # 目标分类信息 (16, 321408)
        batch_size = int(box_preds.shape[0])    # 16

        positives = box_cls_labels > 0
        reg_weights = positives.float()     # 根据掩码来构建出前景权重(1)，背景权重为0
        pos_normalizer = positives.sum(1, keepdim=True).float() # 正则化
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)     # 设定一个裁剪的最小值

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)   # (1, 248, 216, 3, 2, 7)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)   # 对anchor进行重复batch_size遍 (1, 248, 216, 3, 2, 7) -> (1, 321408, 7) -> (16, 321408, 7)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])     # (16, 321408, 7)
        # sin(a - b) = sina * cosb - cosa * sinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)    # 针对角度添加sin损失，有效防止-pi和pi方向相反时损失过大
        loc_loss_src = self.reg_loss_func(box_preds_sin,
                                          reg_targets_sin,
                                          weights=reg_weights)  # 回归损失的具体计算过程 (16, 321408, 7)
        loc_loss = loc_loss_src.sum() / batch_size      # 归一化

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        # 构建分类损失
        if box_dir_cls_preds is not None:   # 方向预测是否为空
            dir_targets = self.get_direction_target(
                anchors,            # (16, 321408, 7)
                box_reg_targets,    # (16, 321408, 7)
                dir_offset=self.model_cfg.DIR_OFFSET,   # 方向偏移量 0.78539 = π/4
                num_bins=self.model_cfg.NUM_DIR_BINS    # BINS的方向数 = 2
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)    # (16, 248, 216, 12) -> (16, 321408, 2)
            weights = positives.type_as(dir_logits)    # 只对正样本预测方向 (16, 321408)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)  # (16, 321408)
            dir_loss = self.dir_loss_func(dir_logits,
                                          dir_targets,
                                          weights=weights)     # 具体方向损失计算函数
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_loss(self):
        """
        用于训练阶段：损失计算的总函数
        """
        cls_loss, tb_dict = self.get_cls_layer_loss()           # 计算分类损失
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()   # 计算回归损失
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss      # 总损失

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        用于测试阶段：基于预测矩阵解码来生成预测框
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):      # True 将anchor进行拼接
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)   # (1, 248, 216, 3, 2, 7)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]  # 3个类别+2个方向 在特征图上的总anchor数 321408
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)     # (16, 321408, 7)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds   # (16, 248, 216, 18) -> (16, 321408, 3)
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)    # (16, 248, 216, 42) -> (16, 321408, 7)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)   # 根据pred和anchor解码为正常的尺寸 (16, 321408, 7)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET      # 0.78539
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET  # 0
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)  # (16, 321408, 2)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]     # (16, 321408)

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)  # pi
            dir_rot = common_utils.limit_period(    # 限制在0到pi之间
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            # period * dir_labels.to(batch_box_preds.dtype) 如果label为1，则为π；否则仍然保存0；
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):  # False
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
