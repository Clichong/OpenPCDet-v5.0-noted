import numpy as np
import torch

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils, loss_utils

# 功能: AnchorHeadTemplate中的函数get_target_assigner调用
class AxisAlignedTargetAssigner(object):
    def __init__(self, model_cfg, class_names, box_coder, match_height=False):
        """
        Args:
            model_cfg:      dense_head配置文件
            class_names:    类别列表: ['Car', 'Pedestrian', 'Cyclist']
            box_coder:      ResidualCoder
            match_height:   False
        """
        super().__init__()

        anchor_generator_cfg = model_cfg.ANCHOR_GENERATOR_CONFIG    # anchor生成配置
        anchor_target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG        # anchor分配配置
        self.box_coder = box_coder          # pcdet.utils.box_coder_utils.ResidualCoder
        self.match_height = match_height    # False
        self.class_names = np.array(class_names)    # ['Car', 'Pedestrian', 'Cyclist']
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]   # ['Car', 'Pedestrian', 'Cyclist']
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None # None
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE    # 512
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES  # False
        self.matched_thresholds = {}    # 正样本匹配: {'Car':0.6, 'Pedestrian':0.5, 'Cyclist':0.5}
        self.unmatched_thresholds = {}  # 负样本匹配: {'Car':0.45, 'Pedestrian':0.35, 'Cyclist':0.35}
        for config in anchor_generator_cfg:
            self.matched_thresholds[config['class_name']] = config['matched_threshold']
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold']

        self.use_multihead = model_cfg.get('USE_MULTIHEAD', False)  # 没有配置MULTIHEAD

        # self.separate_multihead = model_cfg.get('SEPARATE_MULTIHEAD', False)
        # if self.seperate_multihead:
        #     rpn_head_cfgs = model_cfg.RPN_HEAD_CFGS
        #     self.gt_remapping = {}
        #     for rpn_head_cfg in rpn_head_cfgs:
        #         for idx, name in enumerate(rpn_head_cfg['HEAD_CLS_NAME']):
        #             self.gt_remapping[name] = idx + 1

    def assign_targets(self, all_anchors, gt_boxes_with_classes):
        """
        功能：处理一个batch中所有点云的anchors和gt_boxes，计算前景和背景anchor的类别，box编码和回归权重
        Args:
            all_anchors: [(N, 7), ...] [(1，248，216，1，2，7），(1，248，216，1，2，7），(1，248，216，1，2，7)]
            gt_boxes: (B, M, 8)

        Returns:
            all_targets_dict = {
                'box_cls_labels': cls_labels, # (4，321408）
                'box_reg_targets': bbox_targets, # (4，321408，7）
                'reg_weights': reg_weights # (4，321408）
            }
        """
        # 初始化结构
        bbox_targets = []
        cls_labels = []
        reg_weights = []

        batch_size = gt_boxes_with_classes.shape[0]     # 16
        gt_classes = gt_boxes_with_classes[:, :, -1]    # (16, 44)
        gt_boxes = gt_boxes_with_classes[:, :, :-1]     # (16, 44, 7)

        # 依次对batch内的每个点云场景进行处理
        for k in range(batch_size):
            cur_gt = gt_boxes[k]    # 提取第k个点云帧gt，然后提取非零信息，去除非0无效信息 [44, 7] -> [38, 7]
            cnt = cur_gt.__len__() - 1      # 43
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]   # 提取当前第k点云帧的有效gt信息，保留非零项
            cur_gt_classes = gt_classes[k][:cnt + 1].int()   # 提取当前第k点云帧有效gt类别

            target_list = []
            for anchor_class_name, anchors in zip(self.anchor_class_names, all_anchors):    # 对每个类别及其配置anchor进行依次处理
                if cur_gt_classes.shape[0] > 1:
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name)    # 获取类别为'Car'的掩码矩阵:[True, True, ..., False]
                else:
                    mask = torch.tensor([self.class_names[c - 1] == anchor_class_name
                                         for c in cur_gt_classes], dtype=torch.bool)

                if self.use_multihead:  # False
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                    # if self.seperate_multihead:
                    #     selected_classes = cur_gt_classes[mask].clone()
                    #     if len(selected_classes) > 0:
                    #         new_cls_id = self.gt_remapping[anchor_class_name]
                    #         selected_classes[:] = new_cls_id
                    # else:
                    #     selected_classes = cur_gt_classes[mask]
                    selected_classes = cur_gt_classes[mask]
                else:
                    feature_map_size = anchors.shape[:3]   # zyx: (1, 248, 216)
                    anchors = anchors.view(-1, anchors.shape[-1])   # (107136,7) 107136=1x248x216x1x2
                    selected_classes = cur_gt_classes[mask]     # 被选择的类别 (14, )

                single_target = self.assign_targets_single(
                    anchors,        # reshape后的anchor矩阵 (107136,7)
                    cur_gt[mask],   # 根据当前类别的掩码矩阵选择当前处理的类别gt信息  (38, 7) -> (14, 7)
                    gt_classes=selected_classes,    # 当前处理的类别信息 (14, )
                    matched_threshold=self.matched_thresholds[anchor_class_name],       # 当前处理类别的正样本阈值
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name]    # 当前处理类别的负样本阈值
                )
                target_list.append(single_target)
            # 到目前为止，处理完该帧所有类别和anchor的前景和背景，后续就是对每个类别的结果进行拼接处理

            if self.use_multihead:  # False 没有启用多尺度构建head
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(-1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(-1, self.box_coder.code_size) for t in target_list],
                    'reg_weights': [t['reg_weights'].view(-1) for t in target_list]
                }

                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=0)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=0).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=0).view(-1)
            else:
                # 对当前点云场景每个类别的anchor分配结果进行view list 3：[(1，248，216, 2), (1，248，216, 2), (1，248，216, 2)]
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(*feature_map_size, -1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(*feature_map_size, -1, self.box_coder.code_size)  # (1，248，216, 2, 7)
                                        for t in target_list],
                    'reg_weights': [t['reg_weights'].view(*feature_map_size, -1) for t in target_list]  # (1，248，216, 2)
                }

                # 先concat后，在进行reshape list:3 (1，248，216, 2, 7) -> (1，248，216, 6, 7) -> (321408, 7)
                target_dict['box_reg_targets'] = torch.cat(
                    target_dict['box_reg_targets'], dim=-2
                ).view(-1, self.box_coder.code_size)
                # list:3 (1，248，216, 2) --> (1，248，216, 6) -> (321408,)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=-1).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=-1).view(-1)

            # 将当前点云场景处理结果分别追加到对应列表中
            bbox_targets.append(target_dict['box_reg_targets'])
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])

        # 将每个点云帧处理的结果进行stack堆叠
        bbox_targets = torch.stack(bbox_targets, dim=0)    # (16, 321408, 7)
        cls_labels = torch.stack(cls_labels, dim=0)        # (16, 321408)
        reg_weights = torch.stack(reg_weights, dim=0)      # (16, 321408)

        all_targets_dict = {
            'box_cls_labels': cls_labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights

        }
        return all_targets_dict

    def assign_targets_single(self, anchors, gt_boxes, gt_classes, matched_threshold=0.6, unmatched_threshold=0.45):
        """
        针对某一点云帧中的某一类别的anchors和gt_boxes，计算前景和背景anchor的类别，box编码和回归权重
        Args:
            anchors: (107136,7) 由于特征矩阵是固定的 (1, 248, 216)，所以这里的矩阵大小也是固定的
            gt_boxes: （14，7）
            gt_classes: (14,1)
            matched_threshold:0.6
            unmatched_threshold:0.45
        Returns:
            前景anchor
            ret_dict = {
                'box_cls_labels': labels, # (107136,)
                'box_reg_targets': bbox_targets,  # (107136,7)
                'reg_weights': reg_weights, # (107136,)
            }
        """
        num_anchors = anchors.shape[0]   # anchor的数量 107136
        num_gt = gt_boxes.shape[0]       # 当前点云帧处理某一类别gt的数量 16

        # labels用来存储anchor的分配标签，gt_ids用来存储anchor的分配gt
        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1  # (107136, )
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1  # (107136, )

        # 1. 构建每个anchor的正负样本label分配
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:  # 当前类别是否有gt信息以及anchor分配
            # 计算gt和anchors之间的overlap
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7]) \
                if self.match_height else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7])  # 计算anchor和gt之间的iou (107136, 14)

            # 找到每个anchor最匹配的gt的索引和iou
            # NOTE: The speed of these two versions depends the environment and the number of anchors
            # anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda()
            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(dim=1)    # (107136，）找到每个anchor最匹配的gt的索引
            anchor_to_gt_max = anchor_by_gt_overlap[torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax]  # （107136，）找到每个anchor最匹配的gt的iou

            # 提取最匹配的anchor，避免没有anchor满足索设定的阈值
            # gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=0)).cuda()
            gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(dim=0)     # (14,) 找到每个gt最匹配anchor的索引
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)]   # （14，）找到每个gt最匹配anchor的iou
            empty_gt_mask = gt_to_anchor_max == 0   # 如果最匹配iou为0，表示某个gt没有与之匹配的anchor
            gt_to_anchor_max[empty_gt_mask] = -1    # 没有与之匹配的anchor在iou值中设置为-1

            # 以gt为基础，逐个anchor对应，比如第一个gt的最大iou为0.9，则在所有anchor中找iou为0.9的anchor
            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0]   # 找到满足最大iou的每个anchor
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]   # 找到最大iou的gt索引
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]    # 将gt的类别赋值到对应的anchor的label中 (107136,)
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int()          # 将gt的索引赋值到对应的anchor的gt_id中 (107136,)

            # 这里应该对labels和gt_ids的操作应该包含了上面的anchors_with_max_overlap
            pos_inds = anchor_to_gt_max >= matched_threshold        # 找到最匹配的anchor中iou大于给定阈值的mask (107136,)
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]     # 找到最匹配的anchor中iou大于给定阈值的gt的索引 (104,)
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]      # 将pos anchor对应gt的类别赋值到对应的anchor的label中 (107136,)
            gt_ids[pos_inds] = gt_inds_over_thresh.int()            # 将pos anchor对应gt的索引赋值到对应的anchor的gt_id中 (107136,)
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]  # 找到背景anchor索引 (106874，)
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)

        fg_inds = (labels > 0).nonzero()[:, 0]      # 找到前景点的索引 (104,)

        # 是否对正负样本进行采样
        if self.pos_fraction is not None:
            num_fg = int(self.pos_fraction * self.sample_size)
            if len(fg_inds) > num_fg:
                num_disabled = len(fg_inds) - num_fg
                disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                labels[disable_inds] = -1
                fg_inds = (labels > 0).nonzero()[:, 0]

            num_bg = self.sample_size - (labels > 0).sum()
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))]
                labels[enable_inds] = 0
            # bg_inds = torch.nonzero(labels == 0)[:, 0]
        else:
            if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                labels[:] = 0
            else:
                labels[bg_inds] = 0     # 将背景点的label赋值为0
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]    # 将前景点赋值为对应类别（这里重复了一遍）

        # 2. 构建正样本anchor需要预测拟合的编码信息（负样本anchor全部设置为0） 这里可以构建一个corner loss
        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size))   # (107136,7)
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]     # 提取前景对应的gt box信息 (104, 7)
            fg_anchors = anchors[fg_inds, :]    # 提取前景anchor (104, 7)
            bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)    # 编码gt和前景anchor，并赋值到bbox_targets的对应位置

        # 3. 构建正负样本回归权重，其中背景anchor权重为0，前景anchor权重为1
        reg_weights = anchors.new_zeros((num_anchors,))    # 回归权重 (107136,)
        if self.norm_by_num_examples:   # False
            num_examples = (labels >= 0).sum()
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0    # 将前景anchor的权重赋1

        ret_dict = {
            'box_cls_labels': labels,           # 背景anchor的label是0，前景的anchor的label是当前处理的类别1  (107136,)
            'box_reg_targets': bbox_targets,    # 编码后待模型预测拟合的结果，背景anchor的编码信息也是0 (107136,7)
            'reg_weights': reg_weights,         # 背景anchor权重为0，前景anchor权重为1  (107136,)
        }
        return ret_dict
