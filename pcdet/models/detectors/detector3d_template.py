import os

import torch
import torch.nn as nn
import numpy as np
from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils.spconv_utils import find_all_spconv_keys
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads
from ..backbones_2d import map_to_bev
from ..backbones_3d import pfe, vfe
from ..model_utils import model_nms_utils

# 功能：网络结构的基类
class Detector3DTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        """
        Args:
            model_cfg:   yaml配置文件的MODEL部分
            num_class:   类别数目（kitti数据集一般用3个类别：'Car', 'Pedestrian', 'Cyclist'）
            dataset:     训练数据集
        """
        super().__init__()          # Detector3DTemplate继承nn.Module
        self.model_cfg = model_cfg  # MODEL配置文件
        self.num_class = num_class  # 类别数量
        self.dataset = dataset      # 训练数据集
        self.class_names = dataset.class_names  # 类别名称（列表形式）

        # 向pytoch的模块添加持久缓冲区，缓冲区通常不被视为模型参数，不会自动更新，可以使用给定的名称作为属性访问
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        # 3d检测算法可能用到的模块 (这里不能改变他们的顺序，是按顺序来进行模块的构建的)
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'     # 返回当前模式

    def update_global_step(self):
        self.global_step += 1      # 缓冲区注册的参数，默认初始化为0

    def build_networks(self):
        # 'module_list'键是返回结构，其他键是搭建网络模块的传参
        model_info_dict = {
            'module_list': [],      # 具体返回的内容
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,  # 初始点云特征维度 dim=4
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,     # 点云输出特征维度 初始为4
            'grid_size': self.dataset.grid_size,                    # 网格大小：[432 496   1]
            'point_cloud_range': self.dataset.point_cloud_range,    # 点云范围
            'voxel_size': self.dataset.voxel_size,                  # 体素大小：[0.16, 0.16, 4]
            'depth_downsample_factor': self.dataset.depth_downsample_factor
        }

        # 顺序遍历3d检测算法可能用到的模块，如果相关模块有配置参数则进行对于参数的构建
        for module_name in self.module_topology:
            # 如果配置文件没有设定模块参数，则直接返回None与没有修改的model_info_dict
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(      # 动态选择build函数执行
                model_info_dict=model_info_dict     # 既当作函数传参，也将返回结果保留在module_list中
            )
            self.add_module(module_name, module)    # 调用nn.Module中的add_module方，增加网络子模块

        return model_info_dict['module_list']

    # 功能：构建voxel feature encoder(VFE)模块
    def build_vfe(self, model_info_dict):
        # 如果配置文件没有设置vfe，则对应的vfe模块为None
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        # 选择选用的Voxel Feature Encoder模块
        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,               # VFE配置文件
            num_point_features=model_info_dict['num_rawpoint_features'],    # 原始特征维度
            point_cloud_range=model_info_dict['point_cloud_range'],         # 点云范围
            voxel_size=model_info_dict['voxel_size'],   # 体素大小
            grid_size=model_info_dict['grid_size'],     # 网格大小
            depth_downsample_factor=model_info_dict['depth_downsample_factor']  # 下采样因子(这里设置为None)
        )

        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()     # 更新输出特征维度 (4 -> 10)
        model_info_dict['module_list'].append(vfe_module)   # 保存具体参数模块
        return vfe_module, model_info_dict

    # 功能：构建BACKBONE_3D模块
    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        model_info_dict['backbone_channels'] = backbone_3d_module.backbone_channels \
            if hasattr(backbone_3d_module, 'backbone_channels') else None
        return backbone_3d_module, model_info_dict

    # 功能：构建MAP_TO_BEV模块
    def build_map_to_bev_module(self, model_info_dict):
        # 如果配置文件没有设置MAP_TO_BEV，则对应的MAP_TO_BEV模块为None
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,      # MAP_TO_BEV配置部分
            grid_size=model_info_dict['grid_size']    # 传入网格大小
        )
        model_info_dict['module_list'].append(map_to_bev_module)                    # 不涉及模型参数
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features    # 增加了一个key，特征维度不变(64)
        return map_to_bev_module, model_info_dict

    # 功能：构建BACKBONE_2D模块
    def build_backbone_2d(self, model_info_dict):
        # 如果配置文件没有设置BACKBONE_2D，则对应的BACKBONE_2D模块为None
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict.get('num_bev_features', None)    # 在map_to_bev模块增加了这个key
        )
        model_info_dict['module_list'].append(backbone_2d_module)   # 增加2d特征提取网络结构
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features   # 更新通过backbone_2d后的输出特征维度(384)
        return backbone_2d_module, model_info_dict

    # 功能：构建point feature encoder模块
    def build_pfe(self, model_info_dict):
        # 如果配置文件没有设置PFE，则对应的PFE模块为None
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict

        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](
            model_cfg=self.model_cfg.PFE,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_bev_features=model_info_dict['num_bev_features'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features']
        )
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict

    # 功能：构想DENSE_HEAD模块
    def build_dense_head(self, model_info_dict):
        # 如果配置文件没有设置DENSE_HEAD，则对应的DENSE_HEAD模块为None
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict

        # 根据NAME确认选择dense head模块，后续括号是传参
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'],
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),  # OneStage模型没有设置ROI_HEAD，设置为False
            voxel_size=model_info_dict.get('voxel_size', False)
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    def build_point_head(self, model_info_dict):
        if self.model_cfg.get('POINT_HEAD', None) is None:
            return None, model_info_dict

        if self.model_cfg.POINT_HEAD.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            num_point_features = model_info_dict['num_point_features_before_fusion']
        else:
            num_point_features = model_info_dict['num_point_features']

        point_head_module = dense_heads.__all__[self.model_cfg.POINT_HEAD.NAME](
            model_cfg=self.model_cfg.POINT_HEAD,
            input_channels=num_point_features,
            num_class=self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1,
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](
            model_cfg=self.model_cfg.ROI_HEAD,
            input_channels=model_info_dict['num_point_features'],
            backbone_channels= model_info_dict.get('backbone_channels', None),
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']   # 16
        recall_dict = {}
        pred_dicts = []
        # 依次对每帧点云进行处理
        for index in range(batch_size):
            # 1. 分别获取当前点云帧的label信息和真实场景的gt信息
            if batch_dict.get('batch_index', None) is not None:    # False
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]   # 获取第index个点云的预测gt信息 (321408, 7)
            src_box_preds = box_preds   # (321408, 7)
            
            if not isinstance(batch_dict['batch_cls_preds'], list):     # 如果是非列表形式
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]   # 获取第index个点云的label信息 (321408, 3) 这里是one-hot编码形式的预测

                src_cls_preds = cls_preds   # (321408, 3)
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)    # 转化为logits
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            # 2. 对当前点云帧进行nms处理，挑选出被选择的anchor索引
            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]

                    # 调用multi_classes_nms函数进行NMS
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                # torch.max()函数的第一个返回值是每行的最大值，第二个返回值是每行最大值的索引
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)   # 这里是输入cls_preds是已经过sigmoid处理，构建成为概率的
                if batch_dict.get('has_class_labels', False):   # False
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1   # 索引标签+1

                # 调用gpu函数进行nms，返回的是被选择的索引和索引分数
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds,   # 将每个预测的anchor的最大概率当做是置信度
                    box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH  # 0.1
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores          # 预测分数
                final_labels = label_preds[selected]    # 预测类别
                final_boxes = box_preds[selected]       # 预测box信息

            # 3. 计算召回率，且recall_dict是其中的参数，循环调用更新
            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )        

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k >= 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    # 作用：加载断点权重的具体操作
    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)    # 利用pytorch接口直接导入更新后的权重
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    # 作用：加载预训练模型权重
    def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        if not pre_trained_path is None:
            pretrain_checkpoint = torch.load(pre_trained_path, map_location=loc_type)
            pretrain_model_state_disk = pretrain_checkpoint['model_state']
            model_state_disk.update(pretrain_model_state_disk)
            
        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    # 作用：加载断点权重
    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)    # 加载断点权重，权重文件包含模型参数和优化器参数
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)       # 加载模型参数

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])    # 加载优化器参数
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch
