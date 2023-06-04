from collections import defaultdict
from pathlib import Path

import numpy as np
import torch.utils.data as torch_data

from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder


class DatasetTemplate(torch_data.Dataset):

    # 初始化功能：主要是确定好特征编码、数据增强和数据处理几大步骤，这是基类数据集的几个基本操作
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)   # 根据配置文件确定路径
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        # 点云范围
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)

        # 1)点云特征编码
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )

        # 2)点云数据增强(只在训练过程中设置，测试过程为None)
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None

        # 3)点云数据处理
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

        # 点云voxel场景的尺度
        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        if hasattr(self.data_processor, "depth_downsample_factor"):
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        else:
            self.depth_downsample_factor = None

    # 功能：根据self.mode可以获得当前模型是train还是test模式
    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7 or 9), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        
        def get_template_prediction(num_samples):
            box_dim = 9 if self.dataset_cfg.get('TRAIN_WITH_SPEED', False) else 7
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, box_dim]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'metadata' in batch_dict:
                single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    # 对单帧数据进行处理（__getitem__中进行具体调用）
    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                frame_id: str
                calib: Calibration类对象
                points: optional, (N, 3 + C_in) 每个点特征就是未知信息xyz+反射强度 (只保留了FOV视角的点云)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                road_plane: ...
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        # 训练模式下，对存在于class_name中的数据进行增强
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            # 每一个gt是否在待检测的类别列表中
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
            
            if 'calib' in data_dict:
                calib = data_dict['calib']

            # 1）先进行数据增强
            # 数据增强 传入字典参数，**data_dict是将data_dict里面的key-value对都拿出来
            # 下面在原数据的基础上增加gt_boxes_mask，构造新的字典传入data_augmentor的forward函数
            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask      # gt的掩码(布尔变量列表)
                }
            )
            if 'calib' in data_dict:
                data_dict['calib'] = calib

        # 筛选需要检测的gt_boxes，添加分类id在gt的最后一个维度上
        if data_dict.get('gt_boxes', None) is not None:   # True
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)    # 过滤非设定类别
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected] # 根据selected，选取需要的gt_boxes和gt_names
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)       # 设定类别名称的分类number
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)    # 将分类信息拼接在gt的最后一个维度
            data_dict['gt_boxes'] = gt_boxes    # 更新gt信息 (N, 7) -> (N, 8)
            if data_dict.get('gt_boxes2d', None) is not None:   # False
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]

        # 2）使用点的哪些属性 比如x,y,z等，但是这里其实没有改变太大信息
        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        # 3) 最后进行数据处理
        # 对点云进行预处理，包括移除超出point_cloud_range的点、 打乱点的顺序以及将点云转换为voxel
        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        # 在训练过程的某一帧在极端条件下，当前帧没有gt没有符合碰撞测试的采样gt时，就会再挑选另外的帧来作为当前帧的处理结果
        # 相当于重新执行整个数据处理流程作为当前index的处理结果
        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())   # 重新旋转另外的点云帧
            return self.__getitem__(new_index)      # 重新调用__getitem__方法获取新索引的数据字典

        data_dict.pop('gt_names', None)

        return data_dict

    # 对batch数据进行处理，这是因为训练集中不同的点云的gt框不同，pytorch需要输入数据的维度是一致的，所以需要根据最大值构造空矩阵再填充处理
    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)    # 字典的key默认是一个空列表
        # 把batch里面的每个sample按照key-value合并
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)    # 16
        ret = {}

        # 将合并后的key内的value进行拼接，先获取最大值，构造空矩阵，不足的部分补0
        # 因为pytorch要求输入数据维度一致
        for key, val in data_dict.items():
            try:
                # voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                # voxel_coords: optional (num_voxels, 3)
                # voxel_num_points: optional (num_voxels)
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)

                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):  # i作为对应列表的坐标序号
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)   # 在每个坐标前面加上序号，填充的数值是i
                        """
                            np.pad参数介绍：
                            ((0,0),(1,0))
                            在二维数组array第一维（此处便是行）前面填充0行，最后面填充0行；
                            在二维数组array第二维（此处便是列）前面填充1列，最后面填充0列
                            mode='constant'表示指定填充的参数
                            constant_values=i 表示第一维填充i
                        """
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)

                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])    # 获取一个batch中所有帧中3D box最大的数量
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)   # 构造空的box3d矩阵（B, N, 7）
                    for k in range(batch_size):     # 将每帧的点云结果替换掉空矩阵
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]  # val[k]表示一个batch中的第k帧
                    ret[key] = batch_gt_boxes3d     # [N, 5]

                elif key in ['roi_boxes']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, val[0].shape[0], max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k,:, :val[k].shape[1], :] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['roi_scores', 'roi_labels']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, val[0].shape[0], max_gt), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k,:, :val[k].shape[1]] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d

                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        pad_value = 0

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)

                elif key in ['calib']:
                    ret[key] = val

                elif key in ["points_2d"]:
                    max_len = max([len(_val) for _val in val])
                    pad_value = 0
                    points = []
                    for _points in val:
                        pad_width = ((0, max_len-len(_points)), (0,0))
                        points_pad = np.pad(_points,
                                pad_width=pad_width,
                                mode='constant',
                                constant_values=pad_value)
                        points.append(points_pad)
                    ret[key] = np.stack(points, axis=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
