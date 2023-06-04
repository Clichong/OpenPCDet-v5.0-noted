from functools import partial

import numpy as np

from ...utils import common_utils
from . import augmentor_utils, database_sampler

# 数据增强的基类，但各种具体的数据增强实现方式是在augmentor_utils.py脚本中实现
class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []     # 用列表顺序存储数据增强处理顺序(队列，先进先出)
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        # 依次添加各数据增强方式进入队列
        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:  # 如果在不使用增强清单中，则跳过这部分的数据增强设置
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)     # 在对列中顺序添加数据增加模块

    # 功能：基于DataBaseSampler类实现gt样本的采样,实现gt筛选，采样以及copy paste的操作
    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    # 功能: 对点云和gt进行随机反转，具体实现函数是在augmentor_utils中
    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']   # augmentor_utils中只提供了xy轴的反转
            # 调用augmentor_utils中的函数随机翻转box和点云
            gt_boxes, points, enable = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points, return_flip=True
            )
            data_dict['flip_%s'%cur_axis] = enable  # 记录是否选择翻转
            if 'roi_boxes' in data_dict.keys():  # False
                num_frame, num_rois,dim = data_dict['roi_boxes'].shape
                roi_boxes, _, _ = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                data_dict['roi_boxes'].reshape(-1,dim), np.zeros([1,3]), return_flip=True, enable=enable
                )
                data_dict['roi_boxes'] = roi_boxes.reshape(num_frame, num_rois,dim)

        # 更新反转后的gt与点云坐标
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    # 功能: 对点云和gt进行随机旋转，具体实现函数是在augmentor_utils中
    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']   # 获取设置的选择范围
        # 如果旋转范围不是列表，则取正负
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        # 调用augmentor_utils中的函数旋转box和点云
        gt_boxes, points, noise_rot = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range, return_rot=True
        )
        if 'roi_boxes' in data_dict.keys():  # False
            num_frame, num_rois,dim = data_dict['roi_boxes'].shape
            roi_boxes, _, _ = augmentor_utils.global_rotation(
            data_dict['roi_boxes'].reshape(-1, dim), np.zeros([1, 3]), rot_range=rot_range, return_rot=True, noise_rotation=noise_rot)
            data_dict['roi_boxes'] = roi_boxes.reshape(num_frame, num_rois,dim)

        # 更新旋转后的gt与点云坐标
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_rot'] = noise_rot  # 旋转的角度
        return data_dict

    # 功能: 对点云和gt进行随机缩放，具体实现函数是在augmentor_utils中
    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        
        if 'roi_boxes' in data_dict.keys():    # False
            gt_boxes, roi_boxes, points, noise_scale = augmentor_utils.global_scaling_with_roi_boxes(
                data_dict['gt_boxes'], data_dict['roi_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'], return_scale=True
            )
            data_dict['roi_boxes'] = roi_boxes
        else:
            gt_boxes, points, noise_scale = augmentor_utils.global_scaling(
                data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'], return_scale=True
            )

        # 更新缩放后的gt与点云坐标
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_scale'] = noise_scale  # 记录当前随机缩放的尺度大小
        return data_dict

    # 功能：随机图片翻转
    def random_image_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        images = data_dict["images"]
        depth_maps = data_dict["depth_maps"]
        gt_boxes = data_dict['gt_boxes']
        gt_boxes2d = data_dict["gt_boxes2d"]
        calib = data_dict["calib"]
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['horizontal']
            images, depth_maps, gt_boxes = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                images, depth_maps, gt_boxes, calib,
            )

        data_dict['images'] = images
        data_dict['depth_maps'] = depth_maps
        data_dict['gt_boxes'] = gt_boxes
        return data_dict

    # 功能：全局随机平移(整个点云场景的点都平移相同的参数)
    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)
        noise_translate_std = config['NOISE_TRANSLATE_STD']
        assert len(noise_translate_std) == 3

        # 分别对xyz轴产生随机平移距离
        noise_translate = np.array([
            np.random.normal(0, noise_translate_std[0], 1),
            np.random.normal(0, noise_translate_std[1], 1),
            np.random.normal(0, noise_translate_std[2], 1),
        ], dtype=np.float32).T

        # 由于是平移，所以对点和gt信息直接相加即可
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        points[:, :3] += noise_translate
        gt_boxes[:, :3] += noise_translate
                
        if 'roi_boxes' in data_dict.keys():
            data_dict['roi_boxes'][:, :3] += noise_translate
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    # 功能：局部随机平移（对每个gt都进行随机平移，所以gt之间的平移参数不同）
    def random_local_translation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_translation, config=config)
        offset_range = config['LOCAL_TRANSLATION_RANGE']    # 均匀分布的范围
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']

        # 根据字典分别对哪个轴方向进gt的随机平移，这里xy轴的定义距离可以稍微远一点
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            gt_boxes, points = getattr(augmentor_utils, 'random_local_translation_along_%s' % cur_axis)(
                gt_boxes, points, offset_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    # 功能: 局部随机旋转（需要注意对gt角度的更新，中心和尺寸不变）
    def random_local_rotation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_rotation, config=config)
        # 这里是对每个gt进行局部选择，选择的范围相比全局的范围可以大一些
        rot_range = config['LOCAL_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.local_rotation(  # 这里默认是沿z轴进行旋转，也就是从bev角度(xoy平面)看其旋转角度
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    # 功能: 局部随机缩放（需要注意对gt尺寸的更新，中心和角度不变）
    def random_local_scaling(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_scaling, config=config)
        gt_boxes, points = augmentor_utils.local_scaling(   # 对gt的点更新的同时注意gt的尺寸也要更新
            data_dict['gt_boxes'], data_dict['points'], config['LOCAL_SCALE_RANGE']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    # 功能：对超过左右上下阈值的点恶化gt进行筛选
    def random_world_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_world_frustum_dropout, config=config)

        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            # 有上下左右四种配置以供选择
            # top和bottom：对z轴进行处理； left和right：对y轴进行处理
            gt_boxes, points = getattr(augmentor_utils, 'global_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    # 功能：对每个gt内的点超过左右上下阈值的进行筛选
    def random_local_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_frustum_dropout, config=config)

        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(augmentor_utils, 'local_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    # 功能：自监督——SE-SSD提出的一种数据增强方式
    # 主要分为三个部分：1）随机删除金字塔点；2）随机交换两个金字塔的点；3）随机稀疏金字塔的点
    def random_local_pyramid_aug(self, data_dict=None, config=None):
        """
        Refer to the paper:
            SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud
        """
        if data_dict is None:
            return partial(self.random_local_pyramid_aug, config=config)

        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']

        # 1. 随机删除金字塔点
        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_dropout(gt_boxes, points,
                                                                           config['DROP_PROB']) #
        # 2. 随机稀疏金字塔点
        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_sparsify(gt_boxes, points,
                                                                            config['SPARSIFY_PROB'],
                                                                            config['SPARSIFY_MAX_NUM'],
                                                                            pyramids)
        # 3. 随机交换两个金字塔点
        gt_boxes, points = augmentor_utils.local_pyramid_swap(gt_boxes, points,
                                                              config['SWAP_PROB'],
                                                              config['SWAP_MAX_NUM'],
                                                              pyramids)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def forward(self, data_dict):      # 在prepare_data中进行
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        # 遍历增强队列，逐个增强器做数据增强
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        # 将方位角限制在[-pi,pi], 就是前视图FOV的视角
        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )

        # if 'calib' in data_dict:
        #     data_dict.pop('calib')

        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')

        if 'gt_boxes_mask' in data_dict:    # False
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            if 'gt_boxes2d' in data_dict:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]

            data_dict.pop('gt_boxes_mask')
        return data_dict
