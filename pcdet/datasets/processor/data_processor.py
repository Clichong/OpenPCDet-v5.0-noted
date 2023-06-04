from functools import partial

import numpy as np
from skimage import transform

from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass

# 调用spconv的VoxelGeneratorV2来将点划分为Voxel
class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator # Fail
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator # Fail
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator     # Success
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(     # 具体的初始化
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))  # point2voxel的具体实现
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


# 数据处理的基类
class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None

        self.voxel_generator = None

        # 在队列中顺序添加数据处理模块，工厂模式，根据不同的配置，只需要增加相应的方法即可实现不同的调用
        self.data_processor_queue = []
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)    # 直接根据Name确定处理函数名
            self.data_processor_queue.append(cur_processor)     # 在forward函数中调用

    # 功能:过滤范围外的点和gt
    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)     # 返回设定好config参数的函数，供后续的forward进行

        # 过滤范围外的点
        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)   # 返回是否符合范围的掩码
            data_dict['points'] = data_dict['points'][mask]     # 根据掩码来过滤点

        # 过滤范围外的gt
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:   # True
            mask = box_utils.mask_boxes_outside_range_numpy(    # 返回gt中心点是否符合范围的掩码
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1), 
                use_center_to_filter=config.get('USE_CENTER_TO_FILTER', True)   # 选择是否用中心点来进行过滤
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]     # 根据掩码来过滤gt
        return data_dict

    # 功能：随机打乱点
    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        # 训练过程打乱，测试过程不打乱
        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])    # 生成随机序列索引
            points = points[shuffle_idx]    # 根据索引重新编排点顺序
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        
        return data_dict

    # 功能：将点云转换为voxel,调用spconv的VoxelGeneratorV2
    def transform_points_to_voxels(self, data_dict=None, config=None):
        # 初始化确认网格大小与体素大小
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)   # 网格数量
            self.grid_size = np.round(grid_size).astype(np.int64)   # 四舍五入取整
            self.voxel_size = config.VOXEL_SIZE      # 从配置文件中获取指定的体素大小
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,   # 体素大小 [0.16, 0.16, 4]
                coors_range_xyz=self.point_cloud_range,     # 场景范围 [0, -39.68, -3, 69.12, 39.68, 1]
                num_point_features=self.num_point_features, # 每个点特征数量 4
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,   # 每个voxel最大点云数 32
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],  # 场景的最大voxel数 训练模式是 16000
            )

        # 调用spconv的voxel_generator的generate方法生成体素
        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output
        """
            voxels: (num_voxels, max_points_per_voxel, 3 + C)  表示每个体素中有32个点云，每个点有3+C（4）和特征维度
            coordinates: (num_voxels, 3)     在点云场景中voxel的位置信息，pointpillars算法这里的voxel就是pillars，所以只有平面上的2d坐标，没有z维度切分
            num_points: (num_voxels)         表示每个voxel内的有效点数量
        """

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        # 更新待处理数据
        data_dict['voxels'] = voxels                # voxel内的点特征   (N, 32, 4)
        data_dict['voxel_coords'] = coordinates     # voxel位置        (N, 3)
        data_dict['voxel_num_points'] = num_points  # voxel的有效点数量  (N, )
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        # 依次进行各类数据处理，不断更新data_dict
        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)  # 具体执行时才传入data_dict，在初始化时指保留config设置

        return data_dict
