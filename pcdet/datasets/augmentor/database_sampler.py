import pickle

import os
import copy
import numpy as np
from skimage import io
import torch
import SharedArray
import torch.distributed as dist

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils, common_utils, calibration_kitti
from pcdet.datasets.kitti.kitti_object_eval_python import kitti_common

class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names     # 一般是3类：['Car', 'Pedestrian', 'Cyclist']
        self.sampler_cfg = sampler_cfg     # 数据增强中gt_sampling方法的配置文件

        # EasyDick方法：如果有则获取，没有则用后一个参数赋值
        self.img_aug_type = sampler_cfg.get('IMG_AUG_TYPE', None)
        self.img_aug_iou_thresh = sampler_cfg.get('IMG_AUG_IOU_THRESH', 0.5)

        self.logger = logger

        # 构造成{'Car': [], 'Pedestrian': [], 'Cyclist': []}，用kitti_dbinfos_train.pkl的对应类别信息填充
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []  # 空列表存储对应类别信息

        self.use_shared_memory = sampler_cfg.get('USE_SHARED_MEMORY', False)

        # 循环提取序列文件的全部对应类别信息，但是这里只对train构建了kitti_dbinfos_train.pkl
        # 所以这里的使用的dbinfos应该与INFO_PATH的序列对应
        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path     # kitti_dbinfos_train.pkl序列文件绝对路径

            # 文件路径不存在
            if not db_info_path.exists():
                assert len(sampler_cfg.DB_INFO_PATH) == 1
                sampler_cfg.DB_INFO_PATH[0] = sampler_cfg.BACKUP_DB_INFO['DB_INFO_PATH']
                sampler_cfg.DB_DATA_PATH[0] = sampler_cfg.BACKUP_DB_INFO['DB_DATA_PATH']
                db_info_path = self.root_path.resolve() / sampler_cfg.DB_INFO_PATH[0]
                sampler_cfg.NUM_POINT_FEATURES = sampler_cfg.BACKUP_DB_INFO['NUM_POINT_FEATURES']

            # dbinfos_train.pkl文件将kitti数据集的8个类别分别用8个key对应的list来存储其所有信息，每个类别的每个对象用一个dict存储其9个属性
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)     # 加载序列文件（类别信息）
                # 提取所需类别信息存储在对应列表中，比较巧妙的写法。同时利用extend可以追加整个序列所有信息
                [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]

        # 对gt进行额外的采样操作，比如过滤等等
        # 原始的设置只有两个过滤采样：filter_by_min_points 和 filter_by_difficulty
        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)    # 前面是函数入口，后面的传参（需要注意顺序）

        self.gt_database_data_key = self.load_db_to_shared_memory() if self.use_shared_memory else None

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)   # pointpillars设置为False，默认是True

        # 对每个类别重新统计信息，包含采样数量、过滤后的gt数量、以及分配索引(从0开始分配)
        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')   # 以:区分左右，eg ‘Car:15’ -> Car / 15
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,                               # 采样数量
                'pointer': len(self.db_infos[class_name]),              # 类别gt样本数量
                'indices': np.arange(len(self.db_infos[class_name]))    # 分配索引
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __del__(self):
        if self.use_shared_memory:
            self.logger.info('Deleting GT database from shared memory')
            cur_rank, num_gpus = common_utils.get_dist_info()
            sa_key = self.sampler_cfg.DB_DATA_PATH[0]
            if cur_rank % num_gpus == 0 and os.path.exists(f"/dev/shm/{sa_key}"):
                SharedArray.delete(f"shm://{sa_key}")

            if num_gpus > 1:
                dist.barrier()
            self.logger.info('GT database has been removed from shared memory')

    def load_db_to_shared_memory(self):
        self.logger.info('Loading GT database to shared memory')
        cur_rank, world_size, num_gpus = common_utils.get_dist_info(return_gpu_per_machine=True)

        assert self.sampler_cfg.DB_DATA_PATH.__len__() == 1, 'Current only support single DB_DATA'
        db_data_path = self.root_path.resolve() / self.sampler_cfg.DB_DATA_PATH[0]
        sa_key = self.sampler_cfg.DB_DATA_PATH[0]

        if cur_rank % num_gpus == 0 and not os.path.exists(f"/dev/shm/{sa_key}"):
            gt_database_data = np.load(db_data_path)
            common_utils.sa_create(f"shm://{sa_key}", gt_database_data)

        if num_gpus > 1:
            dist.barrier()
        self.logger.info('GT database has been saved to shared memory')
        return sa_key

    # 功能：对太困难的gt样本进行过滤处理，通过removed_difficulty困难等级列表来控制过滤信息
    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        # 其实还是遍历每个类别对象，key既为cur_classname，dinfos是类别的gt列表
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)   # 记录过滤前数目
            # 遍历类别下的每个gt，如果困难等级需要去除则过滤
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty   # 如果gt的困难等级不在removed_difficulty列表中则保存
            ]
            # 打印过滤类比中gt数量的变化情况
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))

        # 返回过滤后的gt信息
        return new_db_infos

    # 功能：对范围内点云数量没有超出阈值的gt对象进行忽略处理
    def filter_by_min_points(self, db_infos, min_gt_points_list):
        # 依次对指定的每个类别进行处理，默认为3类
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')     # 切分字符串:的两边
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():     # 如果没有对应类别的gt信息则不作处理
                # 对符合阈值情况的gt对象进行添加(添加的操作比在原有数据中删除要快)
                filtered_infos = []
                for info in db_infos[name]:    # 遍历某个类别的所有gt信息
                    # 如果对象范围区域内点超出阈值，则添加，否则就是被过滤不作处理
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                # 打印过滤类比中gt数量的变化情况
                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos     # 重新赋值过滤后的gt信息

        # 返回过滤后的gt信息
        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))     # 将索引随机排列
            pointer = 0
        # 获取打乱顺序后的indices前 sample_num 个db_infos（对每个类别进行同样的操作）
        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer   # 更新采样个数 10759 -> 15
        sample_group['indices'] = indices   # 更新为随机打乱的索引
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]  路面方程信息
            calib:

        Returns:
        """
        a, b, c, d = road_planes    # 路面方程信息
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])   # 将box的中心点转换到统一坐标系下
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b     # 计算cam距离路面的距离
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]     # 计算雷达距离地面的高度
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height  # 将box的地面z坐标减去lidar高度
        gt_boxes[:, 2] -= mv_height  # lidar view   将box放到地面上
        return gt_boxes, mv_height

    def copy_paste_to_image_kitti(self, data_dict, crop_feat, gt_number, point_idxes=None):
        kitti_img_aug_type = 'by_depth'
        kitti_img_aug_use_type = 'annotation'

        image = data_dict['images']
        boxes3d = data_dict['gt_boxes']
        boxes2d = data_dict['gt_boxes2d']
        corners_lidar = box_utils.boxes_to_corners_3d(boxes3d)
        if 'depth' in kitti_img_aug_type:
            paste_order = boxes3d[:,0].argsort()
            paste_order = paste_order[::-1]
        else:
            paste_order = np.arange(len(boxes3d),dtype=np.int)

        if 'reverse' in kitti_img_aug_type:
            paste_order = paste_order[::-1]

        paste_mask = -255 * np.ones(image.shape[:2], dtype=np.int)
        fg_mask = np.zeros(image.shape[:2], dtype=np.int)
        overlap_mask = np.zeros(image.shape[:2], dtype=np.int)
        depth_mask = np.zeros((*image.shape[:2], 2), dtype=np.float)
        points_2d, depth_2d = data_dict['calib'].lidar_to_img(data_dict['points'][:,:3])
        points_2d[:,0] = np.clip(points_2d[:,0], a_min=0, a_max=image.shape[1]-1)
        points_2d[:,1] = np.clip(points_2d[:,1], a_min=0, a_max=image.shape[0]-1)
        points_2d = points_2d.astype(np.int)
        for _order in paste_order:
            _box2d = boxes2d[_order]
            image[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] = crop_feat[_order]
            overlap_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] += \
                (paste_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] > 0).astype(np.int)
            paste_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] = _order

            if 'cover' in kitti_img_aug_use_type:
                # HxWx2 for min and max depth of each box region
                depth_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2],0] = corners_lidar[_order,:,0].min()
                depth_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2],1] = corners_lidar[_order,:,0].max()

            # foreground area of original point cloud in image plane
            if _order < gt_number:
                fg_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] = 1

        data_dict['images'] = image

        # if not self.joint_sample:
        #     return data_dict

        new_mask = paste_mask[points_2d[:,1], points_2d[:,0]]==(point_idxes+gt_number)
        if False:  # self.keep_raw:
            raw_mask = (point_idxes == -1)
        else:
            raw_fg = (fg_mask == 1) & (paste_mask >= 0) & (paste_mask < gt_number)
            raw_bg = (fg_mask == 0) & (paste_mask < 0)
            raw_mask = raw_fg[points_2d[:,1], points_2d[:,0]] | raw_bg[points_2d[:,1], points_2d[:,0]]
        keep_mask = new_mask | raw_mask
        data_dict['points_2d'] = points_2d

        if 'annotation' in kitti_img_aug_use_type:
            data_dict['points'] = data_dict['points'][keep_mask]
            data_dict['points_2d'] = data_dict['points_2d'][keep_mask]
        elif 'projection' in kitti_img_aug_use_type:
            overlap_mask[overlap_mask>=1] = 1
            data_dict['overlap_mask'] = overlap_mask
            if 'cover' in kitti_img_aug_use_type:
                data_dict['depth_mask'] = depth_mask

        return data_dict

    def collect_image_crops_kitti(self, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx):
        calib_file = kitti_common.get_calib_path(int(info['image_idx']), self.root_path, relative_path=False)
        sampled_calib = calibration_kitti.Calibration(calib_file)
        points_2d, depth_2d = sampled_calib.lidar_to_img(obj_points[:,:3])

        if True:  # self.point_refine:
            # align calibration metrics for points
            points_ract = data_dict['calib'].img_to_rect(points_2d[:,0], points_2d[:,1], depth_2d)
            points_lidar = data_dict['calib'].rect_to_lidar(points_ract)
            obj_points[:, :3] = points_lidar
            # align calibration metrics for boxes
            box3d_raw = sampled_gt_boxes[idx].reshape(1,-1)
            box3d_coords = box_utils.boxes_to_corners_3d(box3d_raw)[0]
            box3d_box, box3d_depth = sampled_calib.lidar_to_img(box3d_coords)
            box3d_coord_rect = data_dict['calib'].img_to_rect(box3d_box[:,0], box3d_box[:,1], box3d_depth)
            box3d_rect = box_utils.corners_rect_to_camera(box3d_coord_rect).reshape(1,-1)
            box3d_lidar = box_utils.boxes3d_kitti_camera_to_lidar(box3d_rect, data_dict['calib'])
            box2d = box_utils.boxes3d_kitti_camera_to_imageboxes(box3d_rect, data_dict['calib'],
                                                                    data_dict['images'].shape[:2])
            sampled_gt_boxes[idx] = box3d_lidar[0]
            sampled_gt_boxes2d[idx] = box2d[0]

        obj_idx = idx * np.ones(len(obj_points), dtype=np.int)

        # copy crops from images
        img_path = self.root_path /  f'training/image_2/{info["image_idx"]}.png'
        raw_image = io.imread(img_path)
        raw_image = raw_image.astype(np.float32)
        raw_center = info['bbox'].reshape(2,2).mean(0)
        new_box = sampled_gt_boxes2d[idx].astype(np.int)
        new_shape = np.array([new_box[2]-new_box[0], new_box[3]-new_box[1]])
        raw_box = np.concatenate([raw_center-new_shape/2, raw_center+new_shape/2]).astype(np.int)
        raw_box[0::2] = np.clip(raw_box[0::2], a_min=0, a_max=raw_image.shape[1])
        raw_box[1::2] = np.clip(raw_box[1::2], a_min=0, a_max=raw_image.shape[0])
        if (raw_box[2]-raw_box[0])!=new_shape[0] or (raw_box[3]-raw_box[1])!=new_shape[1]:
            new_center = new_box.reshape(2,2).mean(0)
            new_shape = np.array([raw_box[2]-raw_box[0], raw_box[3]-raw_box[1]])
            new_box = np.concatenate([new_center-new_shape/2, new_center+new_shape/2]).astype(np.int)

        img_crop2d = raw_image[raw_box[1]:raw_box[3],raw_box[0]:raw_box[2]] / 255

        return new_box, img_crop2d, obj_points, obj_idx

    def sample_gt_boxes_2d_kitti(self, data_dict, sampled_boxes, valid_mask):
        mv_height = None
        # filter out box2d iou > thres
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_boxes, data_dict['road_plane'], data_dict['calib']
            )

        # sampled_boxes2d = np.stack([x['bbox'] for x in sampled_dict], axis=0).astype(np.float32)
        boxes3d_camera = box_utils.boxes3d_lidar_to_kitti_camera(sampled_boxes, data_dict['calib'])
        sampled_boxes2d = box_utils.boxes3d_kitti_camera_to_imageboxes(boxes3d_camera, data_dict['calib'],
                                                                        data_dict['images'].shape[:2])
        sampled_boxes2d = torch.Tensor(sampled_boxes2d)
        existed_boxes2d = torch.Tensor(data_dict['gt_boxes2d'])
        iou2d1 = box_utils.pairwise_iou(sampled_boxes2d, existed_boxes2d).cpu().numpy()
        iou2d2 = box_utils.pairwise_iou(sampled_boxes2d, sampled_boxes2d).cpu().numpy()
        iou2d2[range(sampled_boxes2d.shape[0]), range(sampled_boxes2d.shape[0])] = 0
        iou2d1 = iou2d1 if iou2d1.shape[1] > 0 else iou2d2

        ret_valid_mask = ((iou2d1.max(axis=1)<self.img_aug_iou_thresh) &
                         (iou2d2.max(axis=1)<self.img_aug_iou_thresh) &
                         (valid_mask))

        sampled_boxes2d = sampled_boxes2d[ret_valid_mask].cpu().numpy()
        if mv_height is not None:
            mv_height = mv_height[ret_valid_mask]
        return sampled_boxes2d, mv_height, ret_valid_mask

    def sample_gt_boxes_2d(self, data_dict, sampled_boxes, valid_mask):
        mv_height = None

        if self.img_aug_type == 'kitti':
            sampled_boxes2d, mv_height, ret_valid_mask = self.sample_gt_boxes_2d_kitti(data_dict, sampled_boxes, valid_mask)
        else:
            raise NotImplementedError

        return sampled_boxes2d, mv_height, ret_valid_mask

    def initilize_image_aug_dict(self, data_dict, gt_boxes_mask):
        img_aug_gt_dict = None
        if self.img_aug_type is None:
            pass
        elif self.img_aug_type == 'kitti':
            obj_index_list, crop_boxes2d = [], []
            gt_number = gt_boxes_mask.sum().astype(np.int)
            gt_boxes2d = data_dict['gt_boxes2d'][gt_boxes_mask].astype(np.int)
            gt_crops2d = [data_dict['images'][_x[1]:_x[3],_x[0]:_x[2]] for _x in gt_boxes2d]

            img_aug_gt_dict = {
                'obj_index_list': obj_index_list,
                'gt_crops2d': gt_crops2d,
                'gt_boxes2d': gt_boxes2d,
                'gt_number': gt_number,
                'crop_boxes2d': crop_boxes2d
            }
        else:
            raise NotImplementedError

        return img_aug_gt_dict

    def collect_image_crops(self, img_aug_gt_dict, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx):
        if self.img_aug_type == 'kitti':
            new_box, img_crop2d, obj_points, obj_idx = self.collect_image_crops_kitti(info, data_dict,
                                                    obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx)
            img_aug_gt_dict['crop_boxes2d'].append(new_box)
            img_aug_gt_dict['gt_crops2d'].append(img_crop2d)
            img_aug_gt_dict['obj_index_list'].append(obj_idx)
        else:
            raise NotImplementedError

        return img_aug_gt_dict, obj_points

    def copy_paste_to_image(self, img_aug_gt_dict, data_dict, points):
        if self.img_aug_type == 'kitti':
            obj_points_idx = np.concatenate(img_aug_gt_dict['obj_index_list'], axis=0)
            point_idxes = -1 * np.ones(len(points), dtype=np.int)
            point_idxes[:obj_points_idx.shape[0]] = obj_points_idx

            data_dict['gt_boxes2d'] = np.concatenate([img_aug_gt_dict['gt_boxes2d'], np.array(img_aug_gt_dict['crop_boxes2d'])], axis=0)
            data_dict = self.copy_paste_to_image_kitti(data_dict, img_aug_gt_dict['gt_crops2d'], img_aug_gt_dict['gt_number'], point_idxes)
            if 'road_plane' in data_dict:
                data_dict.pop('road_plane')
        else:
            raise NotImplementedError
        return data_dict

    # 功能: 将gt移动到道路平面上，首先去除原始的gt(这里还会将原始的gt扩张)内的点云，然后将采样的gt点云与背景点点云拼接构造成新的场景
    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict, mv_height=None, sampled_gt_boxes2d=None):
        """
        Args:
            data_dict:  待准备的batch数据
            sampled_gt_boxes:   采样后的gt数据 {ndarray: (N, 7)}
            total_valid_sampled_dict:   采样+当前帧的gt样本 {list: N}
            mv_height:  None
            sampled_gt_boxes2d: None
        """
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]     # box dim=7
        gt_names = data_dict['gt_names'][gt_boxes_mask]     # 类别名称
        points = data_dict['points']    # 点云特征（包含前景点与背景点）

        # 将gt框移动到道路平面上，返回移动的gt和移动距离
        if self.sampler_cfg.get('USE_ROAD_PLANE', False) and mv_height is None:
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('calib')
            data_dict.pop('road_plane')

        obj_points_list = []

        # convert sampled 3D boxes to image plane
        img_aug_gt_dict = self.initilize_image_aug_dict(data_dict, gt_boxes_mask)   # None

        if self.use_shared_memory:  # False
            gt_database_data = SharedArray.attach(f"shm://{self.gt_database_data_key}")
            gt_database_data.setflags(write=0)
        else:
            gt_database_data = None

        # 逐个采样box处理，获取当前采样的所有gt点云位置信息
        for idx, info in enumerate(total_valid_sampled_dict):
            if self.use_shared_memory:
                start_offset, end_offset = info['global_data_offset']
                obj_points = copy.deepcopy(gt_database_data[start_offset:end_offset])
            else:
                file_path = self.root_path / info['path']   # ../data/kitti/gt_database/002873_Car_3.bin

                obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(    # gt的点云特征 (N, 4)
                    [-1, self.sampler_cfg.NUM_POINT_FEATURES])
                if obj_points.shape[0] != info['num_points_in_gt']:
                    obj_points = np.fromfile(str(file_path), dtype=np.float64).reshape(-1, self.sampler_cfg.NUM_POINT_FEATURES)

            assert obj_points.shape[0] == info['num_points_in_gt']
            obj_points[:, :3] += info['box3d_lidar'][:3].astype(np.float32)     # 还原绝对坐标

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]      # 将gt放在道路平面上

            if self.img_aug_type is not None:
                img_aug_gt_dict, obj_points = self.collect_image_crops(
                    img_aug_gt_dict, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx
                )

            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)    # 将所有的点云gt特征进行拼接，即前景点点云
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])  # 采样的gt类别名称

        if self.sampler_cfg.get('FILTER_OBJ_POINTS_BY_TIMESTAMP', False) or obj_points.shape[-1] != points.shape[-1]:   # False
            if self.sampler_cfg.get('FILTER_OBJ_POINTS_BY_TIMESTAMP', False):
                min_time = min(self.sampler_cfg.TIME_RANGE[0], self.sampler_cfg.TIME_RANGE[1])
                max_time = max(self.sampler_cfg.TIME_RANGE[0], self.sampler_cfg.TIME_RANGE[1])
            else:
                assert obj_points.shape[-1] == points.shape[-1] + 1
                # transform multi-frame GT points to single-frame GT points
                min_time = max_time = 0.0

            time_mask = np.logical_and(obj_points[:, -1] < max_time + 1e-6, obj_points[:, -1] > min_time - 1e-6)
            obj_points = obj_points[time_mask]

        # 将采样的box扩大，sampler_cfg.REMOVE_EXTRA_WIDTH即为dx，dy和dz的放大长度
        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )

        # 核心代码：更新当前点云场景信息
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)    # 只保留背景点，去除前景点
        points = np.concatenate([obj_points[:, :points.shape[-1]], points], axis=0)    # 将采样+原始gt点云与放大后保留的背景点拼接，组成新的点云
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)     # 将类别拼接
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)     # 将box拼接

        # 用新的box,类别和点云更新data_dict,其余的数据增强方式也只会保留索引index和这三个部分的数据
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points

        if self.img_aug_type is not None:   # False
            data_dict = self.copy_paste_to_image(img_aug_gt_dict, data_dict, points)

        return data_dict

    # 功能: 在训练集各类别中随机采样与当前帧点云不重叠的gt，作为当前帧额外的独立采样gt，并添加到当前的点云帧场景中，相当于是一个copy paste操作
    #       而挑选不重叠的gt过程的挑选相当于是一个碰撞测试，避免影响到当前点云帧的原始gt，从而实现gt样本的增多，即数据增强的效果
    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Func:
            相当于调用了forward函数
        """
        gt_boxes = data_dict['gt_boxes']     # (N, 7)
        gt_names = data_dict['gt_names'].astype(str)    # (N, )
        existed_boxes = gt_boxes        # 当前点云帧的gt boxes信息
        total_valid_sampled_dict = []   # 全部有效的采样字典
        sampled_mv_height = []
        sampled_gt_boxes2d = []

        # 每个类别依次处理
        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:  # 限制整个点云中box的数量 False
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
            if int(sample_group['sample_num']) > 0:     # 如果采样box的数量大于0
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)  # 在db_infos[calss_name]中随机采样sample_num个box信息，同事更新sample_group信息
                # 整个训练集中随机采样的gt boxes
                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                assert not self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False), 'Please use latest codes to generate GT_DATABASE'

                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])  # 计算随机采样box和当前帧点云box的iou3d
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])  # 计算随机采样box之间的iou3d
                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0  # 这里将矩阵(行列阵)对角线设置为0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2      # 如果当前帧点云gt > 0
                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0)   # 不存在有重叠区域掩码

                if self.img_aug_type is not None:   # False
                    sampled_boxes2d, mv_height, valid_mask = self.sample_gt_boxes_2d(data_dict, sampled_boxes, valid_mask)
                    sampled_gt_boxes2d.append(sampled_boxes2d)
                    if mv_height is not None:
                        sampled_mv_height.append(mv_height)

                valid_mask = valid_mask.nonzero()[0]    # 选出iou为0的索引index，这里随机采样的gt两两之间均分离
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]  # 选取随机采样中相互独立的gt样本
                valid_sampled_boxes = sampled_boxes[valid_mask]             # 选取随机采样中相互独立的gt boxes

                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes[:, :existed_boxes.shape[-1]]), axis=0)   # 将有效采样box和已经存在的box进行拼接
                total_valid_sampled_dict.extend(valid_sampled_dict)     # 将有效的sampled_dict追加到总dict

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]     # 取出采样样本(existed_boxes的前两个样本为当前帧样本，剩余均为采样样本)

        if total_valid_sampled_dict.__len__() > 0:  # 当各类别随机采样样本总数 > 0
            sampled_gt_boxes2d = np.concatenate(sampled_gt_boxes2d, axis=0) if len(sampled_gt_boxes2d) > 0 else None
            sampled_mv_height = np.concatenate(sampled_mv_height, axis=0) if len(sampled_mv_height) > 0 else None

            # 将随机采样的box加入到当前的场景作为数据增强
            data_dict = self.add_sampled_boxes_to_scene(
                data_dict, sampled_gt_boxes, total_valid_sampled_dict, sampled_mv_height, sampled_gt_boxes2d
            )

        data_dict.pop('gt_boxes_mask')  # 去除gt掩码信息
        return data_dict
