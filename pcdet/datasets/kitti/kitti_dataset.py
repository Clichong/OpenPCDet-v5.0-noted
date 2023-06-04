import copy
import pickle

import numpy as np
from skimage import io
from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate

# KITTI数据集：继承了数据集的基类
class KittiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:   根目录路径，不过这里设置
            dataset_cfg: yaml文件中的DATA_CONFIG设定，这里的 pointpillar 继承了kitti_dataset.yaml
            class_names: kitti数据集的三个类别名称
            training:    训练模式设置为True，如果是测试模式则设定为False
            logger:      日志文件处理对象
        """
        # 设定好点云特征编码，点云数据增强以及点云处理3大步骤
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')     # ../data/kitti/training

        # 获取train.txt/test.txt文件下的点云索引内容，strip()是去除每一行后的回车符
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')    # ../data/kitti/ImageSets/train.txt
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = []       # 存储待处理的全部数据信息
        self.include_kitti_data(self.mode)     # 根据INFO_PATH获取待处理的全部数据信息

    # 功能: 根据设定的pkl文件获取待处理的全部数据信息
    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')

        # 这里的'train'列表只包含了一个kitti_infos_train.pkl，还可以添加额外的val.pkl
        kitti_infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path      # ../data/kitti/kitti_infos_train.pkl
            if not info_path.exists():  # 在构建create_kitti_infos时pkl文件还不存在，kitti_infos设置为空
                continue

            # 读取pkl序列文件的内容，每个对象是一个字典，包含4个key：point_cloud, image, calib, annos
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)   # 用新列表扩展原来的列表，在列表末尾一次性追加另一个序列中的多个值（可追加多个序列对象）

        self.kitti_infos.extend(kitti_infos)    # 根据mode确认存储全部的train/test对象

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

    # 功能：更具split获取 data/kitti/ImageSets 路径下对应的txt文件内容，构建好sample_id_list，也就是训练集相应的索引
    def set_split(self, split):
        super().__init__(       # 这一步感觉没有改变
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split  # train / test / val
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')    # /home/lab/LLC/PointCloud/OpenPCDet/data/kitti/training
        # 选择training、val、testing的其中一个文件
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')    # /home/lab/LLC/PointCloud/OpenPCDet/data/kitti/ImageSets/train.txt

        # 获取txt文件中的内容，这里就是最后目的
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    # 功能: 根据帧号读取相对于的点云bin文件
    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)    # 读取bin文件

    # 功能：根据帧号读取相对应的image图像，并转换到0-1区间
    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    # 功能: 返回帧号对应图像的尺寸[H x W]
    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    # 功能：根据label的txt信息将每个object构建成一个类，所以这里返回的是列表，每个列表对象就是一个gt的object
    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx: str, Sample index
        Returns:
            depth: (H, W), Depth map
        """
        depth_file = self.root_split_path / 'depth_2' / ('%s.png' % idx)
        assert depth_file.exists()
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        return depth

    # 功能: 根据index索引返回其坐标系转换的类
    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)  # 校准txt文件路径
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)    # 实现各坐标系的转换

    # 功能: 返回道路信息
    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    # 功能: 返回符合FOV视角的点云布尔变量列表
    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect: 在rect系下的点云
            img_shape: 图像的尺寸
            calib: 标定信息

        Returns:
            布尔变量列表 if the point in the fov
        """
        # 由lidar转到了img
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)

        # 判断投影点是否在图像范围内 logical_and是逻辑与操作
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)

        # 深度 > 0, 才可以判断在fov视角
        # pts_valid_flag=array([ True,   True,  True, False,   True, True,.....])之类的，一共有M个
        # 用于判断该点云能否有效 （是否用于训练）
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    # 创建pkl文件与gt_database时用到
    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        """
        根据split确定好划分数据的index列表，简单来说就是提取ImageSet中的txt文件内容，然后根据索引提取当前点云帧的信息
        """
        import concurrent.futures as futures

        # 线程函数
        def process_single_scene(sample_idx):
            """
            线程函数，sample_idx就是当前需要整合其信息的点云帧场景
            """
            print('%s sample_idx: %s' % (self.split, sample_idx))
            # 存储当前点云帧索引的全部信息，包含point_cloud、image、calib、annos四个部分
            info = {}

            # 点云信息：点云特征维度和索引
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            # 图像信息：索引和图像高宽
            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)} # 返回图像尺寸
            info['image'] = image_info

            # 根据索引获取Calibration对象
            calib = self.get_calib(sample_idx)
            # 1）内参矩阵P2：(3,4)， 在最后一行进行填充，变成(4,4)矩阵
            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            # 2）校准矩阵R0：(3,3)，用0补充一行一列，构建成(4,4)矩阵，然在R0[3,3]赋值为1
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            # 3）外参矩阵T2V：(3,4)，也是在最后一行进行填充，变成(4,4)矩阵
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            # 构建标定信息：P2、R0_rect和Tr_velo_to_cam矩阵
            """
                相机坐标系 = 内参矩阵 * 校准矩阵 * 外参矩阵 * 点云坐标系
                y = P2 * R0 * Tr_velo_to_cam * x
            """
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}
            info['calib'] = calib_info

            # 构建训练集和验证集时执行（有标签）、构建测试集不执行因为没有标签
            if has_label:
                obj_list = self.get_label(sample_idx)   # 根据index到label中获取标注信息，列表的每个对象就是一个gt的object
                annotations = {}
                # 根据属性将所有obj_list的属性添加进annotations，annotations的每个values都是列表
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                # 除去'DontCare'类别的有效物体数，如10个，object除去“DontCare”4个，还剩num_objects6个
                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])  # 有效object数量（6）
                num_gt = len(annotations['name'])    # 总标注object数量（10）
                # 用-1来表示无效object，有效object正常编号，由此可以得到 index=[0,1,2,3,4,5,-1,-1,-1,-1]
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                # 假设有效物体的个数是N， 取有效物体的 location（N,3）、dimensions（N,3）、rotation_y（N,1）信息
                # kitti中'DontCare'一定放在最后,所以可以这样取值
                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                # 通过计算得到在lidar坐标系下的坐标，loc_lidar:（N,3）
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]      # dim：l/h/w
                loc_lidar[:, 2] += h[:, 0] / 2      # 将物体的坐标原点由物体底部中心移到物体中心

                # (N, 7) [x, y, z, dx, dy, dz, heading]
                # np.newaxis在列上增加一维，因为rots是(N,),相当于是[..., None]进行增维，否者无法进行拼接处理
                # -(np.pi / 2 + rots[..., np.newaxis]) 应为在kitti中，camera坐标系下定义物体朝向与camera的x轴夹角顺时针为正，逆时针为负
                # 在pcdet中，lidar坐标系下定义物体朝向与lidar的x轴夹角逆时针为正，顺时针为负，所以二者本身就正负相反
                # pi / 2是坐标系x轴相差的角度(如图所示)
                # camera:         lidar:
                # Y                    X
                # |                    |
                # |____X         Y_____|
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                # 是否统计每个gt boxes内的点云数量
                if count_inside_pts:
                    points = self.get_lidar(sample_idx)   # 根据索引获取点云 [N, 4]
                    calib = self.get_calib(sample_idx)    # 根据索引获取Calibration对象
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])      # 将lidar坐标系的点变换到rect坐标系

                    # 筛选出FOV视角下的点云，在数据增强的处理中还会再进行一次筛选
                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]  # 根据索引提取有效点

                    # gt_boxes_lidar是(N,7)  [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
                    # 返回值corners_lidar为（N,8,3）
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)

                    # num_gt是这一帧图像里物体的总个数，假设为10，
                    # 则num_points_in_gt=array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=int32)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                    for k in range(num_objects):    # 有效object数量
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])   # 判断点云是否在第k各gt bbox中，使用了scipy的库函数
                        num_points_in_gt[k] = flag.sum()    # 统计当前第k各gt box的点数量
                    annotations['num_points_in_gt'] = num_points_in_gt  # 添加框内点云数量信息

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list  # 对应数据集txt文件的索引
        # 创建线程池，多线程异步处理，增加处理速度
        with futures.ThreadPoolExecutor(num_workers) as executor:   # 多线程
            infos = executor.map(process_single_scene, sample_id_list)  # 对索引列表中依次对每个索引进行process_single_scene处理

        # infos是一个列表，每一个元素代表了一帧的信息（字典）
        return list(infos)

    # 用trainfile的groundtruth产生groundtruth_database，只保存训练数据中的gt_box及其包围的点的信息，用于数据增强
    # 执行语句是：python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        # 保存路径
        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))   # /data/kitti/gt_database
        db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s.pkl' % split)     # kitti_dbinfos_train.pkl

        database_save_path.mkdir(parents=True, exist_ok=True)   # 多级目录创建，如存在不报错
        all_db_infos = {}

        # 传入的参数info_path是一个.pkl文件，ROOT_DIR/data/kitti/kitti_infos_train.pkl
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        # 顺序处理每个整合后的点云帧场景信息
        for k in range(len(infos)):     # len(infos): 3712
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]     # 当前点云帧信息
            sample_idx = info['point_cloud']['lidar_idx']   # 当前的索引id
            points = self.get_lidar(sample_idx)     # 点 [N, 4]
            annos = info['annos']   # 标注信息
            names = annos['name']   # 表示当前帧里面的所有物体objects
            difficulty = annos['difficulty']    # 每个object的难度
            bbox = annos['bbox']    # 2d标注框
            gt_boxes = annos['gt_boxes_lidar']  # gt信息矩阵 [N, 7]

            num_obj = gt_boxes.shape[0]     # 有效object数量
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(    # 返回每个box中的点云索引[0 0 0 1 0 1 1...]，这里与in_hull方法有点类似
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            # 对每个有效的object进行信息存储，一方面gt内的点信息保存在bin文件中，同时构建其info字典保存在相应类别列表中
            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)   # '000000_Pedestrian_0.bin'
                filepath = database_save_path / filename    # 存放'000000_Pedestrian_0.bin'的绝对路径
                gt_points = points[point_indices[i] > 0]    # 只保留在gt内的点，进行筛选

                gt_points[:, :3] -= gt_boxes[i, :3]     # 将第i个box内点转化为局部坐标
                with open(filepath, 'w') as f:      # 把gt_points的信息写入文件里
                    gt_points.tofile(f)

                # 类别是否被选择检测，一般常用的类别是['Car', 'Pedestrian', 'Cyclist']，其他类别可能会被忽略处理
                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))
                    # 根据当前物体的信息组成info
                    db_info = {'name': names[i],
                               'path': db_path,             # gt_database/xxxxx.bin
                               'image_idx': sample_idx,     # 当前点云帧index，多个gt可能共享一个index
                               'gt_idx': i,                 # gt编号
                               'box3d_lidar': gt_boxes[i],  # gt信息 (7, ) [xyz, dxdydz, heading]
                               'num_points_in_gt': gt_points.shape[0],  # gt内的点数
                               'difficulty': difficulty[i],
                               'bbox': bbox[i],     # 在图像上的2d标注框，在标注文件中可以获取 (4, )
                               'score': annos['score'][i]
                    }
                    # 对每个类别进行gt汇总
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)  # 如果存在该类别则追加
                    else:
                        all_db_infos[names[i]] = [db_info]      # 如果不存在该类别则新增

        # 打印每个类别的gt数量
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        # 将序列信息保存，包含了每个类别的gt整合信息
        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs
        # 等于返回训练帧的总个数，等于图片的总个数，帧的总个数
        return len(self.kitti_infos)

    def __getitem__(self, index):
        # index = 1331 单卡训练时打算顺序shuffle=True
        if self._merge_all_iters_to_one_epoch:  # False
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])   # 根据index索引获取相对应的点云信息

        sample_idx = info['point_cloud']['lidar_idx']   # 点云文件名称
        img_shape = info['image']['image_shape']    # 图像的尺寸 [375 1242]
        calib = self.get_calib(sample_idx)          # 返回的是一个类,可以实现各类坐标系的转换
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])   # 获取id列表

        # 定义输入数据的字典包含帧id和标定信息
        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:     # 点云的标注信息 True
            annos = info['annos']
            # 下面函数的作用是 在info中剔除包含'DontCare'的数据信息
            # 不但从name中剔除，余下的location、dimensions等信息也都不考虑在内
            annos = common_utils.drop_info_with_name(annos, name='DontCare')    # 去除name类(Dontcare)

            # 得到有效物体object(N个)的位置、大小和角度信息（N,3）,(N,3),(N)
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']   # 位置、尺寸、旋转方向
            gt_names = annos['name']

            # 构造camera系下的label（N,7），再转换到lidar系下
            # boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
            # boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)  # (nums, 7)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)   # rect_to_lidar (nums, 7)

            # 将新的键值对 添加到输入的字典中去，此时输入中有四个键值对了
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

            # 如果get_item_list中有gt_boxes2d，则将bbox加入到input_dict中
            if "gt_boxes2d" in get_item_list:   # False
                input_dict['gt_boxes2d'] = annos["bbox"]

            # 如果有路面信息，则加入进去
            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        # 加入点云，如果要求FOV视角，则对点云进行裁剪后加入input_dict
        if "points" in get_item_list:   # True
            points = self.get_lidar(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:    # True
                pts_rect = calib.lidar_to_rect(points[:, 0:3])      # rect坐标系下的点云
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)    # 返回针对每个点云的布尔变量
                points = points[fov_flag]     # 过滤出FOV视角点云
            input_dict['points'] = points

        # 加入图片信息
        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        # 加入深度图
        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        # 加入标定信息
        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        # 将输入数据送入prepare_data进一步处理，形成训练数据
        input_dict['calib'] = calib
        data_dict = self.prepare_data(data_dict=input_dict)     # DatasetTemplate中具体实现，进行具体的数据增强与数据处理实现

        # 加入图片宽高信息
        data_dict['image_shape'] = img_shape
        return data_dict

# 功能：进行kitti数据集的数据准备,构建gt_database与各类的pkl标注信息,
# 执行语句是：python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)

    # 设置pkl文件的名称和保存路径，一般保存在 data/kitti/ 目录下
    train_split, val_split = 'train', 'val'
    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
    val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'kitti_infos_trainval.pkl'
    test_filename = save_path / 'kitti_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    # 构建kitti_infos_train.pkl文件，对训练集的数据进行信息统计并保存
    dataset.set_split(train_split)  # 获取ImageSet/training.txt划分好的点云帧索引
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename)
    
    # 构建kitti_infos_val.pkl文件，对验证集的数据进行信息统计并保存
    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)

    # 构建kitti_infos_trainval.pkl文件，将训练集和验证集的信息合并写到一个文件里
    with open(trainval_filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    print('Kitti info trainval file is saved to %s' % trainval_filename)

    # 构建kitti_infos_test.pkl文件，对测试集的数据进行信息统计并保存
    dataset.set_split('test')
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')

    # 构建kitti_dbinfos_train.pkl文件，只保存训练数据中的gt_box及其包围的点的信息，用于数据增强
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()    # OpenPCDet根目录,resolve作用是返回的是当前的文件的绝对路径
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti',  # OpenPCDet/data/kitti
            save_path=ROOT_DIR / 'data' / 'kitti'   # OpenPCDet/data/kitti
        )
