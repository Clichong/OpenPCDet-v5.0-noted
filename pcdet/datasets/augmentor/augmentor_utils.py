import numpy as np
import math
import copy
from ...utils import common_utils
from ...utils import box_utils


# 功能: 沿x轴进行选择随机反转
def random_flip_along_x(gt_boxes, points, return_flip=False, enable=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    if enable is None:
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])   # 一半的概率选择是否翻转
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]    # y坐标翻转
        gt_boxes[:, 6] = -gt_boxes[:, 6]    # 方位角翻转，直接取负数，因为方位角定义为与x轴的夹角（这里按照顺时针的方向取角度）
        points[:, 1] = -points[:, 1]        # 点云y坐标翻转
        
        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]    # 如果有速度，y方向速度翻转
    if return_flip:
        return gt_boxes, points, enable
    return gt_boxes, points


# 功能: 沿y轴进行选择随机反转
def random_flip_along_y(gt_boxes, points, return_flip=False, enable=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    if enable is None:
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])   # 一半的概率选择是否翻转
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]    # x坐标翻转
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)  # 方位角加pi后，取负数（这里按照顺时针的方向取角度）
        points[:, 0] = -points[:, 0]        # 点云x坐标取反

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]    # 如果有速度，x方向速度取反
    if return_flip:
        return gt_boxes, points, enable
    return gt_boxes, points

# 功能: 对点云和box进行整体随机旋转
def global_rotation(gt_boxes, points, rot_range, return_rot=False, noise_rotation=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    if noise_rotation is None:  # 在均匀分布中随机产生旋转角度
        noise_rotation = np.random.uniform(rot_range[0], rot_range[1])

    # 沿z轴旋转noise_rotation弧度，这里之所以取第0个，是因为rotate_points_along_z对batch进行处理，而这里仅处理单个点云
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    # 同样对box的坐标进行旋转
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation    # 对box的方位角进行累加

    if gt_boxes.shape[1] > 7:   # 对速度进行旋转，由于速度仅有x和y两个维度，所以补出第三维度，增加batch维度后进行旋转
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    if return_rot:
        return gt_boxes, points, noise_rotation
    return gt_boxes, points

# 功能: 对点云和box进行整体随机缩放
def global_scaling(gt_boxes, points, scale_range, return_scale=False):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:  # 如果缩放的尺度过小，则直接返回原来的box和点云
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1]) # 在缩放范围内随机产生缩放尺度
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale  # [:, :6]表示xyz，dxdydz均进行缩放
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:] *= noise_scale
        
    if return_scale:
        return gt_boxes, points, noise_scale
    return gt_boxes, points

def global_scaling_with_roi_boxes(gt_boxes, roi_boxes, points, scale_range, return_scale=False):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    roi_boxes[:,:, [0,1,2,3,4,5,7,8]] *= noise_scale
    if return_scale:
        return gt_boxes,roi_boxes, points, noise_scale
    return gt_boxes, roi_boxes, points


def random_image_flip_horizontal(image, depth_map, gt_boxes, calib):
    """
    Performs random horizontal flip augmentation
    Args:
        image: (H_image, W_image, 3), Image
        depth_map: (H_depth, W_depth), Depth map
        gt_boxes: (N, 7), 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
        calib: calibration.Calibration, Calibration object
    Returns:
        aug_image: (H_image, W_image, 3), Augmented image
        aug_depth_map: (H_depth, W_depth), Augmented depth map
        aug_gt_boxes: (N, 7), Augmented 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
    """
    # Randomly augment with 50% chance
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])

    if enable:
        # Flip images
        aug_image = np.fliplr(image)
        aug_depth_map = np.fliplr(depth_map)
        
        # Flip 3D gt_boxes by flipping the centroids in image space
        aug_gt_boxes = copy.copy(gt_boxes)
        locations = aug_gt_boxes[:, :3]
        img_pts, img_depth = calib.lidar_to_img(locations)
        W = image.shape[1]
        img_pts[:, 0] = W - img_pts[:, 0]
        pts_rect = calib.img_to_rect(u=img_pts[:, 0], v=img_pts[:, 1], depth_rect=img_depth)
        pts_lidar = calib.rect_to_lidar(pts_rect)
        aug_gt_boxes[:, :3] = pts_lidar
        aug_gt_boxes[:, 6] = -1 * aug_gt_boxes[:, 6]

    else:
        aug_image = image
        aug_depth_map = depth_map
        aug_gt_boxes = gt_boxes

    return aug_image, aug_depth_map, aug_gt_boxes

# 对gt在x轴上进随机平移
def random_local_translation_along_x(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]  某个点云帧场景的点云组成（此时还是对单帧点云处理，还没有进行batch数据合并）
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    # augs = {}
    # 依次对每个gt进行处理
    for idx, box in enumerate(gt_boxes):
        # 这里由于限制了是在FOV视角，所以平移可以是>0，否则<0的gt后面也会被过滤
        offset = np.random.uniform(offset_range[0], offset_range[1])    # 从一个范围内产生均匀分布的随机值
        # augs[f'object_{idx}'] = offset
        points_in_box, mask = get_points_in_box(points, box)    # 找到在当前gt中的点索引值
        # 对gt的内进行平移，同时也要更改gt的信息，这里只对x轴进行操作 所以只是操作dim=0
        points[mask, 0] += offset
        gt_boxes[idx, 0] += offset
    
        # if gt_boxes.shape[1] > 7:
        #     gt_boxes[idx, 7] += offset
    
    return gt_boxes, points

# 对gt在y轴上进随机平移
def random_local_translation_along_y(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    # augs = {}
    # 同样的方法，对每个gt的点在y轴进行随机平移，更改gt点的位置信息以及gt的对应信息
    for idx, box in enumerate(gt_boxes):
        offset = np.random.uniform(offset_range[0], offset_range[1])
        # augs[f'object_{idx}'] = offset
        points_in_box, mask = get_points_in_box(points, box)
        points[mask, 1] += offset
        
        gt_boxes[idx, 1] += offset
    
        # if gt_boxes.shape[1] > 8:
        #     gt_boxes[idx, 8] += offset
    
    return gt_boxes, points

# 对gt在z轴上进随机平移
def random_local_translation_along_z(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        offset = np.random.uniform(offset_range[0], offset_range[1])    # z轴的范围是-3~1，设定数值不能太大
        # augs[f'object_{idx}'] = offset
        points_in_box, mask = get_points_in_box(points, box)
        points[mask, 2] += offset
        
        gt_boxes[idx, 2] += offset
    
    return gt_boxes, points


# 筛选z轴过高的点和gt
def global_frustum_dropout_top(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])   # 均匀分布中选取一个随机数
    # threshold = max - length * uniform(0 ~ 0.2)
    threshold = np.max(points[:, 2]) - intensity * (np.max(points[:, 2]) - np.min(points[:, 2]))    # 设置一个pillar高度的阈值
    # 只保留在这个高度阈值之下的点和gt
    points = points[points[:, 2] < threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 2] < threshold]
    return gt_boxes, points


# 筛选z轴过低的点和gt
def global_frustum_dropout_bottom(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])   # 0-0.2
    
    threshold = np.min(points[:, 2]) + intensity * (np.max(points[:, 2]) - np.min(points[:, 2]))   # 设置一个pillar底部的阈值
    # 只保留在阈值之上的点和gt
    points = points[points[:, 2] > threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 2] > threshold]
    
    return gt_boxes, points


# 筛选y轴太左的点和gt（kitti的数据集前向是x轴的正方向，左向是y轴的正方向）
def global_frustum_dropout_left(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])
    
    threshold = np.max(points[:, 1]) - intensity * (np.max(points[:, 1]) - np.min(points[:, 1]))    # 左边y轴为正，所以是max
    points = points[points[:, 1] < threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 1] < threshold]
    
    return gt_boxes, points


# 筛选y轴太右的点和gt（kitti的数据集前向是x轴的正方向，左向是y轴的正方向）
def global_frustum_dropout_right(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])
    
    threshold = np.min(points[:, 1]) + intensity * (np.max(points[:, 1]) - np.min(points[:, 1]))    # 右边y轴为负，所以是min
    points = points[points[:, 1] > threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 1] > threshold]
    
    return gt_boxes, points


# 局部随机缩放
def local_scaling(gt_boxes, points, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:  # 缩放范围太小则不作操作
        return gt_boxes, points
    
    # augs = {}
    # 依次对每个gt进行随机缩放
    for idx, box in enumerate(gt_boxes):
        noise_scale = np.random.uniform(scale_range[0], scale_range[1])
        # augs[f'object_{idx}'] = noise_scale
        points_in_box, mask = get_points_in_box(points, box)
        
        # tranlation to axis center 对gt内的点构造局部坐标系
        points[mask, 0] -= box[0]
        points[mask, 1] -= box[1]
        points[mask, 2] -= box[2]
        
        # apply scaling 在局部坐标系中进行缩放
        points[mask, :3] *= noise_scale
        
        # tranlation back to original position 在恢复到点云坐标系
        points[mask, 0] += box[0]
        points[mask, 1] += box[1]
        points[mask, 2] += box[2]

        # 对gt的尺寸进行更新
        gt_boxes[idx, 3:6] *= noise_scale
    return gt_boxes, points


# 局部随机旋转
def local_rotation(gt_boxes, points, rot_range):
    """
    局部旋转
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    # augs = {}
    # 依次对每个gt进行随机沿z轴旋转处理
    for idx, box in enumerate(gt_boxes):
        noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
        # augs[f'object_{idx}'] = noise_rotation
        # 获取当前在gt内的点索引
        points_in_box, mask = get_points_in_box(points, box)

        # gt box xyz position 以gt的中心为原始点进行旋转
        centroid_x = box[0]
        centroid_y = box[1]
        centroid_z = box[2]
        
        # tranlation to axis center 构建以gt中心的局部坐标系
        points[mask, 0] -= centroid_x
        points[mask, 1] -= centroid_y
        points[mask, 2] -= centroid_z
        box[0] -= centroid_x
        box[1] -= centroid_y
        box[2] -= centroid_z
        
        # apply rotation 对局部坐标系在z轴进行旋转，更新点位置以及gt位置信息
        # ps: 这里rotate_points_along_z函数传入的points参数是(B, N, 3 + C)维度，所以需要增加维度以及注意[0]选择
        points[mask, :] = common_utils.rotate_points_along_z(points[np.newaxis, mask, :], np.array([noise_rotation]))[0]
        box[0:3] = common_utils.rotate_points_along_z(box[np.newaxis, np.newaxis, 0:3], np.array([noise_rotation]))[0][0]   # 这一步好像是多余的，原点中心旋转还是原点中心 (0,0,0)
        
        # tranlation back to original position 重新从局部坐标系转为点云场景坐标系
        points[mask, 0] += centroid_x
        points[mask, 1] += centroid_y
        points[mask, 2] += centroid_z
        box[0] += centroid_x
        box[1] += centroid_y
        box[2] += centroid_z

        # 旋转方向直接相加即可
        gt_boxes[idx, 6] += noise_rotation

        # 这里gt一般只有7个维度，没有速度信息
        if gt_boxes.shape[1] > 8:   # False
            gt_boxes[idx, 7:9] = common_utils.rotate_points_along_z(
                np.hstack((gt_boxes[idx, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
                np.array([noise_rotation])
            )[0][:, 0:2]
    
    return gt_boxes, points


# 筛选在gt box内且z轴位置超过阈值的点（一般只会删除少量的几百个点）
def local_frustum_dropout_top(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]    # 获取每个gt信息
        
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)    # 获取当前gt内的点
        threshold = (z + dz / 2) - intensity * dz   # 所获得的阈值一般比z高
        
        points = points[np.logical_not(np.logical_and(mask, points[:, 2] >= threshold))]   # 筛选出在gt内且z轴位置超过阈值的点
    
    return gt_boxes, points


# 筛选在gt box内且z轴位置低于阈值的点
def local_frustum_dropout_bottom(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]
        
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)
        threshold = (z - dz / 2) + intensity * dz   # 所获得的阈值一般比z小
        
        points = points[np.logical_not(np.logical_and(mask, points[:, 2] <= threshold))]    # 筛选在gt box内且z轴位置低于阈值的点
    
    return gt_boxes, points


# 筛选在gt box内且y轴位置超过阈值的点
def local_frustum_dropout_left(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]
        
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)
        threshold = (y + dy / 2) - intensity * dy   # 控制阈值在合理范围内，一般不会变化太大
        
        points = points[np.logical_not(np.logical_and(mask, points[:, 1] >= threshold))]    # 超过阈值筛选
    
    return gt_boxes, points


# 筛选在gt box内且y轴位置低于阈值的点
def local_frustum_dropout_right(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]
        
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)
        threshold = (y - dy / 2) + intensity * dy
        
        points = points[np.logical_not(np.logical_and(mask, points[:, 1] <= threshold))]    # 低于阈值筛选
    
    return gt_boxes, points


# 获取在gt内的点和掩码
def get_points_in_box(points, gt_box):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    cx, cy, cz = gt_box[0], gt_box[1], gt_box[2]
    dx, dy, dz, rz = gt_box[3], gt_box[4], gt_box[5], gt_box[6]
    shift_x, shift_y, shift_z = x - cx, y - cy, z - cz
    
    MARGIN = 1e-1
    cosa, sina = math.cos(-rz), math.sin(-rz)
    local_x = shift_x * cosa + shift_y * (-sina)
    local_y = shift_x * sina + shift_y * cosa
    
    mask = np.logical_and(abs(shift_z) <= dz / 2.0, 
                          np.logical_and(abs(local_x) <= dx / 2.0 + MARGIN, 
                                         abs(local_y) <= dy / 2.0 + MARGIN))
    
    points = points[mask]
    
    return points, mask


def get_pyramids(boxes):
    # 手动整理出金字塔顺序，每个列表表示一个面，然后和中心点来构建出一个金字塔
    pyramid_orders = np.array([
        [0, 1, 5, 4],
        [4, 5, 6, 7],
        [7, 6, 2, 3],
        [3, 2, 1, 0],
        [1, 2, 6, 5],
        [0, 4, 7, 3]
    ])
    boxes_corners = box_utils.boxes_to_corners_3d(boxes).reshape(-1, 24)    # (27, 7) -> (27, 8, 3) -> (27, 24)
    
    pyramid_list = []
    # 遍历每个金字塔（list结构，表示4个角点）
    for order in pyramid_orders:
        # frustum polygon: 5 corners, 5 surfaces 截锥体多边形:5个角，5个面
        # 将中心点和四个角点坐标concat在一起
        pyramid = np.concatenate((
            boxes[:, 0:3],      # gt中心点坐标
            boxes_corners[:, 3 * order[0]: 3 * order[0] + 3],   # 金字塔第一个角点坐标
            boxes_corners[:, 3 * order[1]: 3 * order[1] + 3],   # 金字塔第二个角点坐标
            boxes_corners[:, 3 * order[2]: 3 * order[2] + 3],   # 金字塔第三个角点坐标
            boxes_corners[:, 3 * order[3]: 3 * order[3] + 3]),  # 金子塔第四个角点坐标
            axis=1)     # (27, 15)
        pyramid_list.append(pyramid[:, None, :])    # (27, 15) -> (27, 1, 15)
    pyramids = np.concatenate(pyramid_list, axis=1)  # (27, 6, 15), 15=5*3
    return pyramids


# 构建独热编码：4 -> [0,0,0,0,1,0]
def one_hot(x, num_class=1):
    if num_class is None:
        num_class = 1
    ohx = np.zeros((len(x), num_class))     # (N, 6)
    ohx[range(len(x)), x] = 1       # 在每行的对应列赋值为1 (太妙了)
    return ohx


# 判断点是否在金字塔区域中
def points_in_pyramids_mask(points, pyramids):
    """
    Args:
        points:   (N, 4)
        pyramids: (k, 5, 3)
    """
    pyramids = pyramids.reshape(-1, 5, 3)   # (k, 5, 3)
    flags = np.zeros((points.shape[0], pyramids.shape[0]), dtype=np.bool)   # (N, k)
    # 依次对每个金字塔区域进行处理，判断每个点是否在这些金字塔中，在对应的金字塔结构中的位置为True
    for i, pyramid in enumerate(pyramids):
        flags[:, i] = np.logical_or(flags[:, i], box_utils.in_hull(points[:, 0:3], pyramid))    # 点是否在当前的金字塔中
    return flags


# 随机丢弃金字塔点
def local_pyramid_dropout(gt_boxes, points, dropout_prob, pyramids=None):
    """
    Args:
        gt_boxes:       (N, 7)
        points:         (k, 4)
        dropout_prob:   随机丢弃概率
        pyramids:       (34, 6, 5, 3) each six surface of boxes: [num_boxes, 6, 15=3*5]
    """
    if pyramids is None:
        pyramids = get_pyramids(gt_boxes).reshape([-1, 6, 5, 3])    # (N, 6, 15) -> (N, 6, 5, 3)
    # 1. 随机确定每个gt操作的金字塔，确定每个gt选取金字塔的操作概率，筛选出满足这两项的金字塔区域
    drop_pyramid_indices = np.random.randint(0, 6, (pyramids.shape[0]))     # (N, ) 范围大小在[0, 6)之间
    drop_pyramid_one_hot = one_hot(drop_pyramid_indices, num_class=6)       # (N, 6) 根据索引构建独热编码
    drop_box_mask = np.random.uniform(0, 1, (pyramids.shape[0])) <= dropout_prob    # (N, ) 根据随机概率构造掩码
    if np.sum(drop_box_mask) != 0:
        # np.tile是重复函数， [1, 6]是重复6列
        drop_pyramid_mask = (np.tile(drop_box_mask[:, None], [1, 6]) * drop_pyramid_one_hot) > 0    # (N, 6) 随机选出的丢弃布尔变量矩阵
        drop_pyramids = pyramids[drop_pyramid_mask]     # (9, 5, 3) 确认每个gt随机确定丢弃的金字塔部分,5个点确认一个区域

    # 2. 确定了待操作的金字塔区域就对在这些区域里的点进行点丢弃
        point_masks = points_in_pyramids_mask(points, drop_pyramids)    # (k, 9) 判断点是否在每一个金字塔的总结构中
        points = points[np.logical_not(point_masks.any(-1))]    # 只要点在任意一个金字塔结构中都进行过滤
    # print(drop_box_mask)

    # 3. 更新剩下待操作的gt
    pyramids = pyramids[np.logical_not(drop_box_mask)]   # (34, 6, 5, 3) -> (25, 6, 5, 3) 保留筛选后的金字塔区域，筛选了9个区域
    return gt_boxes, points, pyramids


# 随机稀疏金字塔点
def local_pyramid_sparsify(gt_boxes, points, prob, max_num_pts, pyramids=None):
    """
        Args:
        gt_boxes:       (N, 7)
        points:         (k, 4)
        dropout_prob:   随机丢弃概率
        pyramids:       (25, 6, 5, 3) each six surface of boxes: [num_boxes, 6, 15=3*5]
    """
    if pyramids is None:
        pyramids = get_pyramids(gt_boxes).reshape([-1, 6, 5, 3])  # each six surface of boxes: [num_boxes, 6, 15=3*5]
    # 1. 随机确定每个gt操作的金字塔，确定每个gt选取金字塔的操作概率，筛选出满足这两项的金字塔区域（稀疏区域）
    if pyramids.shape[0] > 0:
        sparsity_prob, sparsity_num = prob, max_num_pts
        sparsify_pyramid_indices = np.random.randint(0, 6, (pyramids.shape[0]))     # 随机选取每个gt的一个金字塔区域
        sparsify_pyramid_one_hot = one_hot(sparsify_pyramid_indices, num_class=6)   # 构早成one-hot形式
        sparsify_box_mask = np.random.uniform(0, 1, (pyramids.shape[0])) <= sparsity_prob   # 随机确定对选取的金字塔结构是否进行稀疏
        sparsify_pyramid_mask = (np.tile(sparsify_box_mask[:, None], [1, 6]) * sparsify_pyramid_one_hot) > 0    # 对选取的金字塔结构同时确认稀疏的金字塔进行操作
        # print(sparsify_box_mask)
        
        pyramid_sampled = pyramids[sparsify_pyramid_mask]  # (4, 5, 3) 确认需要稀疏化的金字塔区域
        # print(pyramid_sampled.shape)

    # 2. 获得多个稀疏金字塔后，只对超过50个点的区域进行稀疏化，依次对每个待稀疏的金字塔区域点进行随机选取稀疏化
        pyramid_sampled_point_masks = points_in_pyramids_mask(points, pyramid_sampled)  # 判断每个点是否在确认的每一个金字塔区域内 (N, 4)
        pyramid_sampled_points_num = pyramid_sampled_point_masks.sum(0)  # [ 15 132   0   9] the number of points in each surface pyramid
        valid_pyramid_sampled_mask = pyramid_sampled_points_num > sparsity_num  # only much than sparsity_num should be sparse
        
        sparsify_pyramids = pyramid_sampled[valid_pyramid_sampled_mask]     # 只对多余50个点的金字塔区域进行稀疏化，这里只有一个区域满足
        if sparsify_pyramids.shape[0] > 0:
            # 2.1 获取每个满足50点的稀疏区域点
            point_masks = pyramid_sampled_point_masks[:, valid_pyramid_sampled_mask]    # (N, 4) -> (N, 1)
            remain_points = points[     # (24583, 4) -> (24451, 4)
                np.logical_not(point_masks.any(-1))]  # points which outside the down sampling pyramid
            to_sparsify_points = [points[point_masks[:, i]] for i in range(point_masks.shape[1])]   # 依次获取每个待稀疏金字塔区域的点

            # 2.2 随机选取点实现稀疏化
            sparsified_points = []
            for sample in to_sparsify_points:
                sampled_indices = np.random.choice(sample.shape[0], size=sparsity_num, replace=False)   # 最大随机选取50个点索引
                sparsified_points.append(sample[sampled_indices])   # 根据索引进行筛选点
            sparsified_points = np.concatenate(sparsified_points, axis=0)   # 将每个稀疏后的区域点进行拼接

    # 3. 稀疏区域外的点与稀疏后的点重新拼接成处理后的点，同时更新剩下待操作的gt
            points = np.concatenate([remain_points, sparsified_points], axis=0)     # 金字塔区域之外的点 + 稀疏后的区域点 = 新场景点云
        pyramids = pyramids[np.logical_not(sparsify_box_mask)]   # (25, 6, 5, 3) -> (24, 6, 5, 3) 保留筛选后的金字塔区域，筛选了1个区域
    return gt_boxes, points, pyramids


# 随机交换两两金字塔点
def local_pyramid_swap(gt_boxes, points, prob, max_num_pts, pyramids=None):
    """
        Args:
        gt_boxes:       (N, 7)
        points:         (k, 4)
        dropout_prob:   随机丢弃概率
        pyramids:       (24, 6, 5, 3) each six surface of boxes: [num_boxes, 6, 15=3*5]
    """

    # 获取点在金字塔结构中的比例
    def get_points_ratio(points, pyramid):
        surface_center = (pyramid[3:6] + pyramid[6:9] + pyramid[9:12] + pyramid[12:]) / 4.0   # 面中心位置
        vector_0, vector_1, vector_2 = pyramid[6:9] - pyramid[3:6], pyramid[12:] - pyramid[3:6], pyramid[0:3] - surface_center
        alphas = ((points[:, 0:3] - pyramid[3:6]) * vector_0).sum(-1) / np.power(vector_0, 2).sum()
        betas = ((points[:, 0:3] - pyramid[3:6]) * vector_1).sum(-1) / np.power(vector_1, 2).sum()
        gammas = ((points[:, 0:3] - surface_center) * vector_2).sum(-1) / np.power(vector_2, 2).sum()
        return [alphas, betas, gammas]

    # 根据比例在另外一个金字塔结构中还原位置
    def recover_points_by_ratio(points_ratio, pyramid):
        alphas, betas, gammas = points_ratio
        surface_center = (pyramid[3:6] + pyramid[6:9] + pyramid[9:12] + pyramid[12:]) / 4.0
        vector_0, vector_1, vector_2 = pyramid[6:9] - pyramid[3:6], pyramid[12:] - pyramid[3:6], pyramid[0:3] - surface_center
        points = (alphas[:, None] * vector_0 + betas[:, None] * vector_1) + pyramid[3:6] + gammas[:, None] * vector_2
        return points
    
    def recover_points_intensity_by_ratio(points_intensity_ratio, max_intensity, min_intensity):
        return points_intensity_ratio * (max_intensity - min_intensity) + min_intensity
    
    # swap partition 1. 随机旋转需要满足条件后的金字塔区域
    if pyramids is None:
        pyramids = get_pyramids(gt_boxes).reshape([-1, 6, 5, 3])  # each six surface of boxes: [num_boxes, 6, 15=3*5]
    swap_prob, num_thres = prob, max_num_pts
    swap_pyramid_mask = np.random.uniform(0, 1, (pyramids.shape[0])) <= swap_prob   # 随机筛选出待处理的金字塔
    
    if swap_pyramid_mask.sum() > 0:
        point_masks = points_in_pyramids_mask(points, pyramids)     # 判断每个点在每个gt的6个金字塔区域
        point_nums = point_masks.sum(0).reshape(pyramids.shape[0], -1)  # [N, 6]  每个金字塔中的点数量
        non_zero_pyramids_mask = point_nums > num_thres  # gt各金字塔大于50个点的掩码表
        selected_pyramids = non_zero_pyramids_mask * swap_pyramid_mask[:, None]  # 选择满足点数量以及被随机筛选出的金字塔
        # print(selected_pyramids)
        if selected_pyramids.sum() > 0:   # 3
            # get to_swap pyramids
            index_i, index_j = np.nonzero(selected_pyramids)     # 获得非零的xy索引 index_i: [0 1 11 14]  index_j: [1 0 1 5]
            # 对满足条件的gt中随机选取一个面进行交换，获取交换面的索引
            selected_pyramid_indices = [np.random.choice(index_j[index_i == i]) \
                                            if e and (index_i == i).any() else 0 for i, e in enumerate(swap_pyramid_mask)]
            selected_pyramids_mask = selected_pyramids * one_hot(selected_pyramid_indices, num_class=6) == 1
            to_swap_pyramids = pyramids[selected_pyramids_mask]     # 筛选后的待交换的金字塔 (n, 5, 3)
            
            # get swapped pyramids 2. 随机选择符合条件的同一类型的金字塔的其他区域
            index_i, index_j = np.nonzero(selected_pyramids_mask)
            non_zero_pyramids_mask[selected_pyramids_mask] = False    # 已筛选的金字塔避免再次挑选
            # 挑选同一位置的其他金字塔结构 [1 0 1 9]
            swapped_index_i = np.array([np.random.choice(np.where(non_zero_pyramids_mask[:, j])[0]) if \
                                            np.where(non_zero_pyramids_mask[:, j])[0].shape[0] > 0 else
                                        index_i[i] for i, j in enumerate(index_j.tolist())])    # 如果当前位置没有其他金字塔结构则用原来的
            swapped_indicies = np.concatenate([swapped_index_i[:, None], index_j[:, None]], axis=1)
            # 挑选出同一列但不同行的其他金字塔区域，根据索引在原tensor中获取
            swapped_pyramids = pyramids[
                swapped_indicies[:, 0].astype(np.int32), swapped_indicies[:, 1].astype(np.int32)]   # (n, 5, 3)
            
            # concat to_swap&swapped pyramids 3. 筛选交换和被交换金字塔区域外的所有点
            swap_pyramids = np.concatenate([to_swap_pyramids, swapped_pyramids], axis=0)    # (2*n, 5, 3)
            swap_point_masks = points_in_pyramids_mask(points, swap_pyramids)   # (N, 2*n)
            remain_points = points[np.logical_not(swap_point_masks.any(-1))]    # 筛选出不在交换金字塔中的其他点
            
            # swap pyramids
            points_res = []
            num_swapped_pyramids = swapped_pyramids.shape[0]    # n
            # 4. 对于同一类型的两两金字塔区域，确认其点在金字塔中的比例，然后根据比例映射在待交换的金字塔区域中，反射强度也需要更新
            for i in range(num_swapped_pyramids):
                to_swap_pyramid = to_swap_pyramids[i]
                swapped_pyramid = swapped_pyramids[i]
                # 4.1 获得两个金字塔的点
                to_swap_points = points[swap_point_masks[:, i]]     # (57, 4)
                swapped_points = points[swap_point_masks[:, i + num_swapped_pyramids]]     # (423, 4)

                # for intensity transform  4.2 对反射强度进行归一化操作
                to_swap_points_intensity_ratio = (to_swap_points[:, -1:] - to_swap_points[:, -1:].min()) / \
                                                 np.clip(
                                                     (to_swap_points[:, -1:].max() - to_swap_points[:, -1:].min()),
                                                     1e-6, 1)
                swapped_points_intensity_ratio = (swapped_points[:, -1:] - swapped_points[:, -1:].min()) / \
                                                 np.clip(
                                                     (swapped_points[:, -1:].max() - swapped_points[:, -1:].min()),
                                                     1e-6, 1)

                # 4.3 交换金字塔的同时特考虑到了反射率的迁移
                to_swap_points_ratio = get_points_ratio(to_swap_points, to_swap_pyramid.reshape(15))
                swapped_points_ratio = get_points_ratio(swapped_points, swapped_pyramid.reshape(15))

                # 4.4 根据比例恢复点在另外一个金字塔中的具体位置
                new_to_swap_points = recover_points_by_ratio(swapped_points_ratio, to_swap_pyramid.reshape(15))
                new_swapped_points = recover_points_by_ratio(to_swap_points_ratio, swapped_pyramid.reshape(15))

                # for intensity transform 4.5 反归一化进行复原
                new_to_swap_points_intensity = recover_points_intensity_by_ratio(
                    swapped_points_intensity_ratio, to_swap_points[:, -1:].max(),
                    to_swap_points[:, -1:].min())
                new_swapped_points_intensity = recover_points_intensity_by_ratio(
                    to_swap_points_intensity_ratio, swapped_points[:, -1:].max(),
                    swapped_points[:, -1:].min())
                
                # new_to_swap_points = np.concatenate([new_to_swap_points, swapped_points[:, -1:]], axis=1)
                # new_swapped_points = np.concatenate([new_swapped_points, to_swap_points[:, -1:]], axis=1)
                # 将交换后的点与反射强度进行拼接
                new_to_swap_points = np.concatenate([new_to_swap_points, new_to_swap_points_intensity], axis=1)     # (423, 4)
                new_swapped_points = np.concatenate([new_swapped_points, new_swapped_points_intensity], axis=1)     # (57, 4)
                
                points_res.append(new_to_swap_points)
                points_res.append(new_swapped_points)
            
            points_res = np.concatenate(points_res, axis=0)     # 交换处理后的金字塔区域点
            points = np.concatenate([remain_points, points_res], axis=0)    # 点拼接在一起重新构造出点云场景
    return gt_boxes, points
