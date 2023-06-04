import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            # out_channels = out_channels     # 这里直接将NUM_FILTERS作为每层的linear神经元数
            out_channels = out_channels // 2   # origin

        # 如果使用norm，就是简单的linear+bn； 否则只有linear
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)   # 默认参数：eps=1e-5, momentum=0.1
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000   # 如果当前的batch点数量太大，需要进行特殊处理

    def forward(self, inputs):
        """
        如果当前batch的点数量太大，需要进行分批进行linear处理，否则效果随机性较大，这里一个小批次的点数量设置为50000
        """
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])   # 分part来进行模型的升维处理
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)   # [(50000, 32, 64), (50000, 32, 64), (2716, 32, 64)]
        else:
            x = self.linear(inputs)     # 如果点数量少于5w可以直接linear处理
        torch.backends.cudnn.enabled = False    # 取消配置最高效率算法
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x   # 对channels=64的维度进行norm
        torch.backends.cudnn.enabled = True     # 配置最高效率算法
        x = F.relu(x)   # 激活函数
        x_max = torch.max(x, dim=1, keepdim=True)[0]    # 相当于是maxpool操作（pointnet中提出）

        if self.last_vfe:
            return x_max    # 若是单层那么就不需要(102483, 1, 64)
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)      # (102483, 1, 64) -> (102483, 32, 64)
            x_concatenated = torch.cat([x, x_repeat], dim=2)    # 重复数据和初始数据进行拼接操作 (102483, 32, 64) -> (102483, 32, 128)
            return x_concatenated

# 功能：PointPillar的VFE部分具体网络实现，集成基类的VFE模板
class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        """
        Args:
            model_cfg:                  MODEL_VFE配置部分
                NAME: PillarVFE
                WITH_DISTANCE: False
                USE_ABSLOTE_XYZ: True
                USE_NORM: True
                NUM_FILTERS: [64]
            num_point_features:         原始特征维度
            voxel_size:                 体素大小
            point_cloud_range:          点云范围
            grid_size:                  网格大小
            depth_downsample_factor:    None
        """
        super().__init__(model_cfg=model_cfg)   # 基类保存了配置文件，两个子函数会重新复写

        self.use_norm = self.model_cfg.USE_NORM     # True
        self.with_distance = self.model_cfg.WITH_DISTANCE   # False
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ  # True

        # 如果use_absolute_xyz==True，则num_point_features=4+6，否则为3
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:    # 如果使用距离特征，即使用sqrt(x^2+y^2+z^2)，则使用特征加1
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS   # [64]
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)    # [10, 64]

        # 构造线性层：10 -> 64
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)    # 加入网络结构

        self.voxel_x = voxel_size[0]    # 0.16
        self.voxel_y = voxel_size[1]    # 0.16
        self.voxel_z = voxel_size[2]    # 4
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]     # 0.16/2 + 0 = 0.08
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]     # 0.16/2 + (-39.68) = -39.6
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]     # 4/2 + (-3) = -1

    # 功能：返回输出特征维度大小
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """
        功能：构造voxel每个点的一个掩码矩阵，voxel数量为N，每个voxel最大点数量为32，那么返回的就是(N, 32)的布尔变量矩阵
        """
        actual_num = torch.unsqueeze(actual_num, axis + 1)  # (102483, 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1    # [1, -1]
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num     # (102483, 32)
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        """
        batch_dict:
            points:(97687,5)
            frame_id:(4,) --> (2238,2148,673,593)
            gt_boxes:(4,40,8)--> (x,y,z,dx,dy,dz,ry,class)
            use_lead_xyz:(4,) --> (1,1,1,1)
            voxels:(31530,32,4) --> (x,y,z,intensity)
            voxel_coords:(31530,4) --> (batch_index,z,y,x) 在dataset.collate_batch中增加了batch索引
            voxel_num_points:(31530,)
            image_shape:(4,2) [[375 1242],[374 1238],[375 1242],[375 1242]]
            batch_size:4
        """
        # shape: (102483, 32, 4) / (102483,) / (102483, 4)
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # 1)计算voxel内每个点离voxel中心点的相对距离
        # 求每个voxel的平均值(102483, 1, 3) / (102483, 1, 1) = (102483, 1, 3)
        # 被求和的维度，在求和后会变为1，如果没有keepdim=True的设置，python会默认压缩该维度
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)   # 计算每个voxel内有效点的中心坐标(无效点部分用已0表示)
        f_cluster = voxel_features[:, :, :3] - points_mean  # voxel内的每个点与中心坐标的偏移量   (102483, 32, 3)

        # 2) 计算voxel内每个点离网格的相对距离
        # coords是网格点坐标，不是实际坐标，乘以voxel大小再加上偏移量是恢复网格中心点实际坐标
        f_center = torch.zeros_like(voxel_features[:, :, :3])   # voxel内每个点与当前点网格的相对距离 (102483, 32, 3)
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        # 3) 构建额外特征进行编码操作

        if self.use_absolute_xyz:   # 是否是有xyz绝对位置信息
            features = [voxel_features, f_cluster, f_center]    # dim: 4 + 3 + 3
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:   # 是否使用离原点的距离信息
            # torch.norm的第一个2指的是求2范数，第二个2是在第三维度求范数，其实求解二范数就相当于是求解离原点的距离
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)  # (102483, 32, 10)

        # 3) 每个voxel点的掩码构造
        voxel_count = features.shape[1]     # 32 每个voxel内最多有32个有效点
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)   # (102483, 32)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)     # (102483, 32, 1)
        features *= mask

        # 4) 特征编码
        # (102483, 32, 10) -> (102483, 32, 64) -> (102483, 1, 64) -> (102483, 64)
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()   # 降维
        batch_dict['pillar_features'] = features    # 保留特征处理的结果
        return batch_dict
