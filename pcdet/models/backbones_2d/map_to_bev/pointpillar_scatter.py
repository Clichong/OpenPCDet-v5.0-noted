import torch
import torch.nn as nn

# 功能：PointPillar的MAP_TO_BEV部分具体网络实现
class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        """
        Args:
            model_cfg:  MAP_TO_BEV的配置部分
                NAME: PointPillarScatter
                NUM_BEV_FEATURES: 64
            grid_size:  网格大小
        """
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size   # PointPillar将整个点云场景切分为平面网格，所以这里的z维度一定是1
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            pillar_features:(31530,64)
            voxels:(31530,32,4) --> (x,y,z,intensity)
            voxel_coords:(31530,4) --> (batch_index,z,y,x) 在dataset.collate_batch中增加了batch索引
            voxel_num_points:(31530,)
        Returns:
            batch_spatial_features:(4, 64, 496, 432)
        """
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']     # (102483, 64) / (102483, 4)
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1    # 16

        # 依次对每个点云帧场景进行处理
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(      # 构建[64, 1x432x496]的0矩阵
                self.num_bev_features,   # 64
                self.nz * self.nx * self.ny,    # 1x432x496
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx  # 构建batch掩码
            this_coords = coords[batch_mask, :]     # 用来挑选出当前真个batch数据中第batch_idx的点云帧场景
            # this_coords: [7857, 4]  4个维度的含义分别为：(batch_index,z,y,x) 由于pointpillars只有一层，所以计算索引的方法如下所示
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]   # 网格的一维展开索引
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]    # 根据mask提取pillar_features [7857, 64]
            pillars = pillars.t()   # 矩阵转置 [64, 7857]
            spatial_feature[:, indices] = pillars       # 在索引位置填充pillars
            batch_spatial_features.append(spatial_feature)      # 将空间特征加入list,每个元素为(64,214272)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)     # 堆叠拼接 (16, 64, 214272)
        # reshape回原空间(伪图像)--> (16, 64, 496, 432)， 再保存结果
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx) # (16, 64, 496, 432)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
