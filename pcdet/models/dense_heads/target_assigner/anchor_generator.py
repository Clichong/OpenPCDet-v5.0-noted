import torch

# 功能: AnchorHeadTemplate中的函数generate_anchors调用
class AnchorGenerator(object):
    def __init__(self, anchor_range, anchor_generator_config):
        """
        Args:
            anchor_range:   点云范围
            anchor_generator_config:  类别的anchor配置
        """
        super().__init__()
        self.anchor_generator_cfg = anchor_generator_config
        self.anchor_range = anchor_range
        self.anchor_sizes = [config['anchor_sizes'] for config in anchor_generator_config]              # list:3 --> [[[3.9, 1.6, 1.56]],[[0.8, 0.6, 1.73]],[[1.76, 0.6, 1.73]]]
        self.anchor_rotations = [config['anchor_rotations'] for config in anchor_generator_config]      # list:3 --> [[0, 1.57],[0, 1.57],[0, 1.57]]
        self.anchor_heights = [config['anchor_bottom_heights'] for config in anchor_generator_config]   # list:3 --> [[-1.78],[-0.6],[-0.6]]
        self.align_center = [config.get('align_center', False) for config in anchor_generator_config]   # list:3 --> [False, False, False]

        assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(self.anchor_heights)   # 要与类别个数一致
        self.num_of_anchor_sets = len(self.anchor_sizes)    # 3种anchor设置

    # 功能：对每个类别具体在每个位置生成anchor的函数
    def generate_anchors(self, grid_sizes):
        assert len(grid_sizes) == self.num_of_anchor_sets
        all_anchors = []
        num_anchors_per_location = []
        # 对每个类别都生成anchor矩阵
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
                grid_sizes, self.anchor_sizes, self.anchor_rotations, self.anchor_heights, self.align_center):

            # 统计每个位置anchor数量以及xy方向的步长(z方向不作划分)
            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))   # 2
            if align_center:    # 中心对其
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / grid_size[0]
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (grid_size[0] - 1)   # x方向步长
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (grid_size[1] - 1)   # y方向步长
                x_offset, y_offset = 0, 0

            # 根据步长构建xy方向的间隔点
            x_shifts = torch.arange(    # (69.12 - 0) / (216 - 1) = 0.321488  间隔点有216个，所以步长为0.321488
                self.anchor_range[0] + x_offset, self.anchor_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            ).cuda()
            y_shifts = torch.arange(    # (39.68 - (-39.68)) / (248 - 1) = 0.321295 间隔点有248个，所以步长为0.321295
                self.anchor_range[1] + y_offset, self.anchor_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            ).cuda()
            z_shifts = x_shifts.new_tensor(anchor_height)   # [-1.78] PointPillar不对z轴进行区间划分

            # 获取当前类别的anchor大小信息以及角度信息(位置信息已获取)
            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__() # 1,2
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)  # [0, 1.57]
            anchor_size = x_shifts.new_tensor(anchor_size)          # [[3.9, 1.6, 1.56]]

            # 根据xyz步长构建三维网格坐标 [x_grid, y_grid, z_grid] --> [(216,248,1), (216,248,1),(216,248,1)]
            x_shifts, y_shifts, z_shifts = torch.meshgrid([
                x_shifts, y_shifts, z_shifts
            ])  # [x_grid, y_grid, z_grid]
            # meshgrid可以理解为在原来的维度上进行扩展, (np.meshgrid 和 torch.meshgrid 是返回结果不一样的)
            # 例如:
            # x原来为（216，）-->（216，1, 1）--> (216,248,1)
            # y原来为（248，）--> (1，248，1）--> (216,248,1)
            # z原来为 (1,) --> (1,1,1) --> (216,248,1)

            # xyz位置信息堆叠，完成anchor位置信息的构建: (216,248,1,3)
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)  # [x,y,z,3]-->[216,248,1,3]

            # 将anchor的位置信息与尺寸大小进行组合: (216,248,1,1,6)
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)        # (216,248,1,3) -> (216,248,1,1,3)
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])  # (1,1,1,1,3) -> (216,248,1,1,3)
            anchors = torch.cat((anchors, anchor_size), dim=-1)     # anchors的位置+大小 --> (216,248,1,1,6)

            # 将anchor的位置信息、尺寸大小、旋转角信息进行组合: (216,248,1,1,2,7)
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)   # (216,248,1,1,1,6) -> (216,248,1,1,2,6)
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat([*anchors.shape[0:3], num_anchor_size, 1, 1])  # (1,1,1,1,2,1) -> (216,248,1,1,2,1)
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)  # anchors的位置+大小+旋转方向 --> (216,248,1,1,2,7)

            # 最后调整anchor的维度: (1,248,216,1,2,7)
            # 最后一维的7表示的特征信息为: [x, y, z, dx, dy, dz, rot], [位置信息xyz, 尺寸信息, 旋转角度]
            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()    # (216,248,1,1,2,7) ->  (1,248,216,1,2,7)
            #anchors = anchors.view(-1, anchors.shape[-1])

            anchors[..., 2] += anchors[..., 5] / 2  # z轴的位置信息设置为anchor高度一半
            all_anchors.append(anchors)

        return all_anchors, num_anchors_per_location


if __name__ == '__main__':
    from easydict import EasyDict
    config = [
        EasyDict({
            'anchor_sizes': [[2.1, 4.7, 1.7], [0.86, 0.91, 1.73], [0.84, 1.78, 1.78]],
            'anchor_rotations': [0, 1.57],
            'anchor_heights': [0, 0.5]
        })
    ]

    A = AnchorGenerator(
        anchor_range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
        anchor_generator_config=config
    )
    import pdb
    pdb.set_trace()
    A.generate_anchors([[188, 188]])
