import torch.nn as nn

# 功能：作为VFE模块继承的基类
class VFETemplate(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        """
        Args:
            model_cfg:
                NAME: PillarVFE         # 选择VFE模块
                WITH_DISTANCE: False
                USE_ABSLOTE_XYZ: True
                USE_NORM: True
                NUM_FILTERS: [64]
        """
        super().__init__()
        self.model_cfg = model_cfg     # 保存VFE部分的配置参数

    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        """
        Args:
            **kwargs:

        Returns:
            batch_dict:
                ...
                vfe_features: (num_voxels, C)
        """
        raise NotImplementedError
