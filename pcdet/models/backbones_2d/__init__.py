from .base_bev_backbone import BaseBEVBackbone, BaseBEVBackboneV1
from .dynamic_bev_backbone import DynamicBEVBackbone

# 根据MODEL中的BACKBONE_2D确定选择的模块
__all__ = {
    'BaseBEVBackbone': BaseBEVBackbone,
    'BaseBEVBackboneV1': BaseBEVBackboneV1,
    'DynamicBEVBackbone': DynamicBEVBackbone    # 提取的动态多尺度加权特征提取模块
}
