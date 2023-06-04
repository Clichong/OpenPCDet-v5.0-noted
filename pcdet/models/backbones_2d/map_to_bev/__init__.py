from .height_compression import HeightCompression
from .pointpillar_scatter import PointPillarScatter
from .conv2d_collapse import Conv2DCollapse

# 根据MODEL中的MAP_TO_BEV确定选择的模块
__all__ = {
    'HeightCompression': HeightCompression,
    'PointPillarScatter': PointPillarScatter,
    'Conv2DCollapse': Conv2DCollapse
}
