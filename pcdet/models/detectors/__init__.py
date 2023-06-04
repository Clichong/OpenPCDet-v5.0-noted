from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .centerpoint import CenterPoint
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .mppnet import MPPNet
from .mppnet_e2e import MPPNetE2E
from .pillarnet import PillarNet

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,
    'PillarNet': PillarNet,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'MPPNet': MPPNet,
    'MPPNetE2E': MPPNetE2E,
    # 'PillarNet': PillarNet
}

# 功能：搭建整个网络模型的入口
def build_detector(model_cfg, num_class, dataset):
    # 根据yaml文件中MODLE中的NAME来确定构造哪个模型
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
