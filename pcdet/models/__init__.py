from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector

try:
    import kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')


# 功能：搭建网络模型，真正搭建网络的入口是detectors目录下的build_detector
def build_network(model_cfg, num_class, dataset):
    """
    Args:
        model_cfg:   yaml配置文件的MODEL部分
        num_class:   类别数目（kitti数据集一般用3个类别：'Car', 'Pedestrian', 'Cyclist'）
        dataset:     训练数据集
    """
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model

# 功能: 将数据转移到gpu上加速
def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


# 功能: 具体模型前向传播过程，进行损失计算
def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)    # 前向传播入口

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()  # 批次处理+1
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
