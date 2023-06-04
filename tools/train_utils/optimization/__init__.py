from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle


# 功能：构造优化器
def build_optimizer(model, optim_cfg):
    # 采用optim的Adam初始化
    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)

    # 采用optim的sgd初始化
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )

    # 采用adam_onecycle优化器
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':
        # 功能: 取出model的每个子模块
        def children(m: nn.Module):
            return list(m.children())   # PointPillars: PillarVFE/PointPillarScatter/BaseBEVBackbone/AnchorHeadSingle

        # 功能: 统计model有多少个子模块
        def num_children(m: nn.Module) -> int:
            return len(children(m))     # PointPillars: 4个子模块

        # 将所有的网络层堆叠成一个列表
        # a = [[1], [2], [3], [4], [5]]
        # sum(a, []) # [1, 2, 3, 4, 5]
        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]

        # 将列表中的个网络层构建成一个Sequential序列
        # [Sequential(
        #   (0): Linear(in_features=10, out_features=64, bias=False)
        #   (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        #   ...)]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        # partial接收函数optim.Adam作为参数，固定optim.Adam的参数betas=(0.9, 0.99), 其实相当于是固定参数的Adam优化器
        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))     # 初始化adam优化器

        # 将参数划分为两组: bn层与其他非bn层
        optimizer = OptimWrapper.create(
            optimizer_func,             # Adam
            3e-3,                       # lr
            get_layer_groups(model),    # model网络层的序列
            wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError

    return optimizer


# 功能：构建学习率训练策略
def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    """
    Func:
        构建学习率调度器：三种方式adam_onecycle、LambdaLR、CosineWarmupLR
    Args:
        optimizer:      优化器
        total_iters_each_epoch:  一个epoch的迭代数 len(train_loader)，kitti数据集这里为232
        total_epochs:   总的epoch训练次数
        last_epoch:     上一次epoch数(-1)
        optim_cfg:      优化器配置
    """
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]   # 232 * [35, 45] -> [8120, 10440]

    # 自定义学习率调度函数
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            # 如果当前epoch数大于节点值，则更新学习率
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY  # LR_DECAY: 0.1
        # 防止学习率过小
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR) # LR_CLIP: 0.0000001

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs # 232 * 20 = 4640

    # 构建adam_onecycle学习率调度器
    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)   # 在特定的epoch进行学习衰减

        # warm up：在刚刚开始训练时以很小的学习率进行训练，使得网络熟悉数据，随着训练的进行学习率慢慢变大，
        # 到了一定程度，以设置的初始学习率进行训练，接着过了一些inter后，学习率再慢慢变小；学习率变化：上升——平稳——下降；
        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler
