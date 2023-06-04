import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')  # 进程数
    parser.add_argument('--extra_tag', type=str, default='default',
                        help='extra tag for this experiment, default is kitti_models/pointpillar/default')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='use fixed seed')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=1, help='max number of saved checkpoint')  # 不可以设置为0
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    # 这里的参数设置 nargs = argparse.REMAINDER，所有剩余的参数，均转化为一个列表赋值给此项
    # 使用样例：--set MODEL.VFE.FEATURE_SELECT.USE_DISTANCE_INFO True
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False,
                        help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')

    args = parser.parse_args()

    # 从yaml文件中加载自cfg
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem      # from cfg/models/xxx.yaml get xxx
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()  # 相关参数设置
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'  # gpu要被batchsize整除平分
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)     # 保证相同输入的输出是一致的

    # 输出结果的各种保存路径
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)   # parents=True递归创建目录文件，exist_ok=True如果存在不报错
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)    # 日志处理

    # log to file: 记录信息
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():        # 打印args参数设置信息
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)     # 打印cfg文件信息
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))    # 代码行执行linux的复制命令

    # 利用tensorboard进行训练过程可视化(更推荐使用wandb)
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer & scheduler---------------------------
    # 1)构建dataset和dataloader
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,    # 数据集配置部分
        class_names=cfg.CLASS_NAMES,    # 数据集类别列表
        batch_size=args.batch_size,     # 分配到每块gpu上的batch_size, 非总的batch_size
        dist=dist_train,                # dist分布式训练标志(bool变量)
        workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None     # 默认不设定随机种子
    )

    # 2)构建网络结构
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)    # BN同步
    model.cuda()

    # 3)构建优化器(后续可以从断点文件或者预训练文件中加载优化器参数)
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # 有设置预训练模型优先加载预训练模型，没有则加载最新的断点权重，两者均没有则重头开始训练
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:   # 加载预训练模型
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:       # 如果不设定加载断点模型，默认按照最后修改时间权重来加载进行继续训练
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))  # 罗列所有.pth文件
              
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)    # 按照文件最近修改时间来进行排序
            while len(ckpt_list) > 0:
                try:    # 尝试加载最近修改时间权重文件，如果无法加载则替补下一个最近权重文件加载
                    it, start_epoch = model.load_params_with_optimizer(
                        ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
                    )
                    last_epoch = start_epoch + 1
                    break
                except: # 最后的权重文件无法加载则将其排除
                    ckpt_list = ckpt_list[:-1]

    # 设置分布式训练并打印模型结构
    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:      # 多卡分布式训练
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)  # 打印模型参数

    # 4)构建学习率策略
    lr_scheduler, lr_warmup_scheduler = build_scheduler(    # OneCycle, None
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # 训练代码
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),    # 模型前向传播入口函数
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch, 
        logger=logger, 
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record, 
        show_gpu_stat=not args.wo_gpu_stat
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # 构建测试集
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    # 测试文件路径
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)  # Only evaluate the last args.num_epochs_to_eval epochs

    # 调用test中的repeat_eval_ckpt进行模型评估（核心评估代码）
    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
