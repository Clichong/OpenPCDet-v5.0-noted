import glob
import os

import torch
import tqdm
import time
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils

# 单个epoch训练流程
def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, 
                    use_logger_to_record=False, logger=None, logger_iter_interval=50, cur_epoch=None, 
                    total_epochs=None, ckpt_save_dir=None, ckpt_save_time_interval=300, show_gpu_stat=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)    # 转化为迭代器,可以通过next函数来从迭代器中获取一个批次的数据

    ckpt_save_cnt = 1   # 统计模型保留次数
    start_it = accumulated_iter % total_it_each_epoch
    # 表示进程序号，用于进程间通讯，表征进程优先级，rank = 0的主机为master节点
    if rank == 0:   # 在主进程设置训练进度条,其他进程不用设置
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()     # 各类时间的统计
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()

    end = time.time()
    # 开始迭代每批次数据集数据
    for cur_it in range(start_it, total_it_each_epoch):
        try:
            batch = next(dataloader_iter)   # 通过next函数获取迭代器一个批次的数据
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')
        
        data_timer = time.time()
        cur_data_time = data_timer - end    # 获取一个batch数据所耗费的时间

        lr_scheduler.step(accumulated_iter)     # 学习率更新

        # 获取当前学习率
        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        # 在tensorboard中添加学习率和当前迭代次数
        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()           # 设置为训练模式
        optimizer.zero_grad()   # 梯度清零

        loss, tb_dict, disp_dict = model_func(model, batch)     # 模型前向传播,返回各类损失计算结果

        loss.backward()     # 损失反向传播
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)   # 裁剪参数迭代的梯度范数
        optimizer.step()    # 梯度更新

        accumulated_iter += 1   # 累积处理批次数据+1
 
        cur_forward_time = time.time() - data_timer     # 完成一个批次前向传播的时间
        cur_batch_time = time.time() - end              # 完成一个批次处理的全部时间
        end = time.time()       # 批次处理时间的更新

        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            data_time.update(avg_data_time)          # 记录加载batch数据的时间
            forward_time.update(avg_forward_time)    # 记录前向传播batch数据的时间
            batch_time.update(avg_batch_time)        # 记录一个batch数据处理流程的时间
            
            disp_dict.update({      # 更新字典的key和values (时间记录了当前值和平均值)
                'loss': loss.item(), 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })
            
            if use_logger_to_record:    # 是否记录在磁盘与控制台中
                # 每迭代logger_iter_interval次打印一次信息 + 首尾两次打印信息
                if accumulated_iter % logger_iter_interval == 0 or cur_it == start_it or cur_it + 1 == total_it_each_epoch:
                    trained_time_past_all = tbar.format_dict['elapsed']
                    second_each_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1.0)

                    trained_time_each_epoch = pbar.format_dict['elapsed']
                    remaining_second_each_epoch = second_each_iter * (total_it_each_epoch - cur_it)
                    remaining_second_all = second_each_iter * ((total_epochs - cur_epoch) * total_it_each_epoch - cur_it)
                    # 统计各类信息
                    disp_str = ', '.join([f'{key}={val}' for key, val in disp_dict.items() if key != 'lr'])
                    disp_str += f', lr={disp_dict["lr"]}'
                    batch_size = batch.get('batch_size', None)  # 获取batch size

                    # 在train精度条中显示的内容
                    logger.info(f'epoch: {cur_epoch}/{total_epochs}, acc_iter={accumulated_iter}, cur_iter={cur_it}/{total_it_each_epoch}, batch_size={batch_size}, '
                                f'time_cost(epoch): {tbar.format_interval(trained_time_each_epoch)}/{tbar.format_interval(remaining_second_each_epoch)}, '
                                f'time_cost(all): {tbar.format_interval(trained_time_past_all)}/{tbar.format_interval(remaining_second_all)}, '
                                f'{disp_str}')

                    # 每150次迭代再打印一次gpustat洗洗
                    if show_gpu_stat and accumulated_iter % (3 * logger_iter_interval) == 0:
                        # To show the GPU utilization, please install gpustat through "pip install gpustat"
                        gpu_info = os.popen('gpustat').read()
                        logger.info(gpu_info)
            else:                
                pbar.update()   # 更新进度条
                pbar.set_postfix(dict(total_it=accumulated_iter))
                tbar.set_postfix(disp_dict)
                # tbar.refresh()

            if tb_log is not None:  # tensorboard记录各种loss和学习率lr的变化
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
            
            # save intermediate ckpt every {ckpt_save_time_interval} seconds
            # 保留的是最新模型，命名为'latest_model'
            time_past_this_epoch = pbar.format_dict['elapsed']
            if time_past_this_epoch // ckpt_save_time_interval >= ckpt_save_cnt:
                ckpt_name = ckpt_save_dir / 'latest_model'
                save_checkpoint(
                    checkpoint_state(model, optimizer, cur_epoch, accumulated_iter), filename=ckpt_name,
                )
                logger.info(f'Save latest model to {ckpt_name}')
                ckpt_save_cnt += 1  # 保留次数+1
                
    if rank == 0:
        pbar.close()    # 关闭本轮epoch进度条
    return accumulated_iter     # 返回累积的迭代次数


# 全epoch训练流程
def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, 
                use_logger_to_record=False, logger=None, logger_iter_interval=None, ckpt_save_time_interval=None, show_gpu_stat=False):
    """
    Args:
        model:模型
        optimizer: 优化器
        train_loader: Dataloader
        model_func: 模型函数装饰器，其在model的__init__.py中:主要是将数据放到模型上在返回loss
        lr_scheduler: 学习率调度器
        optim_cfg: 优化器配置
        start_epoch: 起始epoch
        total_epochs: 总共epoch数量
        start_iter: 起始迭代数
        rank:进程号
        tb_log: tensorboad的log
        ckpt_save_dir: checkpoint存储文件夹路径
        train_sampler: 训练数据采样器 DistributedSampler
        lr_warmup_scheduler: 学习率热身调度器
        ckpt_save_interval: checkpoint存储间隔，默认为1
        max_ckpt_save_num: 最大的checkpoint存储数量，默认为50
        merge_all_iters_to_one_epoch: 是否将所有iter合并为一个epoch
    """
    accumulated_iter = start_iter   # 累计已迭代次数 开始的时刻为0
    # 进度条设置 start_epoch -> total_epochs
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)     # kitti:232
        if merge_all_iters_to_one_epoch:    # False
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)    # 将Dataloader转换为迭代格式
        for cur_epoch in tbar:      # 获取当前的训练epoch数
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # 1.选择是否进行warm up学习率预热
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler    # OneCycle

            # 2.训练一个epoch，返回的是累积的处理batch数量
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,         # 学习率调度器
                accumulated_iter=accumulated_iter,  # 传入当前处理的batch数量
                optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                cur_epoch=cur_epoch, total_epochs=total_epochs,
                use_logger_to_record=use_logger_to_record, 
                logger=logger, logger_iter_interval=logger_iter_interval,
                ckpt_save_dir=ckpt_save_dir, ckpt_save_time_interval=ckpt_save_time_interval, 
                show_gpu_stat=show_gpu_stat
            )

            # 3.保存训练模型(每ckpt_save_interval次保存一次模型)
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                # 对当前保留的全部模型按时间排序，glob会包含全部路径
                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                # 只保留最新的前k个模型, 删除最前面的文件
                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                # 保留当前模型
                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    # 获取优化器的state_dict
    optim_state = optimizer.state_dict() if optimizer is not None else None

    # 获取模型的state_dict
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        if torch.__version__ >= '1.4':
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename, _use_new_zipfile_serialization=False)
        else:
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    # 真正保留模型的操作
    filename = '{}.pth'.format(filename)
    if torch.__version__ >= '1.4':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)
