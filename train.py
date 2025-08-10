# YOLOv5  by Ultralytics, GPL-3.0 license
"""
在自定义数据集上训练 YOLOv5 模型。
模型权重与数据集可按需自动下载（源自官方最新版本）。

单卡训练示例：
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # 以预训练模型为起点（推荐）
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # 从零开始训练

多卡 DDP 训练示例：
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py \
        --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

相关链接：
Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
"""

import argparse
import math
import os
import random
import sys
import time
import cv2
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 添加根目录到 PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对路径

import val as validate  # 用于每轮训练结束后的 mAP 验证
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
# from utils.dataloaders import create_dataloader, create_dataloader_val
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, check_amp, check_dataset, check_file, check_git_status, check_img_size,
                           check_requirements, check_suffix, check_yaml, colorstr, get_latest_run, increment_path,
                           init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, print_args, print_mutation, strip_optimizer, yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
os.environ['MASTER_PORT'] = '8888'


def train(hyp, opt, device, callbacks):  # hyp 可以是 hyp.yaml 的路径，也可以是超参数字典
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    callbacks.run('on_pretrain_routine_start')

    # 目录与权重保存路径
    w = save_dir / 'weights'  # 权重目录
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # 创建目录
    last, best = w / 'last.pt', w / 'best.pt'

    # 超参数处理
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # 加载超参数字典
    LOGGER.info(colorstr('超参数: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # 用于保存超参数到检查点

    # 保存运行设置
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # 日志记录器
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # 日志记录器实例

        # 注册回调函数
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # 处理自定义数据集工件链接
        data_dict = loggers.remote_dataset
        if resume:  # 如果从远程工件恢复运行
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # 配置
    plots = not evolve and not opt.noplots  # 创建图表
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # 检查是否为 None
    train_path, train_path2, val_path, val_path2 = data_dict['train'], data_dict['train2'], data_dict['val'], data_dict['val2']
    nc = 1 if single_cls else int(data_dict['nc'])  # 类别数量
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # 类别名称
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO 数据集

    # 模型
    check_suffix(weights, '.pt')  # 检查权重文件后缀
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # 如果本地没有则下载
        ckpt = torch.load(weights, map_location='cpu')  # 加载检查点到 CPU 以避免 CUDA 内存泄露
        model = Model(cfg or ckpt['model'].yaml, ch=4, nc=nc, anchors=hyp.get('anchors')).to(device)  # 创建模型
        # model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # 创建模型
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # 排除的键
        csd = ckpt['model'].float().state_dict()  # 检查点状态字典转为 FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # 取交集
        model.load_state_dict(csd, strict=False)  # 加载权重

        LOGGER.info(f'从 {weights} 传输了 {len(csd)}/{len(model.state_dict())} 个参数')  # 报告
    else:
        model = Model(cfg, ch=4, nc=nc, anchors=hyp.get('anchors')).to(device)  # 创建模型
    amp = check_amp(model)  # 检查 AMP

    # 冻结层
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # 要冻结的层
    for k, v in model.named_parameters():
        v.requires_grad = True  # 训练所有层
        if any(x in k for x in freeze):
            LOGGER.info(f'冻结 {k}')
            v.requires_grad = False

    # 图像尺寸
    gs = max(int(model.stride.max()), 32)  # 网格尺寸（最大步长）
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # 验证图像尺寸是 gs 的倍数

    # 批次大小
    if RANK == -1 and batch_size == -1:  # 仅单GPU，估算最佳批次大小
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    # 优化器
    nbs = 64  # 标准批次大小
    accumulate = max(round(nbs / batch_size), 1)  # 优化前累积损失的次数
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # 缩放权重衰减
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # 调度器
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # 余弦 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # 线性
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA（指数移动平均）
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # 恢复训练
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # DP 模式
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('警告 ⚠ 不推荐使用 DP 模式，请使用 torch.distributed.run 进行最佳的 DDP 多GPU 训练。\n'
                       '请参考多GPU教程：https://github.com/ultralytics/yolov5/issues/475')
        model = torch.nn.DataParallel(model)

    # 同步批归一化
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('使用同步批归一化()')

    # 训练数据加载器
    train_loader, dataset = create_dataloader(train_path,
                                              train_path2,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('训练: '),
                                              shuffle=True,
                                              subset=0.1 if opt.mini else 1.0)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # 最大标签类别
    assert mlc < nc, f'标签类别 {mlc} 超过了 nc={nc} 在 {data} 中。可能的类别标签为 0-{nc - 1}'

    # 进程 0
    if RANK in {-1, 0}:
        # val_loader = create_dataloader_val(val_path,
        val_loader = create_dataloader(val_path,
                                       val_path2,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('验证: '),
                                       subset=0.1 if opt.mini else 1.0)[0]

        if not resume:
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # 运行自动锚框
            model.half().float()  # 预先降低锚框精度

        callbacks.run('on_pretrain_routine_end', labels, names)

    # DDP 模式
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # 模型属性
    nl = de_parallel(model).model[-1].nl  # 检测层数量（用于缩放超参数）
    hyp['box'] *= 3 / nl  # 缩放到层数
    hyp['cls'] *= nc / 80 * 3 / nl  # 缩放到类别和层数
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # 缩放到图像尺寸和层数
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # 将类别数量附加到模型
    model.hyp = hyp  # 将超参数附加到模型
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # 附加类别权重
    model.names = names

    # 开始训练
    t0 = time.time()
    nb = len(train_loader)  # 批次数量
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # 预热迭代次数，最多（3轮，100次迭代）
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # 限制预热时间 < 1/2 的训练时间
    last_opt_step = -1
    maps = np.zeros(nc)  # 每个类别的 mAP
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, 验证损失(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # 不要移动
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)  # 初始化损失类
    callbacks.run('on_train_start')
    LOGGER.info(f'图像尺寸 {imgsz} 训练，{imgsz} 验证\n'
                f'使用 {train_loader.num_workers * WORLD_SIZE} 个数据加载器工作进程\n'
                f"日志结果保存到 {colorstr('bold', save_dir)}\n"
                f'开始训练 {epochs} 轮...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run('on_train_epoch_start')
        model.train()

        # 更新图像权重（可选，仅单GPU）
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # 类别权重
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # 图像权重
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # 随机加权索引

        # 更新马赛克边界（可选）
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # 高度，宽度边界

        mloss = torch.zeros(3, device=device)  # 平均损失
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 7) % ('轮次', 'GPU内存', 'box损失', 'obj损失', 'cls损失', '实例数', '尺寸'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # 进度条
        optimizer.zero_grad()

        for i, (imgs, imgs2, targets, paths, paths2, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch  # 累计批次数（从训练开始）
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 转 float32, 0-255 转 0.0-1.0
            imgs2 = imgs2.to(device, non_blocking=True).float() / 255  # uint8 转 float32, 0-255 转 0.0-1.0

            # 预热
            if ni <= nw:
                xi = [0, nw]  # x 插值
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou 损失比率（obj_loss = 1.0 或 iou）
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias 学习率从 0.1 下降到 lr0，其他学习率从 0.0 上升到 lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # 多尺度
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # 尺寸
                sf = sz / max(imgs.shape[2:])  # 缩放因子
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # 新形状（拉伸到 gs 的倍数）
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
                    imgs2 = nn.functional.interpolate(imgs2, size=ns, mode='bilinear', align_corners=False)

            # 前向传播
            with torch.cuda.amp.autocast(amp):
                imgs = torch.cat((imgs, imgs2), dim=1)
                imgs = imgs[:, :4, :, :]
                pred = model(imgs)  # 前向传播
                loss, loss_items = compute_loss(pred, targets.to(device))  # 损失按批次大小缩放
                if RANK != -1:
                    loss *= WORLD_SIZE  # DDP 模式下设备间梯度平均
                if opt.quad:
                    loss *= 4.

            # 反向传播
            scaler.scale(loss).backward()

            # 优化 - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # 取消梯度缩放
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # 梯度裁剪
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # 记录
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # 更新平均损失
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', model, ni, imgs[:, :3, :, :], targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # 结束批次 ------------------------------------------------------------------------------------------------

        # 调度器
        lr = [x['lr'] for x in optimizer.param_groups]  # 用于日志记录器
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # 计算 mAP
                results, maps, _ = validate.run(data_dict,
                                                batch_size=batch_size // WORLD_SIZE * 2,
                                                imgsz=imgsz,
                                                half=amp,
                                                model=ema.ema,
                                                single_cls=single_cls,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                plots=False,
                                                callbacks=callbacks,
                                                compute_loss=compute_loss)

            # 更新最佳 mAP
            fi = fitness(np.array(results).reshape(1, -1))  # [P, R, mAP@.5, mAP@.5-.95] 的加权组合
            stop = stopper(epoch=epoch, fitness=fi)  # 早停检查
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # 保存模型
            if (not nosave) or (final_epoch and not evolve):  # 如果保存
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                    'opt': vars(opt),
                    'date': datetime.now().isoformat()}

                # 保存 last，best 并删除
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # 早停
        if RANK != -1:  # 如果 DDP 训练
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # 广播 'stop' 到所有进程
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # 必须中断所有 DDP 进程

        # 结束 epoch ----------------------------------------------------------------------------------------------------
    # 结束训练 -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} 轮训练在 {(time.time() - t0) / 3600:.3f} 小时内完成。')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # 剥离优化器
                if f is best:
                    LOGGER.info(f'\n验证 {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # 最佳 pycocotools 在 iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss)  # 用图表验证最佳模型
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, epoch, results)

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='初始权重路径')
    parser.add_argument('--cfg', type=str, default='models/v5s_ai_game.yaml', help='模型配置文件路径')
    parser.add_argument('--data', type=str, default=ROOT / 'train_file/train_file.yaml')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='超参数文件路径')
    parser.add_argument('--epochs', type=int, default=300, help='总训练轮数')  # 从600降到300
    parser.add_argument('--batch-size', type=int, default=32, help='所有GPU的总批次大小，-1为自动批次')  # 从512降到128
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=512, help='训练、验证图像尺寸（像素）')  # 从640降到512
    parser.add_argument('--rect', action='store_true', help='矩形训练')
    parser.add_argument('--resume', nargs='?', const=False, default=False, help='恢复最近的训练')
    parser.add_argument('--nosave', action='store_true', help='只保存最终检查点')
    parser.add_argument('--noval', action='store_true', help='只验证最后一轮')
    parser.add_argument('--noautoanchor', action='store_true', help='禁用自动锚框')
    parser.add_argument('--noplots', action='store_true', help='不保存图表文件')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='进化超参数x代')
    parser.add_argument('--bucket', type=str, default='', help='gsutil 存储桶')
    parser.add_argument('--cache', type=str, nargs='?', const='disk', help='--cache 图像到"ram"（默认）或"disk"')  # 默认改为disk
    parser.add_argument('--image-weights', action='store_true', help='训练时使用加权图像选择')
    parser.add_argument('--device', default='0', help='cuda设备，即0或0,1,2,3或cpu')
    parser.add_argument('--multi-scale', action='store_true', help='变化图像尺寸+/-50%%')
    parser.add_argument('--single-cls', action='store_true', help='将多类数据作为单类训练')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='AdamW', help='优化器')  # 改为AdamW
    parser.add_argument('--sync-bn', action='store_true', help='使用同步批归一化，仅在DDP模式下可用')
    parser.add_argument('--workers', type=int, default=0, help='最大数据加载器工作进程数（DDP模式下每个RANK）')  # 改为0
    parser.add_argument('--project', default=ROOT / 'runs/train', help='保存到项目/名称')
    parser.add_argument('--name', default='binocular_exp', help='保存到项目/名称')  # 改个更有意义的名字
    parser.add_argument('--exist-ok', action='store_true', help='现有项目/名称确定，不递增')
    parser.add_argument('--quad', action='store_true', help='四元数据加载器')
    parser.add_argument('--cos-lr', action='store_true', help='余弦学习率调度器')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='标签平滑epsilon')  # 从0.0改为0.1
    parser.add_argument('--patience', type=int, default=50, help='早停耐心值（没有改进的轮数）')  # 从100降到50
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='冻结层：backbone=10，前3层=0 1 2')
    parser.add_argument('--save-period', type=int, default=25, help='每x轮保存检查点（<1时禁用）')  # 从-1改为25
    parser.add_argument('--seed', type=int, default=42, help='全局训练种子')  # 从0改为42
    parser.add_argument('--local_rank', type=int, default=-1, help='自动DDP多GPU参数，请勿修改')
    parser.add_argument('--mini', action='store_true', help='使用1/10数据进行快速训练')
    # 日志记录器参数
    parser.add_argument('--entity', default=None, help='实体')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='上传数据，"val"选项')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='设置边界框图像记录间隔')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='要使用的数据集工件版本')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    # 检查
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements()

    # 恢复训练（从指定的或最近的last.pt）
    if opt.resume and not check_wandb_resume(opt) and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  # 训练选项yaml
        opt_data = opt.data  # 原始数据集
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)  # 替换
        # 确保兼容新添加的 mini 选项
        if not hasattr(opt, 'mini'):
            opt.mini = False
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # 恢复
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # 避免HUB恢复认证超时
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # 检查
        assert len(opt.cfg) or len(opt.weights), '必须指定--cfg或--weights之一'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # 如果默认项目名称，重命名为runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # 将resume传递给exist_ok并禁用resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # 使用model.yaml作为名称
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP 模式
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = '与YOLOv5多GPU DDP训练不兼容'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}，请传递有效的--batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} 必须是WORLD_SIZE的倍数'
        assert torch.cuda.device_count() > LOCAL_RANK, 'DDP命令的CUDA设备不足'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # 训练
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)

    # 进化超参数（可选）
    else:
        # 超参数进化元数据（变异尺度0-1，下限，上限）
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # 初始学习率（SGD=1E-2，Adam=1E-3）
            'lrf': (1, 0.01, 1.0),  # 最终OneCycleLR学习率（lr0 * lrf）
            'momentum': (0.3, 0.6, 0.98),  # SGD动量/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # 优化器权重衰减
            'warmup_epochs': (1, 0.0, 5.0),  # 预热轮数（小数可以）
            'warmup_momentum': (1, 0.0, 0.95),  # 预热初始动量
            'warmup_bias_lr': (1, 0.0, 0.2),  # 预热初始偏置学习率
            'box': (1, 0.02, 0.2),  # 边界框损失增益
            'cls': (1, 0.2, 4.0),  # 分类损失增益
            'cls_pw': (1, 0.5, 2.0),  # 分类BCELoss正权重
            'obj': (1, 0.2, 4.0),  # 目标损失增益（根据像素缩放）
            'obj_pw': (1, 0.5, 2.0),  # 目标BCELoss正权重
            'iou_t': (0, 0.1, 0.7),  # IoU训练阈值
            'anchor_t': (1, 2.0, 8.0),  # 锚框倍数阈值
            'anchors': (2, 2.0, 10.0),  # 每个输出网格的锚框数（0表示忽略）
            'fl_gamma': (0, 0.0, 2.0),  # 焦点损失gamma（efficientDet默认gamma=1.5）
            'hsv_h': (1, 0.0, 0.1),  # 图像HSV-色调增强（分数）
            'hsv_s': (1, 0.0, 0.9),  # 图像HSV-饱和度增强（分数）
            'hsv_v': (1, 0.0, 0.9),  # 图像HSV-亮度增强（分数）
            'degrees': (1, 0.0, 45.0),  # 图像旋转（+/-度）
            'translate': (1, 0.0, 0.9),  # 图像平移（+/-分数）
            'scale': (1, 0.0, 0.9),  # 图像缩放（+/-增益）
            'shear': (1, 0.0, 10.0),  # 图像剪切（+/-度）
            'perspective': (0, 0.0, 0.001),  # 图像透视（+/-分数），范围0-0.001
            'flipud': (1, 0.0, 1.0),  # 图像垂直翻转（概率）
            'fliplr': (0, 0.0, 1.0),  # 图像水平翻转（概率）
            'mosaic': (1, 0.0, 1.0),  # 图像马赛克（概率）
            'mixup': (1, 0.0, 1.0),  # 图像混合（概率）
            'copy_paste': (1, 0.0, 1.0)}  # 分割复制粘贴（概率）

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # 加载超参数字典
            if 'anchors' not in hyp:  # hyp.yaml中anchors被注释
                hyp['anchors'] = 3
        if opt.noautoanchor:
            del hyp['anchors'], meta['anchors']
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # 只验证/保存最后一轮
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # 可进化索引
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}')  # 如果存在则下载evolve.csv

        for _ in range(opt.evolve):  # 进化代数
            if evolve_csv.exists():  # 如果evolve.csv存在：选择最佳超参数并变异
                # 选择父代
                parent = 'single'  # 父代选择方法：'single'或'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # 考虑的先前结果数量
                x = x[np.argsort(-fitness(x))][:n]  # 顶部n个变异
                w = fitness(x) - fitness(x).min() + 1E-6  # 权重（和>0）
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # 随机选择
                    x = x[random.choices(range(n), weights=w)[0]]  # 加权选择
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # 加权组合

                # 变异
                mp, s = 0.8, 0.2  # 变异概率，sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # 增益0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # 变异直到发生变化（防止重复）
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # 变异

            # 约束到限制
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # 下限
                hyp[k] = min(hyp[k], v[2])  # 上限
                hyp[k] = round(hyp[k], 5)  # 有效数字

            # 训练变异
            results = train(hyp.copy(), opt, device, callbacks)
            callbacks = Callbacks()
            # 写入变异结果
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)

        # 绘制结果
        plot_evolve(evolve_csv)
        LOGGER.info(f'超参数进化完成 {opt.evolve} 代\n'
                    f"结果保存到 {colorstr('bold', save_dir)}\n"
                    f'使用示例：$ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # 使用方法：import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)