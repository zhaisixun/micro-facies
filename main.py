import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner

from datasets import build_dataset
from engine import train_one_epoch, evaluate

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import models.convnext
import models.convnext_isotropic

from welllog.predict import save_predictions_csv, evaluate_full_well
from welllog.well_split import auto_split_wells


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='convnext_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")

    # EMA related parameters
    # 对模型的参数做平均，以求提高测试指标并增加模型鲁棒
    parser.add_argument('--model_ema', type=str2bool, default=False)   # 使用EMA（指数移动平均）
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')   # EMA衰减率
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')   # 强制将EMA模型移动到CPU
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')   # 使用EMA进行评估

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',  
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',   # 优化器epsilon
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',   # 梯度裁剪
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',   # 动量
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,  # 权重衰减
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',    # 学习率
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)  
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',  
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,  # 裁剪最小/最大比例
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,  # 切换到裁剪的概率
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',  # 混合模式
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    # parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
    #                     help='dataset path')
    # parser.add_argument('--eval_data_path', default=None, type=str,
    #                     help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)

    parser.add_argument('--data_set', default='WELLLOG_XLSX', choices=['CIFAR', 'IMNET', 'image_folder', 'WELLLOG_XLSX'],
                        type=str, help='ImageNet dataset path')
    # parser.add_argument('--xlsx_path', default='./facies-sand-gr-diff.xlsx', type=str,
    #                     help='xlsx path for WELLLOG_XLSX dataset')
    parser.add_argument('--xlsx_path', default='./xiang-gr-diff.xlsx', type=str,
                        help='xlsx path for WELLLOG_XLSX dataset')
    parser.add_argument('--feature_cols', default='GR,GR_diff1,GR_diff2,AC,PHIN', type=str,
                        help='comma-separated feature columns for WELLLOG_XLSX')
    parser.add_argument('--label_col', default='xiang', type=str,
                        help='label column for WELLLOG_XLSX')
    parser.add_argument('--train_wells', default='EP10-2-1,LF8-1-3,EP21-3-1,HZ26-6-1,KP11-4-3', type=str,
                        help='comma-separated train wells (sheet names) for WELLLOG_XLSX')
    parser.add_argument('--val_wells', default='LF8-1-1', type=str,
                        help='comma-separated val wells (sheet names) for WELLLOG_XLSX')
    parser.add_argument('--auto_split_wells', type=str2bool, default=True,
                        help='Auto-detect all wells from xlsx and split into train/val, '
                             'ignoring --train_wells / --val_wells. '
                             'Train set is guaranteed to cover all classes.')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Fraction of wells to use as val when --auto_split_wells is True '
                             '(default: 0.2)')
    parser.add_argument('--window_size', default=100, type=int,
                        help='sliding window size for WELLLOG_XLSX')
    parser.add_argument('--window_stride', default=0, type=int,   
                        help='sliding window stride for WELLLOG_XLSX; 0 = auto (window_size // 2)')   # 0 默认50%重叠
    parser.add_argument('--depth_col', default='DEPT', type=str,
                        help='depth column name in xlsx for WELLLOG_XLSX')
    parser.add_argument('--min_segment_length', default=5, type=int,
                        help='post-processing: merge predicted segments shorter than this '
                             'into the longer adjacent segment (unit: depth points). '
                             'Set to 1 to disable. Default 5 = 0.5m at 0.1m sampling.')
    parser.add_argument('--output_dir', default='./outputs/xiang_w100',        ########################
                        help='path where to save, empty for no saving')    
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)



    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')   # 恢复训练
    parser.add_argument('--auto_resume', type=str2bool, default=False)  # 自动恢复训练
    parser.add_argument('--save_ckpt', type=str2bool, default=True)  # 保存模型
    parser.add_argument('--save_ckpt_freq', default=1, type=int)  # 保存模型频率
    parser.add_argument('--save_ckpt_num', default=3, type=int)  # 保存模型数量

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',   # 
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')   # 只评估
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')   # 启用分布式评估
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')   # 禁用评估 during training
    parser.add_argument('--num_workers', default=16, type=int)   # 指定数据加载时使用的工作进程数
    parser.add_argument('--pin_mem', type=str2bool, default=True,   # 将CPU内存绑定到DataLoader，以提高效率
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,  
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)  # 
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=str2bool, default=False,  # 使用AMP（自动混合精度）
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")

    # Weights and Biases arguments
    parser.add_argument('--enable_wandb', type=str2bool, default=False,  
                        help="enable logging to Weights and Biases")
    parser.add_argument('--project', default='convnext', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--wandb_ckpt', type=str2bool, default=False,
                        help="Save model checkpoints as W&B Artifacts.")

    return parser

def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    if args.data_set == "WELLLOG_XLSX":
        if not args.xlsx_path:
            raise ValueError("--xlsx_path is required for WELLLOG_XLSX")
        if args.auto_split_wells:
            train_wells, val_wells = auto_split_wells(
                args.xlsx_path,
                label_col=args.label_col,
                val_ratio=args.val_ratio,
                seed=args.seed,
            )
            args.train_wells = ",".join(train_wells)
            args.val_wells   = ",".join(val_wells)
            print(f"[auto_split_wells] train ({len(train_wells)}): {train_wells}")
            print(f"[auto_split_wells] val   ({len(val_wells)}):   {val_wells}")
        elif not args.train_wells or not args.val_wells:
            raise ValueError(
                "--train_wells and --val_wells are required for WELLLOG_XLSX "
                "(or enable --auto_split_wells true)"
            )

    # fix the seed for reproducibility   # 固定随机种子，以提高可重复性
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)   
    np.random.seed(seed)
    cudnn.benchmark = True   

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)  # 构建训练集
    if args.disable_eval:   # 禁用评估
        args.dist_eval = False
        dataset_val = None   # 验证集为空
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)   # 构建验证集    

    num_tasks = utils.get_world_size()    # 总进程数
    global_rank = utils.get_rank()    # 当前进程编号

    sampler_train = torch.utils.data.DistributedSampler(            # 分布式采样器
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:   # 启用分布式评估
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(            # 分布式采样器
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)   # 顺序采样器

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if global_rank == 0 and args.enable_wandb:
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if args.data_set == "WELLLOG_XLSX":
        mixup_active = False
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = create_model(
        args.model, 
        pretrained=False, 
        in_chans=5,
        num_classes=args.nb_classes, 
        drop_path_rate=args.drop_path,
        layer_scale_init_value=args.layer_scale_init_value,
        head_init_scale=args.head_init_scale,
        )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        num_layers = 12 # convnext layers divided into 12 parts, each with a different decayed lr value.  
        assert args.model in ['convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'], \
             "Layer Decay impl only supports convnext_small/base/large/xlarge"   # 层衰减只支持convnext_small/base/large/xlarge
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)   # 分布式数据并行
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)

    loss_scaler = NativeScaler() # if args.use_amp is False, this won't be used

    print("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        return

    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        if wandb_logger:
            wandb_logger.set_steps()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, wandb_logger=wandb_logger, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            use_amp=args.use_amp
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        if data_loader_val is not None:
            is_welllog = (getattr(args, 'data_set', None) == 'WELLLOG_XLSX')
            if is_welllog and dataset_val is not None:
                test_stats = evaluate_full_well(
                    dataset_val, model, device,
                    use_amp=args.use_amp,
                    min_segment_length=getattr(args, 'min_segment_length', 1),
                )
                print(f"Full-well accuracy: {test_stats['acc1']:.2f}%")
            else:
                test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
                print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            # repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp)
                print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
                if max_accuracy_ema < test_stats_ema["acc1"]:
                    max_accuracy_ema = test_stats_ema["acc1"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                    print(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
                if log_writer is not None:
                    log_writer.update(test_acc1_ema=test_stats_ema['acc1'], head="perf", step=epoch)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)

    if wandb_logger and args.wandb_ckpt and args.save_ckpt and args.output_dir:
        wandb_logger.log_checkpoints()

    # After training: generate dense per-depth-point prediction CSV for val wells.
    # Load best checkpoint so predictions align with the best reported accuracy.
    if args.data_set == "WELLLOG_XLSX" and dataset_val is not None \
            and args.output_dir and utils.is_main_process():
        best_ckpt = os.path.join(args.output_dir, "checkpoint-best.pth")
        if os.path.exists(best_ckpt):
            best_state = torch.load(best_ckpt, map_location="cpu")
            ckpt_model = best_state.get("model", best_state)
            model_without_ddp.load_state_dict(ckpt_model)
            print(f"Loaded best checkpoint for prediction: {best_ckpt}")
        csv_path = os.path.join(args.output_dir, "predictions.csv")
        save_predictions_csv(dataset_val, model, device, csv_path,
                             use_amp=args.use_amp,
                             min_segment_length=args.min_segment_length)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)##############################
