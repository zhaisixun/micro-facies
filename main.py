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

from datasets import build_dataset, build_welllog_collate_fn
from engine import train_one_epoch, evaluate

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import models.convnext
import models.convnext1d
import models.convnext_isotropic

from welllog.predict import evaluate_full_well, evaluate_and_save_csv
from welllog.well_split import auto_split_wells, list_all_wells, resolve_eval_wells
from welllog.prototype import build_class_prototypes, save_prototypes, load_prototypes
from models.welllog_metric import WellLogMetricModel, adapt_checkpoint_state_dict
from welllog.metric_loss import SupervisedContrastiveLoss
from welllog.segmentation_loss import SegmentationLoss


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
    parser.add_argument('--input_size', default=128, type=int,    
                        help='image input size')
    parser.add_argument('--task_mode', default='segmentation',    #任务模式，图像分类还是语义分割
                        choices=['classification', 'segmentation'],
                        help='WELLLOG_XLSX task: center-point classification or dense 1D segmentation.')
    parser.add_argument('--seg_decoder', default='uper',   # 使用简单分割头 还是逐层解码器upernet
                        choices=['lite', 'uper'],
                        help='1D segmentation decoder: lite (single-scale head) or uper (UPerNet-style).')
    parser.add_argument('--decoder_channels', default=256, type=int,
                        help='Bottleneck channels for UPerNet-style 1D decoder.')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")
    parser.add_argument(
        '--dlka_stages', default=None, type=str,
        help='Replace 7x7 DwConv with deformable_LKA on these stages (0-3).'
             'Examples: "2,3" (recommended), "all", or omit for vanilla ConvNeXt.',
    )
    parser.add_argument('--use_1d_conv', type=str2bool, default=True,   # 是否使用一维卷积
                        help='WELLLOG_XLSX: use 1D ConvNeXt on (C, L) sequences instead of '
                             '2D ConvNeXt with bilinear stretch to (C, H, W). Default: False.')

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
    parser.add_argument('--class_weight', type=str2bool, default=True,  ############ 类别权重
                        help='Use inverse-frequency class weights in cross-entropy '
                             '(computed from training set class counts). '
                             'Ignored when --class_weights is set.')
    parser.add_argument('--class_weights', default='', type=str,
                        help='Manual per-class CE weights, comma-separated by class index '
                             '(e.g. "1,8,8,1" for 4 classes). Overrides --class_weight auto weights.')
    parser.add_argument('--class_weights_normalize', type=str2bool, default=True,
                        help='When using --class_weights, divide by mean so average weight is 1.')
    parser.add_argument('--window_purity_thresh', type=float, default=0.0,   ############ 窗口纯度硬过滤
                        help='Discard training windows whose purity (fraction of center-class points) '
                             'is below this threshold. 0.0 = disabled (keep all windows). '
                             'Ignored when --window_require_pure is true.')
    parser.add_argument('--purity_weight_power', type=float, default=0.0,    ############ 纯度软加权
                        help='Weight per-sample CE loss by purity^power. '
                             '0.0 = disabled (uniform weight), 1.0 = linear downweight at boundaries.')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--loss_mode', default='ce_focal_dice',
                        choices=['ce', 'ce_focal', 'ce_dice', 'ce_focal_dice'],
                        help='Segmentation loss composition when --task_mode segmentation.')
    parser.add_argument('--ignore_index', default=-100, type=int,
                        help='Ignore label index for invalid points in segmentation windows.')
    parser.add_argument('--focal_gamma', default=2.0, type=float,
                        help='Focal loss gamma for segmentation.')
    parser.add_argument('--ce_weight', default=1.0, type=float,
                        help='CE loss weight for segmentation.')
    parser.add_argument('--focal_weight', default=2.0, type=float,
                        help='Focal loss weight for segmentation.')
    parser.add_argument('--dice_weight', default=0.5, type=float,
                        help='Dice loss weight for segmentation.')
    parser.add_argument('--seg_oversample', type=str2bool, default=False,
                        help='Oversample training windows that contain minority classes '
                             'when --task_mode segmentation (single-GPU only).')
    parser.add_argument('--seg_oversample_boost', type=float, default=5.0,
                        help='Extra multiplier for windows containing rare classes '
                             '(point freq < 5%% of training points) when seg_oversample is on.')
    parser.add_argument('--best_metric', default='', choices=['', 'acc1', 'miou'],
                        help='Metric for saving checkpoint-best. '
                             'Default: miou for segmentation, acc1 for classification.')

    # Supervised contrastive learning (metric learning)
    parser.add_argument('--use_supcon', type=str2bool, default=False,
                        help='Enable CE + supervised contrastive (SupCon) joint training.')
    parser.add_argument('--supcon_weight', type=float, default=0.5,
                        help='Weight for SupCon loss: L = L_CE + supcon_weight * L_SupCon.')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Projection head output dimension for SupCon.')
    parser.add_argument('--supcon_temperature', type=float, default=0.07,
                        help='Temperature for SupCon loss.')
    parser.add_argument('--balanced_sampler', type=str2bool, default=True,
                        help='When use_supcon=True, oversample minority classes in training '
                             '(single-GPU only; disabled under distributed training). '
                             'For segmentation, prefer --seg_oversample.')
    parser.add_argument('--infer_mode', default='linear', choices=['linear', 'prototype'],   # 推理模式。普通是linear推理，使用supcon loss时用 prototype推理。
                        help='Full-well inference: linear head argmax or nearest class prototype.')
    parser.add_argument('--prototype_cache', default='', type=str,
                        help='Path to class_prototypes.pt; default: output_dir/class_prototypes.pt')

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



    # 数据集/输入配置
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
    parser.add_argument('--xlsx_path', default='./facies-gr-diff0614-用GR-CNL-DEN.xlsx', type=str,   # 数据集路径
                        help='xlsx path for WELLLOG_XLSX dataset')
    parser.add_argument('--feature_cols', default='GR,CNL,DEN', type=str,  # 特征列
                        help='comma-separated feature columns for WELLLOG_XLSX')
    parser.add_argument('--label_col', default='facies', type=str,  # 标签列
                        help='label column for WELLLOG_XLSX')
    parser.add_argument('--train_wells', default='EP10-2-1,LF8-1-3,EP21-3-1,HZ26-6-1,KP11-4-3', type=str,
                        help='comma-separated train wells (sheet names) for WELLLOG_XLSX')   # 手动指定训练wells
    parser.add_argument('--val_wells', default='LF8-1-1', type=str,
                        help='comma-separated val wells (sheet names) for WELLLOG_XLSX')   # 手动指定验证wells
    parser.add_argument('--eval_wells', default='', type=str,   # 全部训练测试
                        help='Eval wells override for WELLLOG_XLSX. '
                             '"same_as_train" = evaluate on training wells; '
                             '"all" = use every well in xlsx for both train and eval; '
                             'or comma-separated sheet names. Overrides --val_wells when set.')
    parser.add_argument('--auto_split_wells', type=str2bool, default=True,    # 自动划分训练测试wells
                        help='Auto-detect all wells from xlsx and split into train/val, '
                             'ignoring --train_wells / --val_wells. '
                             'Train set is guaranteed to cover all classes.')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Fraction of wells to use as val when --auto_split_wells is True '
                             '(default: 0.2)')
    parser.add_argument('--window_size', default=128, type=int,      # 原始是128 
                        help='sliding window size for WELLLOG_XLSX')   # 滑动窗口大小
    parser.add_argument('--well_input_mode', default='sliding_window',
                        choices=['sliding_window', 'whole_well'],
                        help='WELLLOG_XLSX input strategy: fixed sliding windows or one variable-length sample per well.')
    parser.add_argument('--whole_well_max_length', default=0, type=int,
                        help='Optional max sequence length in whole_well mode. '
                             '0 = use full well; >0 crops/pads training wells to this length '
                             '(train=random crop, val/eval=keep head segment).')
    parser.add_argument('--use_pad_mask', type=str2bool, default=True,
                        help='In whole_well mode, mask padded positions in model forward '
                             'so zero padding does not affect convolutions.')
    parser.add_argument('--window_stride', default=0, type=int,   
                        help='sliding window stride for WELLLOG_XLSX; 0 = auto '
                             '(segmentation: window_size // 4, classification: window_size // 2)')   # 滑动窗口步长
    parser.add_argument('--depth_col', default='DEPT', type=str,
                        help='depth column name in xlsx for WELLLOG_XLSX')
    

    # resmote 过采样
    # parser.add_argument('--resmote', type=str2bool, default=False,
    #                     help='Enable well-constrained RESMOTE on training wells only '
    #                          '(sliding_window mode).')
    # parser.add_argument('--resmote_classes', default='1,2', type=str,
    #                     help='Comma-separated minority class indices to augment (e.g. "1,2").')
    # parser.add_argument('--resmote_target_ratio', type=float, default=0.15,
    #                     help='Target point count per minority class = majority_count * ratio '
    #                          'when --resmote_target_count is 0.')
    # parser.add_argument('--resmote_target_count', default=0, type=int,
    #                     help='Absolute target point count per minority class; overrides ratio when > 0.')
    # parser.add_argument('--resmote_k', default=5, type=int,
    #                     help='k neighbors for well-constrained SMOTE/RESMOTE.')
    # parser.add_argument('--resmote_depth_radius', default=5.0, type=float,
    #                     help='Max depth distance (m) for RESMOTE neighbor search within a well.')
    # parser.add_argument('--resmote_iterations', default=3, type=int,
    #                     help='RESMOTE repeat iterations per minority class.')
    # parser.add_argument('--resmote_lof', type=str2bool, default=False,
    #                     help='Apply LOF outlier removal on minority points before RESMOTE.')
    # parser.add_argument('--resmote_lof_contamination', default=0.05, type=float,
    #                     help='LOF contamination rate when --resmote_lof is enabled.')
    # parser.add_argument('--resmote_seed', default=-1, type=int,
    #                     help='Random seed for RESMOTE; -1 uses --seed.')
    parser.add_argument('--min_segment_length', default=1, type=int,    ########### 后处理，合并预测段中小于此长度的段
                        help='post-processing: merge predicted segments shorter than this '
                             'into the longer adjacent segment (unit: depth points). '
                             'Set to 1 to disable. Default 5 = 0.5m at 0.1m sampling.')
    parser.add_argument('--infer_batch_size', default=256, type=int,
                        help='batch size for full-well inference after training (default: 256). '
                             'Larger values are faster but use more GPU memory.')
    parser.add_argument('--infer_stride', default=0, type=int,
                        help='Full-well segmentation inference stride; 0 = window_size // 4.')
    parser.add_argument('--infer_fusion', default='weighted_center',
                        choices=['mean', 'weighted_center', 'vote'],
                        help='Overlap fusion strategy for segmentation inference.')
    parser.add_argument('--output_dir', default='./outputs/seg_w128_0614',        ########################
                        help='path where to save, empty for no saving')    
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)


    # 训练相关
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
                        help='Disabling evaluation during training')
    parser.add_argument('--eval_full_well_each_epoch', type=str2bool, default=False,
                        help='WELLLOG_XLSX: run slow full-well point-wise eval every epoch. '
                             'Default False uses batched val-loader eval for best-ckpt selection; '
                             'full-well CSV is still generated after training.')
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


def _num_feature_cols(args):
    return len([c.strip() for c in args.feature_cols.split(",") if c.strip()])


_WELLLOG_MODEL_1D_MAP = {
    "convnext_tiny": "convnext1d_tiny",
    "convnext_small": "convnext1d_small",
    "convnext_base": "convnext1d_base",
    "convnext_large": "convnext1d_large",
    "convnext_tiny_dlka": "convnext1d_tiny",
    "convnext_small_dlka": "convnext1d_small",
    "convnext_base_dlka": "convnext1d_base",
}

_WELLLOG_MODEL_1D_SEG_LITE_MAP = {
    "convnext_tiny": "convnext1d_tiny_seg",
    "convnext_small": "convnext1d_small_seg",
    "convnext_base": "convnext1d_base_seg",
    "convnext_large": "convnext1d_large_seg",
    "convnext_tiny_dlka": "convnext1d_tiny_seg",
    "convnext_small_dlka": "convnext1d_small_seg",
    "convnext_base_dlka": "convnext1d_base_seg",
    "convnext1d_tiny": "convnext1d_tiny_seg",
    "convnext1d_small": "convnext1d_small_seg",
    "convnext1d_base": "convnext1d_base_seg",
    "convnext1d_large": "convnext1d_large_seg",
}

_WELLLOG_MODEL_1D_SEG_UPER_MAP = {
    "convnext_tiny": "convnext1d_tiny_uper_seg",
    "convnext_small": "convnext1d_small_uper_seg",
    "convnext_base": "convnext1d_base_uper_seg",
    "convnext_large": "convnext1d_large_uper_seg",
    "convnext_tiny_dlka": "convnext1d_tiny_uper_seg",
    "convnext_small_dlka": "convnext1d_small_uper_seg",
    "convnext_base_dlka": "convnext1d_base_uper_seg",
    "convnext1d_tiny": "convnext1d_tiny_uper_seg",
    "convnext1d_small": "convnext1d_small_uper_seg",
    "convnext1d_base": "convnext1d_base_uper_seg",
    "convnext1d_large": "convnext1d_large_uper_seg",
    "convnext1d_tiny_seg": "convnext1d_tiny_uper_seg",
    "convnext1d_small_seg": "convnext1d_small_uper_seg",
    "convnext1d_base_seg": "convnext1d_base_uper_seg",
    "convnext1d_large_seg": "convnext1d_large_uper_seg",
}


def _resolve_welllog_model(args):
    """Pick 2D or 1D ConvNeXt variant according to --use_1d_conv."""
    model_name = args.model
    if getattr(args, "task_mode", "classification") == "segmentation":
        if not getattr(args, "use_1d_conv", False):
            raise ValueError("task_mode=segmentation currently requires --use_1d_conv true.")
        seg_decoder = getattr(args, "seg_decoder", "uper")
        if model_name.endswith("_uper_seg"):
            return model_name, None
        if model_name.endswith("_seg") and seg_decoder == "lite":
            return model_name, None
        seg_map = (
            _WELLLOG_MODEL_1D_SEG_UPER_MAP
            if seg_decoder == "uper"
            else _WELLLOG_MODEL_1D_SEG_LITE_MAP
        )
        mapped = seg_map.get(model_name)
        if mapped is None:
            raise ValueError(
                f"--task_mode segmentation but --model '{model_name}' has no segmentation counterpart "
                f"for seg_decoder={seg_decoder}. "
                f"Use one of: {sorted(seg_map)} or convnext1d_*_{'uper_' if seg_decoder == 'uper' else ''}seg directly."
            )
        warnings = []
        if args.dlka_stages is not None:
            warnings.append("dlka_stages is ignored in 1D segmentation mode (DLKA is 2D-only)")
        if model_name != mapped:
            warnings.append(f"mapped {model_name} -> {mapped} (seg_decoder={seg_decoder})")
        return mapped, warnings

    if not getattr(args, "use_1d_conv", False):
        return model_name, None

    if model_name.startswith("convnext1d_"):
        return model_name, None

    mapped = _WELLLOG_MODEL_1D_MAP.get(model_name)
    if mapped is None:
        raise ValueError(
            f"--use_1d_conv true but --model '{model_name}' has no 1D counterpart. "
            f"Use one of: {sorted(_WELLLOG_MODEL_1D_MAP)} or convnext1d_* directly."
        )

    warnings = []
    if args.dlka_stages is not None:
        warnings.append("dlka_stages is ignored in 1D mode (DLKA is 2D-only)")
    if model_name.endswith("_dlka"):
        warnings.append(f"mapped {model_name} -> {mapped} without DLKA")
    elif model_name != mapped:
        warnings.append(f"mapped {model_name} -> {mapped}")
    return mapped, warnings


def _prototype_path(args):
    if args.prototype_cache:
        return args.prototype_cache
    if args.output_dir:
        return os.path.join(args.output_dir, "class_prototypes.pt")
    return "class_prototypes.pt"


def _build_or_load_prototypes(args, model_without_ddp, dataset_train, device, rebuild=True):
    """Build class prototypes from training set, or load from cache when rebuild=False."""
    path = _prototype_path(args)
    if not rebuild and os.path.isfile(path):
        print(f"Loading class prototypes from {path}")
        prototypes, _ = load_prototypes(path, device)
        return prototypes, path

    print("Building class prototypes from training set...")
    prototypes, counts = build_class_prototypes(
        model_without_ddp,
        dataset_train,
        device,
        batch_size=min(args.batch_size * 2, 256),
        num_workers=args.num_workers,
        use_amp=args.use_amp,
    )
    if utils.is_main_process():
        save_prototypes(
            path,
            prototypes,
            dataset_train.label_map,
            counts=counts,
            embedding_dim=args.embedding_dim if args.use_supcon else prototypes.shape[1],
        )
        inv_map = dataset_train.inv_label_map
        for k in range(len(dataset_train.label_map)):
            name = inv_map.get(k, k)
            print(f"  prototype class {name}: {int(counts[k].item())} train samples")
        print(f"Class prototypes saved -> {path}")
    return prototypes, path

def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    if args.data_set == "WELLLOG_XLSX":
        if not args.xlsx_path:
            raise ValueError("--xlsx_path is required for WELLLOG_XLSX")
        eval_wells_key = str(getattr(args, "eval_wells", "") or "").strip().lower()
        if eval_wells_key == "all":
            if args.auto_split_wells:
                print("[eval_wells=all] Ignoring --auto_split_wells; using all wells for train and eval.")
            all_wells = list_all_wells(args.xlsx_path, args.label_col)
            args.train_wells = ",".join(all_wells)
            args.val_wells = args.train_wells
            print(f"[eval_wells=all] train/eval ({len(all_wells)}): {all_wells}")
        elif args.auto_split_wells:
            train_wells, val_wells = auto_split_wells(
                args.xlsx_path,
                label_col=args.label_col,
                val_ratio=args.val_ratio,
                seed=args.seed,
            )
            args.train_wells = ",".join(train_wells)
            args.val_wells = ",".join(val_wells)
            print(f"[auto_split_wells] train ({len(train_wells)}): {train_wells}")
            print(f"[auto_split_wells] val   ({len(val_wells)}):   {val_wells}")
        elif not args.train_wells:
            raise ValueError(
                "--train_wells is required for WELLLOG_XLSX when "
                "--auto_split_wells is false and --eval_wells is not 'all'."
            )

        if eval_wells_key and eval_wells_key != "all":
            train_list = [w.strip() for w in args.train_wells.split(",") if w.strip()]
            val_list = resolve_eval_wells(args.eval_wells, train_list)
            args.val_wells = ",".join(val_list)
            if eval_wells_key == "same_as_train":
                print(f"[eval_wells=same_as_train] eval on train wells ({len(val_list)}): {val_list}")
            else:
                print(f"[eval_wells] eval wells ({len(val_list)}): {val_list}")
        elif not eval_wells_key and not args.auto_split_wells and not args.val_wells:
            raise ValueError(
                "--val_wells is required for WELLLOG_XLSX when "
                "--auto_split_wells is false and --eval_wells is not set."
            )
        if args.task_mode == "segmentation":
            if not args.use_1d_conv:
                raise ValueError("WELLLOG_XLSX segmentation currently supports only --use_1d_conv true.")
            well_input_mode = getattr(args, "well_input_mode", "sliding_window")
            if well_input_mode == "whole_well":
                if args.input_size != args.window_size:
                    print(
                        "[whole_well] input_size/window_size are ignored; "
                        "each batch item uses the full (or cropped) well length."
                    )
            elif args.input_size != args.window_size:
                print(
                    "[segmentation] input_size must match window_size for point-wise label alignment; "
                    f"setting input_size {args.input_size} -> {args.window_size}."
                )
                args.input_size = args.window_size
            if args.use_supcon:
                print("[segmentation] SupCon is sample-level in this codebase; disabling --use_supcon.")
                args.use_supcon = False
            if args.infer_mode != "linear":
                print("[segmentation] Prototype inference is not supported; using infer_mode=linear.")
                args.infer_mode = "linear"
        if getattr(args, "well_input_mode", "sliding_window") == "whole_well":
            if args.task_mode != "segmentation":
                raise ValueError("well_input_mode=whole_well currently requires --task_mode segmentation.")
            if not args.use_1d_conv:
                raise ValueError("well_input_mode=whole_well requires --use_1d_conv true.")
            if args.seg_oversample:
                print("[whole_well] seg_oversample is disabled (one sample per well).")
                args.seg_oversample = False
            if args.batch_size > 8:
                print(
                    f"[whole_well] Recommend smaller batch_size for long sequences "
                    f"(current batch_size={args.batch_size})."
                )
        if args.infer_mode == "prototype" and not args.use_supcon:
            print(
                "Warning: infer_mode=prototype without use_supcon — "
                "prototypes will use backbone features (no projection head)."
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

    if args.use_supcon and args.batch_size < args.nb_classes:
        print(
            f"Warning: batch_size ({args.batch_size}) < nb_classes ({args.nb_classes}); "
            "SupCon may lack positive pairs for some classes."
        )

    num_tasks = utils.get_world_size()    # 总进程数
    global_rank = utils.get_rank()    # 当前进程编号

    args.best_metric = utils.resolve_best_metric(args.task_mode, args.best_metric)
    print(f"Best checkpoint metric: {args.best_metric}")

    use_seg_oversample = (
        args.task_mode == "segmentation"
        and args.seg_oversample
        and args.data_set == "WELLLOG_XLSX"
        and num_tasks == 1
    )
    use_supcon_balanced_sampler = (
        args.use_supcon
        and args.balanced_sampler
        and args.data_set == "WELLLOG_XLSX"
        and num_tasks == 1
        and not use_seg_oversample
    )
    if args.seg_oversample and args.task_mode == "segmentation" and num_tasks > 1:
        print(
            "Warning: seg_oversample disabled under distributed training; "
            "using DistributedSampler instead."
        )
    if args.use_supcon and args.balanced_sampler and num_tasks > 1:
        print(
            "Warning: balanced_sampler disabled under distributed training; "
            "using DistributedSampler instead."
        )

    if use_seg_oversample:
        sample_weights, boosted = utils.build_welllog_segmentation_sample_weights(
            dataset_train,
            boost=args.seg_oversample_boost,
        )
        print(
            f"Sampler_train = WeightedRandomSampler (segmentation oversample, "
            f"boost={args.seg_oversample_boost}, rare windows={boosted}/{len(sample_weights)})"
        )
        sampler_train = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True,
        )
    elif use_supcon_balanced_sampler:
        sample_weights = utils.build_welllog_sample_weights(dataset_train)
        sampler_train = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True,
        )
        print("Sampler_train = WeightedRandomSampler (balanced for SupCon)")
    else:
        sampler_train = torch.utils.data.DistributedSampler(
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

    welllog_collate_fn = (
        build_welllog_collate_fn(args)
        if getattr(args, "data_set", None) == "WELLLOG_XLSX"
        else None
    )
    drop_last_train = welllog_collate_fn is None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=drop_last_train,
        shuffle=False,
        collate_fn=welllog_collate_fn,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            collate_fn=welllog_collate_fn,
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

    dlka_stages = None
    if args.dlka_stages is not None and not args.use_1d_conv:
        s = args.dlka_stages.strip().lower()
        if s in ("all", "true", "1"):
            dlka_stages = "all"
        else:
            dlka_stages = tuple(int(x) for x in s.split(","))

    model_name, model_warnings = _resolve_welllog_model(args)
    args.resolved_model = model_name
    for msg in model_warnings or []:
        print(f"[use_1d_conv] Warning: {msg}")
    if args.use_1d_conv:
        seg_info = (
            f", seg_decoder={args.seg_decoder}"
            if getattr(args, "task_mode", "classification") == "segmentation"
            else ""
        )
        print(
            f"[use_1d_conv] Enabled: model={model_name}, "
            f"input shape (C, L){seg_info}, well_input_mode={getattr(args, 'well_input_mode', 'sliding_window')}"
        )
    else:
        print(
            f"[use_1d_conv] Disabled: model={model_name}, "
            f"input shape (C, {args.input_size}, {args.input_size})"
        )

    model_kwargs = dict(
        in_chans=_num_feature_cols(args),
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        layer_scale_init_value=args.layer_scale_init_value,
        head_init_scale=args.head_init_scale,
        dlka_stages=dlka_stages,
    )
    if getattr(args, "task_mode", "classification") == "segmentation" and getattr(args, "seg_decoder", "uper") == "uper":
        model_kwargs["decoder_channels"] = args.decoder_channels

    backbone = create_model(
        model_name,
        pretrained=False,
        **model_kwargs,
    )

    if args.use_supcon:
        model = WellLogMetricModel(
            backbone,
            embedding_dim=args.embedding_dim,
            use_supcon=True,
        )
        print(f"WellLogMetricModel enabled (SupCon, embedding_dim={args.embedding_dim})")
    else:
        model = backbone

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
        if args.use_supcon:
            checkpoint_model = adapt_checkpoint_state_dict(checkpoint_model)
        state_dict = model.state_dict()
        for k in list(checkpoint_model.keys()):
            if k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
        if args.use_supcon:
            print("Note: projection_head weights are randomly initialized when loading a non-SupCon checkpoint.")
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
    if len(dataset_train) > 0:
        num_training_steps_per_epoch = max(
            1, (len(dataset_train) + total_batch_size - 1) // total_batch_size
        )
    else:
        num_training_steps_per_epoch = 0
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        num_layers = 12 # convnext layers divided into 12 parts, each with a different decayed lr value.  
        assert args.resolved_model in [
            'convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge',
            'convnext1d_small', 'convnext1d_base', 'convnext1d_large',
            'convnext1d_tiny_seg', 'convnext1d_small_seg', 'convnext1d_base_seg', 'convnext1d_large_seg',
            'convnext1d_tiny_uper_seg', 'convnext1d_small_uper_seg',
            'convnext1d_base_uper_seg', 'convnext1d_large_uper_seg',
        ], "Layer Decay impl only supports convnext_small/base/large/xlarge and convnext1d_*"
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

    class_weight_tensor = None
    inv_map = getattr(dataset_train, 'inv_label_map', {})
    manual_weights = getattr(args, 'class_weights', '') or ''
    if manual_weights.strip():
        class_weight_tensor = utils.parse_manual_class_weights(
            manual_weights,
            args.nb_classes,
            normalize=args.class_weights_normalize,
        ).to(device)
        weight_info = {
            inv_map.get(i, i): round(class_weight_tensor[i].item(), 4)
            for i in range(args.nb_classes)
        }
        print(
            f"Class weights (manual, normalize={args.class_weights_normalize}): {weight_info}"
        )
    elif args.class_weight and hasattr(dataset_train, 'class_counts'):
        class_weight_tensor = utils.compute_class_weights(
            dataset_train.class_counts, args.nb_classes).to(device)
        weight_info = {
            inv_map.get(i, i): round(class_weight_tensor[i].item(), 4)
            for i in range(args.nb_classes)
        }
        print(f"Class weights (inverse freq, mean=1): {weight_info}")
    elif args.class_weight:
        print("Warning: --class_weight set but dataset has no class_counts; using unweighted loss.")

    if args.task_mode == "segmentation":
        criterion = SegmentationLoss(
            mode=args.loss_mode,
            weight=class_weight_tensor,
            ignore_index=args.ignore_index,
            focal_gamma=args.focal_gamma,
            ce_weight=args.ce_weight,
            focal_weight=args.focal_weight,
            dice_weight=args.dice_weight,
        )
    elif mixup_fn is not None:
        if class_weight_tensor is not None:
            print("Warning: class weights are ignored when mixup/cutmix is enabled.")
        criterion = SoftTargetCrossEntropy()
    elif class_weight_tensor is not None:
        criterion = utils.WeightedCrossEntropyLoss(
            weight=class_weight_tensor,
            label_smoothing=args.smoothing if args.smoothing > 0 else 0.0,
        )
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    supcon_criterion = None
    if args.use_supcon:
        supcon_criterion = SupervisedContrastiveLoss(temperature=args.supcon_temperature)
        print(
            f"SupCon enabled: weight={args.supcon_weight}, "
            f"temperature={args.supcon_temperature}, embedding_dim={args.embedding_dim}"
        )

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    def _prototypes_for_infer(rebuild=True):
        if args.infer_mode != "prototype":
            return None
        proto, _ = _build_or_load_prototypes(
            args, model_without_ddp, dataset_train, device, rebuild=rebuild,
        )
        return proto

    if args.eval:  # 评估模式
        print("Eval only mode")
        is_welllog = getattr(args, 'data_set', None) == 'WELLLOG_XLSX'
        if is_welllog and dataset_val is not None:
            if args.task_mode == "segmentation":
                report_name = "segmentation_report.txt"
            else:
                report_name = (
                    "classification_report_prototype.txt"
                    if args.infer_mode == "prototype"
                    else "classification_report.txt"
                )
            report_path = (
                os.path.join(args.output_dir, report_name)
                if args.output_dir else None
            )
            test_stats = evaluate_full_well(
                dataset_val, model, device,
                use_amp=args.use_amp,
                min_segment_length=getattr(args, 'min_segment_length', 1),
                infer_batch_size=getattr(args, 'infer_batch_size', 256),
                report_path=report_path,
                infer_mode=args.infer_mode,
                prototypes=_prototypes_for_infer(
                    rebuild=not os.path.isfile(_prototype_path(args))
                ),
                infer_stride=args.infer_stride,
                infer_fusion=args.infer_fusion,
            )
            if "miou" in test_stats:
                print(f"Full-well accuracy: {test_stats['acc1']:.5f}%  mIoU: {test_stats['miou']:.5f}%")
            else:
                print(f"Full-well accuracy: {test_stats['acc1']:.5f}%")
        else:
            test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp, criterion=criterion,
                                  use_pad_mask=getattr(args, 'use_pad_mask', True))
            print(f"Accuracy on {len(dataset_val)} val samples: {test_stats['acc1']:.5f}%")
        return

    max_best_score = 0.0
    if args.model_ema and args.model_ema_eval:
        max_best_score_ema = 0.0

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and isinstance(
            data_loader_train.sampler, torch.utils.data.DistributedSampler
        ):
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
            use_amp=args.use_amp,
            use_supcon=args.use_supcon,
            supcon_criterion=supcon_criterion,
            supcon_weight=args.supcon_weight,
            use_pad_mask=getattr(args, 'use_pad_mask', True),
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        if data_loader_val is not None:
            is_welllog = getattr(args, 'data_set', None) == 'WELLLOG_XLSX'
            use_full_well = is_welllog and args.eval_full_well_each_epoch
            if use_full_well and dataset_val is not None:
                test_stats = evaluate_full_well(
                    dataset_val, model, device,
                    use_amp=args.use_amp,
                    min_segment_length=getattr(args, 'min_segment_length', 1),
                    infer_batch_size=getattr(args, 'infer_batch_size', 256),
                    infer_mode=args.infer_mode,
                    prototypes=_prototypes_for_infer(rebuild=True),
                    infer_stride=args.infer_stride,
                    infer_fusion=args.infer_fusion,
                )
                if "miou" in test_stats:
                    print(f"Full-well accuracy: {test_stats['acc1']:.2f}%  mIoU: {test_stats['miou']:.2f}%")
                else:
                    print(f"Full-well accuracy: {test_stats['acc1']:.2f}%")
            else:
                test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp, criterion=criterion,
                                  use_pad_mask=getattr(args, 'use_pad_mask', True))
                if is_welllog:
                    print(
                        f"Val batch accuracy (fast): {test_stats['acc1']:.2f}% "
                        f"on {len(dataset_val)} window samples"
                    )
                else:
                    print(
                        f"Accuracy of the model on the {len(dataset_val)} "
                        f"test images: {test_stats['acc1']:.1f}%"
                    )
            epoch_score = utils.checkpoint_score(test_stats, args.best_metric)
            if max_best_score < epoch_score:
                max_best_score = epoch_score
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            print(f'Max {args.best_metric}: {max_best_score:.2f}%')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)
                if "miou" in test_stats:
                    log_writer.update(test_miou=test_stats['miou'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            # repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp, criterion=criterion,
                                          use_pad_mask=getattr(args, 'use_pad_mask', True))
                print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
                epoch_score_ema = utils.checkpoint_score(test_stats_ema, args.best_metric)
                if max_best_score_ema < epoch_score_ema:
                    max_best_score_ema = epoch_score_ema
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                    print(f'Max EMA {args.best_metric}: {max_best_score_ema:.2f}%')
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

    # After training (WELLLOG): load best ckpt, full-well eval + dense CSV (original logic).
    if args.data_set == "WELLLOG_XLSX" and dataset_val is not None \
            and args.output_dir and utils.is_main_process():
        best_ckpt = os.path.join(args.output_dir, "checkpoint-best.pth")
        if os.path.exists(best_ckpt):
            best_state = torch.load(best_ckpt, map_location="cpu")
            ckpt_model = best_state.get("model", best_state)
            if args.use_supcon:
                ckpt_model = adapt_checkpoint_state_dict(ckpt_model)
            model_without_ddp.load_state_dict(ckpt_model, strict=False)
            print(f"Loaded best checkpoint for full-well export: {best_ckpt}")
        else:
            print("No checkpoint-best.pth found; using final epoch weights for export.")
        export_prototypes = _prototypes_for_infer(rebuild=True)
        csv_path = os.path.join(args.output_dir, "predictions.csv")
        fw_stats = evaluate_and_save_csv(
            dataset_val, model, device, csv_path,
            use_amp=args.use_amp,
            min_segment_length=args.min_segment_length,
            infer_batch_size=getattr(args, 'infer_batch_size', 256),
            infer_mode=args.infer_mode,
            prototypes=export_prototypes,
            infer_stride=args.infer_stride,
            infer_fusion=args.infer_fusion,
        )
        print(
            f"Final full-well accuracy (best ckpt, infer_mode={args.infer_mode}): "
            f"{fw_stats['acc1']:.2f}%"
            + (f"  mIoU: {fw_stats['miou']:.2f}%" if "miou" in fw_stats else "")
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)##############################
