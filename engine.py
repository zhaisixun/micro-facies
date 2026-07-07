# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _point_accuracy(output: torch.Tensor, target: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    pred = output.argmax(dim=1)
    valid = target != ignore_index
    if not valid.any():
        return output.new_zeros(())
    return (pred[valid] == target[valid]).float().mean()


def _mean_iou(output: torch.Tensor, target: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    pred = output.argmax(dim=1)
    valid = target != ignore_index
    if not valid.any():
        return output.new_zeros(())

    num_classes = output.shape[1]
    ious = []
    pred = pred[valid]
    target = target[valid]
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        union = pred_mask.logical_or(target_mask).sum()
        if union == 0:
            continue
        intersection = pred_mask.logical_and(target_mask).sum()
        ious.append(intersection.float() / union.float())
    if not ious:
        return output.new_zeros(())
    return torch.stack(ious).mean()


def _batch_lengths(batch, device: torch.device, use_pad_mask: bool = True):
    if not use_pad_mask or len(batch) < 4:
        return None
    return batch[3].to(device, non_blocking=True)


def _model_forward(model: torch.nn.Module, samples: torch.Tensor, lengths=None):
    if lengths is None:
        return model(samples)
    return model(samples, lengths=lengths)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False,
                    use_supcon: bool = False, supcon_criterion=None, supcon_weight: float = 0.5,
                    use_pad_mask: bool = True):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if use_supcon:
        metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=10, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_supcon', utils.SmoothedValue(window_size=10, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()
    core_model = _unwrap_model(model)
    supcon_active = use_supcon and supcon_criterion is not None and hasattr(core_model, "forward_train")

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # batch 为三元组 (samples, targets, sample_weights)
        samples, targets, sample_weights = batch[0], batch[1], batch[2]
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        sample_weights = sample_weights.to(device, non_blocking=True)
        lengths = _batch_lengths(batch, device, use_pad_mask=use_pad_mask)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        loss_ce_value = None
        loss_sc_value = None

        if supcon_active and mixup_fn is None:
            if use_amp:
                with torch.cuda.amp.autocast():
                    output, z_proj = core_model.forward_train(samples)
                    loss_ce = (utils.get_per_sample_loss(criterion, output, targets) * sample_weights).mean()
                    loss_sc = supcon_criterion(z_proj, targets)
                    loss = loss_ce + supcon_weight * loss_sc
            else:
                output, z_proj = core_model.forward_train(samples)
                loss_ce = (utils.get_per_sample_loss(criterion, output, targets) * sample_weights).mean()
                loss_sc = supcon_criterion(z_proj, targets)
                loss = loss_ce + supcon_weight * loss_sc
            loss_ce_value = loss_ce.item()
            loss_sc_value = loss_sc.item()
        else:
            if use_amp:
                with torch.cuda.amp.autocast():
                    output = _model_forward(model, samples, lengths=lengths)
                    if output.ndim == 3 and targets.ndim == 2:
                        loss = criterion(output, targets, sample_weight=sample_weights)
                    else:
                        loss = (utils.get_per_sample_loss(criterion, output, targets) * sample_weights).mean()
            else:
                output = _model_forward(model, samples, lengths=lengths)
                if output.ndim == 3 and targets.ndim == 2:
                    loss = criterion(output, targets, sample_weight=sample_weights)
                else:
                    loss = (utils.get_per_sample_loss(criterion, output, targets) * sample_weights).mean()

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if mixup_fn is None:
            if output.ndim == 3 and targets.ndim == 2:
                ignore_index = getattr(criterion, "ignore_index", -100)
                class_acc = _point_accuracy(output, targets, ignore_index=ignore_index)
            else:
                class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        if loss_ce_value is not None:
            metric_logger.update(loss_ce=loss_ce_value)
        if loss_sc_value is not None:
            metric_logger.update(loss_supcon=loss_sc_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            if loss_ce_value is not None:
                log_writer.update(loss_ce=loss_ce_value, head="loss")
            if loss_sc_value is not None:
                log_writer.update(loss_supcon=loss_sc_value, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if loss_ce_value is not None:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_loss_ce': loss_ce_value}, commit=False)
            if loss_sc_value is not None:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_loss_supcon': loss_sc_value}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False, criterion=None, use_pad_mask: bool = True):
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    maxk = 5
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        lengths = _batch_lengths(batch, device, use_pad_mask=use_pad_mask)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = _model_forward(model, images, lengths=lengths)
                loss = criterion(output, target)
        else:
            output = _model_forward(model, images, lengths=lengths)
            loss = criterion(output, target)

        if output.ndim == 3 and target.ndim == 2:
            ignore_index = getattr(criterion, "ignore_index", -100)
            acc1 = _point_accuracy(output, target, ignore_index=ignore_index) * 100.0
            miou = _mean_iou(output, target, ignore_index=ignore_index) * 100.0
            acc5 = acc1
            valid = (target != ignore_index).sum().item()
            batch_size = max(valid, 1)
        else:
            num_classes = output.size(1)
            maxk = min(5, num_classes)
            topk = (1,) if maxk == 1 else (1, maxk)
            accs = accuracy(output, target, topk=topk)
            acc1 = accs[0]
            acc5 = accs[1] if len(accs) > 1 else accs[0]
            miou = None
            batch_size = images.shape[0]

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        if miou is not None:
            metric_logger.meters['miou'].update(miou.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    topk_label = 5 if maxk >= 5 else maxk
    if hasattr(metric_logger, "miou"):
        print('* Acc@1 {top1.global_avg:.3f} mIoU {miou.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, miou=metric_logger.miou,
                      losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} Acc@{k} {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5,
                      losses=metric_logger.loss, k=topk_label))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
