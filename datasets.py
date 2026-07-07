# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os

from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

from welllog.dataset import WellLogSlidingWindowDataset, WellLogWholeWellDataset
from welllog.collate import collate_whole_well


def build_welllog_collate_fn(args):
    well_input_mode = getattr(args, "well_input_mode", "sliding_window")
    if well_input_mode != "whole_well":
        return None
    ignore_index = getattr(args, "ignore_index", -100)

    def _collate(batch):
        return collate_whole_well(batch, ignore_index=ignore_index)

    return _collate


def build_dataset(is_train, args):
    if args.data_set == "WELLLOG_XLSX":
        # Training set builds the label_map; val set reuses it to keep indices consistent.
        label_map = None if is_train else getattr(args, '_welllog_label_map', None)
        well_input_mode = getattr(args, "well_input_mode", "sliding_window")

        if well_input_mode == "whole_well":
            dataset = WellLogWholeWellDataset(args=args, is_train=is_train, label_map=label_map)
        else:
            task_mode = getattr(args, "task_mode", "classification")
            if not is_train and task_mode != "segmentation":
                # Val set uses stride=1 so evaluate() covers every depth point (point-wise accuracy).
                import copy
                val_args = copy.copy(args)
                val_args.window_stride = 1
                dataset = WellLogSlidingWindowDataset(args=val_args, is_train=False, label_map=label_map)
            else:
                dataset = WellLogSlidingWindowDataset(args=args, is_train=is_train, label_map=label_map)

        if is_train:
            # Cache so the val dataset can reuse the same mapping.
            args._welllog_label_map = dataset.label_map
        nb_classes = len(dataset.label_map)
        split = "Train" if is_train else "Val"
        inv = {v: k for k, v in dataset.label_map.items()}
        readable_counts = {inv.get(k, k): v for k, v in sorted(dataset.class_counts.items())}
        print(f"[{split}] WellLog dataset samples: {len(dataset)}")
        print(f"[{split}] task mode: {getattr(dataset, 'task_mode', 'classification')}")
        print(f"[{split}] well input mode: {getattr(dataset, 'well_input_mode', 'sliding_window')}")
        print(f"[{split}] conv mode: {'1D (C,L)' if getattr(args, 'use_1d_conv', False) else '2D (C,H,W)'}")
        print(f"[{split}] input_mode: {getattr(args, 'input_mode', 'raw')}")
        if getattr(args, "window_require_pure", False):
            print(f"[{split}] window_require_pure: True (discard mixed-class windows)")
        filter_stats = getattr(args, "_window_filter_stats", None)
        if filter_stats:
            print(
                f"[{split}] windows total={filter_stats.get('total_windows', 0)} "
                f"kept={filter_stats.get('kept_windows', 0)} "
                f"discarded_mixed={filter_stats.get('discarded_mixed_or_invalid', 0)} "
                f"discarded_purity={filter_stats.get('discarded_purity', 0)} "
                # f"gan_synthetic={filter_stats.get('gan_synthetic', 0)}"
            )
        print(f"[{split}] Number of classes: {nb_classes}")
        print(f"[{split}] label_map: {dataset.label_map}")
        print(f"[{split}] Class counts (original facies label): {readable_counts}")
        # if is_train:
        #     resmote_stats = getattr(args, "_resmote_stats", None)
        #     if resmote_stats and resmote_stats.get("enabled"):
        #         inv = {v: k for k, v in dataset.label_map.items()}
        #         before = {
        #             inv.get(int(k), k): v for k, v in sorted(resmote_stats.get("before", {}).items())
        #         }
        #         after = {
        #             inv.get(int(k), k): v for k, v in sorted(resmote_stats.get("after", {}).items())
        #         }
        #         targets = {
        #             inv.get(int(k), k): v for k, v in sorted(resmote_stats.get("targets", {}).items())
        #         }
        #         print(f"[RESMOTE] target_classes: {resmote_stats.get('target_classes')}")
        #         print(f"[RESMOTE] before (real points): {before}")
        #         print(f"[RESMOTE] targets: {targets}")
        #         print(f"[RESMOTE] after  (all points): {after}")
        #         print(f"[RESMOTE] inserted synthetic points: {resmote_stats.get('inserted', 0)}")
        #         inserted_by_class = resmote_stats.get("inserted_by_class") or {}
        #         if inserted_by_class:
        #             readable_inserted = {
        #                 inv.get(int(k), k): v for k, v in sorted(inserted_by_class.items())
        #             }
        #             print(f"[RESMOTE] inserted by class: {readable_inserted}")
        #         for warning in resmote_stats.get("warnings") or []:
        #             print(f"[RESMOTE] Warning: {warning}")
        #     gan_stats = getattr(args, "_gan_stats", None)
        #     if gan_stats and gan_stats.get("enabled"):
        #         inv = {v: k for k, v in dataset.label_map.items()}
        #         pure_by_class = gan_stats.get("pure_windows_by_class") or {}
        #         readable_pure = {
        #             inv.get(int(k), k): v for k, v in sorted(pure_by_class.items())
        #         }
        #         gen_by_class = gan_stats.get("generated_by_class") or {}
        #         readable_gen = {
        #             inv.get(int(k), k): v for k, v in sorted(gen_by_class.items())
        #         }
        #         print(f"[GAN] target_classes: {gan_stats.get('target_classes')}")
        #         print(f"[GAN] pure windows by class: {readable_pure}")
        #         print(f"[GAN] generated by class: {readable_gen}")
        #         print(
        #             f"[GAN] ckpt_dir={gan_stats.get('ckpt_dir')} "
        #             f"loaded={gan_stats.get('loaded', 0)} trained={gan_stats.get('trained', 0)}"
        #         )
        #         for warning in gan_stats.get("warnings") or []:
        #             print(f"[GAN] Warning: {warning}")
        return dataset, nb_classes

    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
