# Copyright (c) Meta Platforms, Inc. and affiliates.
# 1D ConvNeXt for well-log sequence classification and segmentation.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

from welllog.padding_mask import (
    apply_feature_mask,
    downsample_valid_mask,
    lengths_to_mask,
    masked_adaptive_avg_pool1d,
    resize_mask,
)


def _init_conv1d_linear(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class LayerNorm1d(nn.Module):
    """LayerNorm for (N, C, L) channels_first or (N, L, C) channels_last."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x


class Block1d(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.spatial_mix = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm1d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0 else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.spatial_mix(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)
        x = input + self.drop_path(x)
        return x


class ConvNeXt1d(nn.Module):
    """1D ConvNeXt for sequence inputs of shape (N, C, L)."""

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            ignored = ", ".join(sorted(kwargs.keys()))
            print(f"ConvNeXt1d: ignoring unsupported kwargs: {ignored}")

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=4, stride=4),   # stem层下采样先缩小4倍
            LayerNorm1d(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):  # 共4个block，每两个block之间一个下采样层
            downsample_layer = nn.Sequential(
                LayerNorm1d(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):   # 4个convnext block
            stage = nn.Sequential(
                *[
                    Block1d(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])   # block数
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        _init_conv1d_linear(m)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)   # 加上stem一共4层下采样
            x = self.stages[i](x)
        return self.norm(x.mean(-1))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class SegmentationHead1d(nn.Module):
    """Lightweight 1D segmentation head on full-resolution feature maps."""

    def __init__(self, in_dim, num_classes, hidden_dim=128, head_init_scale=1.0):
        super().__init__()
        mid_dim = max(hidden_dim // 2, num_classes)
        # 原文是：Conv3×3 -> BatchNorm -> Relu ->Conv1×1
        # 现在是：Conv1×3 -> GELU -> Conv1×3 -> GELU -> Conv1×1
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, mid_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(mid_dim, num_classes, kernel_size=1),
        )
        self.head_init_scale = float(head_init_scale)

    def forward(self, x):
        return self.net(x)


class PPM1d(nn.Module):
    """1D Pyramid Pooling Module (PPM) used by UPerHead."""

    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6)):
        # in_channels = out_channels = 256
        super().__init__()
        self.pool_scales = pool_scales
        branch_channels = out_channels // len(pool_scales)   #256//4=64, 原文是512
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool1d(scale),
                # 原文是4096->512，现在改成了256->64
                nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
                LayerNorm1d(branch_channels, eps=1e-6, data_format="channels_first"),
                nn.GELU(),
            )
            for scale in pool_scales
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels + branch_channels * len(pool_scales), out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm1d(out_channels, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
        )

    def forward(self, x, mask=None):
        size = x.shape[-1]
        ppm_outs = [x]
        for branch, scale in zip(self.branches, self.pool_scales):
            if mask is None:
                out = branch(x)
            else:
                out = masked_adaptive_avg_pool1d(x, mask, scale)
                for layer in list(branch.children())[1:]:
                    out = layer(out)
            out = F.interpolate(out, size=size, mode="linear", align_corners=False)
            if mask is not None:
                out = apply_feature_mask(out, mask)
            ppm_outs.append(out)
        fused = self.bottleneck(torch.cat(ppm_outs, dim=1))
        if mask is not None:
            fused = apply_feature_mask(fused, mask)
        return fused


class UPerHead1d(nn.Module):
    """1D UPerNet decode head: PPM on deepest features + FPN multi-scale fusion."""

    def __init__(
        self,
        in_channels,
        num_classes,
        channels=256,
        pool_scales=(1, 2, 3, 6),
        head_init_scale=1.0,
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.channels = int(channels)   # 256
        self.num_stages = len(self.in_channels)

        self.lateral_convs = nn.ModuleList([    # 通道对齐
            nn.Sequential(
                nn.Conv1d(in_ch, self.channels, kernel_size=1, bias=False),  # 将不同stage尺度的特征图进行通道对齐，与decoder的特征图拼接，统一为256通道
                LayerNorm1d(self.channels, eps=1e-6, data_format="channels_first"),
                nn.GELU(),
            )
            for in_ch in self.in_channels
        ])
        self.ppm = PPM1d(self.channels, self.channels, pool_scales=pool_scales)    # 金字塔池化模块，将不同尺度的特征图进行融合
        self.fpn_convs = nn.ModuleList([    # feature pyramid network 右半部分作为deocder的主要部分
            nn.Sequential(
                nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1, bias=False),
                LayerNorm1d(self.channels, eps=1e-6, data_format="channels_first"),
                nn.GELU(),
            )
            for _ in self.in_channels
        ])
        self.fpn_bottleneck = nn.Sequential(    # 瓶颈层 fusion
            nn.Conv1d(self.channels * self.num_stages, self.channels, kernel_size=3, padding=1, bias=False),
            LayerNorm1d(self.channels, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
        )
        self.cls_seg = nn.Conv1d(self.channels, num_classes, kernel_size=1)    
        self.head_init_scale = float(head_init_scale)

    def _resize_add(self, higher_res_feat, lower_res_feat):
        if higher_res_feat.shape[-1] != lower_res_feat.shape[-1]:
            lower_res_feat = F.interpolate(
                lower_res_feat,
                size=higher_res_feat.shape[-1],
                mode="linear",
                align_corners=False,
            )
        return higher_res_feat + lower_res_feat

    # 通道对齐 -> ppm金字塔池化 -> feature pyramid network -> 瓶颈层 -> 分割头
    def forward(self, inputs, masks=None):
        if len(inputs) != self.num_stages:
            raise ValueError(
                f"UPerHead1d expects {self.num_stages} feature maps, got {len(inputs)}."
            )
        if masks is not None and len(masks) != self.num_stages:
            raise ValueError(
                f"UPerHead1d expects {self.num_stages} masks, got {len(masks)}."
            )

        laterals = []
        for conv, feat, stage_mask in zip(
            self.lateral_convs,
            inputs,
            masks if masks is not None else [None] * self.num_stages,
        ):
            lat = conv(feat)
            if stage_mask is not None:
                lat = apply_feature_mask(lat, stage_mask)
            laterals.append(lat)

        deepest_mask = masks[-1] if masks is not None else None
        laterals[-1] = self.ppm(laterals[-1], mask=deepest_mask)

        for i in range(self.num_stages - 2, -1, -1):   # 从倒数第2层，stage3开始上采样并拼接
            laterals[i] = self._resize_add(laterals[i], laterals[i + 1])
            if masks is not None:
                laterals[i] = apply_feature_mask(laterals[i], masks[i])

        fpn_outs = []
        for conv, lat, stage_mask in zip(
            self.fpn_convs,
            laterals,
            masks if masks is not None else [None] * self.num_stages,
        ):
            out = conv(lat)
            if stage_mask is not None:
                out = apply_feature_mask(out, stage_mask)
            fpn_outs.append(out)

        target_size = fpn_outs[0].shape[-1]
        aligned_outs = []
        for out, stage_mask in zip(
            fpn_outs,
            masks if masks is not None else [None] * self.num_stages,
        ):
            if out.shape[-1] != target_size:
                out = F.interpolate(out, size=target_size, mode="linear", align_corners=False)
            if stage_mask is not None:
                stage_mask = resize_mask(stage_mask, target_size)
                out = apply_feature_mask(out, stage_mask)
            aligned_outs.append(out)

        fused = self.fpn_bottleneck(torch.cat(aligned_outs, dim=1))
        if masks is not None:
            fused = apply_feature_mask(fused, masks[0])
        return self.cls_seg(fused)


class ConvNeXt1dSeg(nn.Module):
    """1D ConvNeXt encoder with full-resolution sequence segmentation head."""

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        seg_head_dim=128,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            ignored = ", ".join(sorted(kwargs.keys()))
            print(f"ConvNeXt1dSeg: ignoring unsupported kwargs: {ignored}")

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm1d(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm1d(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block1d(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = LayerNorm1d(dims[-1], eps=1e-6, data_format="channels_first")
        self.seg_head = SegmentationHead1d(
            dims[-1],   
            num_classes,
            hidden_dim=seg_head_dim,
            head_init_scale=head_init_scale,
        )

        self.apply(self._init_weights)
        self.seg_head.net[-1].weight.data.mul_(self.seg_head.head_init_scale)
        self.seg_head.net[-1].bias.data.mul_(self.seg_head.head_init_scale)

    def _init_weights(self, m):
        _init_conv1d_linear(m)

    def forward_features(self, x, lengths=None):
        mask = None
        if lengths is not None:
            mask = lengths_to_mask(lengths, x.shape[-1])
            x = apply_feature_mask(x, mask)
        for i in range(4):
            x = self.downsample_layers[i](x)
            if mask is not None:
                kernel_size, stride = [(4, 4), (2, 2), (2, 2), (2, 2)][i]
                mask = downsample_valid_mask(mask, kernel_size=kernel_size, stride=stride)
                x = apply_feature_mask(x, mask)
            x = self.stages[i](x)
            if mask is not None:
                x = apply_feature_mask(x, mask)
        return self.norm(x)

    def forward(self, x, lengths=None):
        input_len = x.shape[-1]
        x = self.forward_features(x, lengths=lengths)
        x = F.interpolate(x, size=input_len, mode="linear", align_corners=False)
        x = self.seg_head(x)
        if lengths is not None:
            x = apply_feature_mask(x, lengths_to_mask(lengths, input_len))
        return x


class ConvNeXt1dUPerSeg(nn.Module):
    """1D ConvNeXt encoder + UPerNet-style multi-scale decode head."""

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        decoder_channels=256,
        pool_scales=(1, 2, 3, 6),
        encoder_output_stride=32,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            ignored = ", ".join(sorted(kwargs.keys()))
            print(f"ConvNeXt1dUPerSeg: ignoring unsupported kwargs: {ignored}")

        if encoder_output_stride not in (8, 16, 32):
            raise ValueError("encoder_output_stride must be one of 8, 16, or 32")
        self.encoder_output_stride = int(encoder_output_stride)
        if self.encoder_output_stride == 32:
            downsample_cfgs = [(4, 4, 0), (2, 2, 0), (2, 2, 0), (2, 2, 0)]
        elif self.encoder_output_stride == 16:
            downsample_cfgs = [(3, 2, 1), (2, 2, 0), (2, 2, 0), (2, 2, 0)]
        else:
            downsample_cfgs = [(3, 2, 1), (2, 2, 0), (2, 2, 0), (3, 1, 1)]
        self._downsample_cfgs = downsample_cfgs

        self.downsample_layers = nn.ModuleList()
        stem_kernel, stem_stride, stem_padding = downsample_cfgs[0]
        stem = nn.Sequential(
            nn.Conv1d(
                in_chans, dims[0], kernel_size=stem_kernel,
                stride=stem_stride, padding=stem_padding,
            ),
            LayerNorm1d(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            kernel_size, stride, padding = downsample_cfgs[i + 1]
            downsample_layer = nn.Sequential(
                LayerNorm1d(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(
                    dims[i], dims[i + 1], kernel_size=kernel_size,
                    stride=stride, padding=padding,
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block1d(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.stage_norms = nn.ModuleList([
            LayerNorm1d(dim, eps=1e-6, data_format="channels_first") for dim in dims
        ])
        self.decode_head = UPerHead1d(
            in_channels=dims,
            num_classes=num_classes,
            channels=decoder_channels,
            pool_scales=pool_scales,
            head_init_scale=head_init_scale,
        )

        self.apply(self._init_weights)
        self.decode_head.cls_seg.weight.data.mul_(self.decode_head.head_init_scale)
        self.decode_head.cls_seg.bias.data.mul_(self.decode_head.head_init_scale)

    def _init_weights(self, m):
        _init_conv1d_linear(m)

    def _encoder_downsample_cfgs(self):
        return self._downsample_cfgs

    def forward_encoder_multi(self, x, lengths=None):
        """Return normalized feature maps from all four encoder stages."""
        feats = []
        mask = None
        stage_masks = []
        if lengths is not None:
            mask = lengths_to_mask(lengths, x.shape[-1])
            x = apply_feature_mask(x, mask)

        for i in range(4):
            x = self.downsample_layers[i](x)
            if mask is not None:
                kernel_size, stride, padding = self._encoder_downsample_cfgs()[i]
                mask = downsample_valid_mask(
                    mask, kernel_size=kernel_size, stride=stride, padding=padding
                )
                x = apply_feature_mask(x, mask)
            x = self.stages[i](x)
            if mask is not None:
                x = apply_feature_mask(x, mask)
            stage_masks.append(mask)
            feats.append(self.stage_norms[i](x))
        return feats if lengths is None else (feats, stage_masks)

    def forward_features(self, x, lengths=None):
        """Deepest stage feature map for API compatibility."""
        out = self.forward_encoder_multi(x, lengths=lengths)
        if lengths is None:
            return out[-1]
        feats, _ = out
        return feats[-1]

    def forward(self, x, lengths=None):
        input_len = x.shape[-1]
        if lengths is None:
            feats = self.forward_encoder_multi(x)
            stage_masks = None
        else:
            feats, stage_masks = self.forward_encoder_multi(x, lengths=lengths)
        logits = self.decode_head(feats, masks=stage_masks)
        if logits.shape[-1] != input_len:
            logits = F.interpolate(logits, size=input_len, mode="linear", align_corners=False)
        if lengths is not None:
            logits = apply_feature_mask(logits, lengths_to_mask(lengths, input_len))
        return logits


@register_model
def convnext1d_tiny(pretrained=False, **kwargs):
    model = ConvNeXt1d(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        raise NotImplementedError("ConvNeXt1d has no official pretrained weights.")
    return model


@register_model
def convnext1d_small(pretrained=False, **kwargs):
    model = ConvNeXt1d(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        raise NotImplementedError("ConvNeXt1d has no official pretrained weights.")
    return model


@register_model
def convnext1d_base(pretrained=False, **kwargs):
    model = ConvNeXt1d(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        raise NotImplementedError("ConvNeXt1d has no official pretrained weights.")
    return model


@register_model
def convnext1d_large(pretrained=False, **kwargs):
    model = ConvNeXt1d(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        raise NotImplementedError("ConvNeXt1d has no official pretrained weights.")
    return model


@register_model
def convnext1d_tiny_seg(pretrained=False, **kwargs):
    model = ConvNeXt1dSeg(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        raise NotImplementedError("ConvNeXt1dSeg has no official pretrained weights.")
    return model


@register_model
def convnext1d_small_seg(pretrained=False, **kwargs):
    model = ConvNeXt1dSeg(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        raise NotImplementedError("ConvNeXt1dSeg has no official pretrained weights.")
    return model


@register_model
def convnext1d_base_seg(pretrained=False, **kwargs):
    model = ConvNeXt1dSeg(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        raise NotImplementedError("ConvNeXt1dSeg has no official pretrained weights.")
    return model


@register_model
def convnext1d_large_seg(pretrained=False, **kwargs):
    model = ConvNeXt1dSeg(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        raise NotImplementedError("ConvNeXt1dSeg has no official pretrained weights.")
    return model


@register_model
def convnext1d_tiny_uper_seg(pretrained=False, **kwargs):
    model = ConvNeXt1dUPerSeg(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        raise NotImplementedError("ConvNeXt1dUPerSeg has no official pretrained weights.")
    return model


@register_model
def convnext1d_small_uper_seg(pretrained=False, **kwargs):
    model = ConvNeXt1dUPerSeg(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        raise NotImplementedError("ConvNeXt1dUPerSeg has no official pretrained weights.")
    return model


@register_model
def convnext1d_base_uper_seg(pretrained=False, **kwargs):
    model = ConvNeXt1dUPerSeg(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        raise NotImplementedError("ConvNeXt1dUPerSeg has no official pretrained weights.")
    return model


@register_model
def convnext1d_large_uper_seg(pretrained=False, **kwargs):
    model = ConvNeXt1dUPerSeg(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        raise NotImplementedError("ConvNeXt1dUPerSeg has no official pretrained weights.")
    return model
