"""Masked curve reconstruction with an explicitly separated encoder and decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convnext1d import Block1d, LayerNorm1d, UPerHead1d, _init_conv1d_linear


def _downsample_configs(output_stride):
    if output_stride == 32:
        return [(4, 4, 0), (2, 2, 0), (2, 2, 0), (2, 2, 0)]
    if output_stride == 16:
        return [(3, 2, 1), (2, 2, 0), (2, 2, 0), (2, 2, 0)]
    if output_stride == 8:
        return [(3, 2, 1), (2, 2, 0), (2, 2, 0), (3, 1, 1)]
    raise ValueError("encoder_output_stride must be one of 8, 16, or 32")


class ConvNeXt1dEncoder(nn.Module):
    """Four-stage ConvNeXt encoder returning every stage for dense reconstruction."""

    def __init__(
        self, in_chans=3, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
        drop_path_rate=0.0, layer_scale_init_value=1e-6,
        encoder_output_stride=8,
    ):
        super().__init__()
        self.dims = tuple(dims)
        self.encoder_output_stride = int(encoder_output_stride)
        configs = _downsample_configs(self.encoder_output_stride)

        self.downsample_layers = nn.ModuleList()
        for stage_idx, (kernel, stride, padding) in enumerate(configs):
            in_dim = in_chans if stage_idx == 0 else dims[stage_idx - 1]
            layers = []
            if stage_idx > 0:
                layers.append(LayerNorm1d(in_dim, eps=1e-6, data_format="channels_first"))
            layers.extend([
                nn.Conv1d(in_dim, dims[stage_idx], kernel, stride, padding),
                LayerNorm1d(dims[stage_idx], eps=1e-6, data_format="channels_first"),
            ] if stage_idx == 0 else [
                nn.Conv1d(in_dim, dims[stage_idx], kernel, stride, padding),
            ])
            self.downsample_layers.append(nn.Sequential(*layers))

        rates = [value.item() for value in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        offset = 0
        for stage_idx, depth in enumerate(depths):
            self.stages.append(nn.Sequential(*[
                Block1d(
                    dims[stage_idx],
                    drop_path=rates[offset + block_idx],
                    layer_scale_init_value=layer_scale_init_value,
                )
                for block_idx in range(depth)
            ]))
            offset += depth
        self.stage_norms = nn.ModuleList([
            LayerNorm1d(dim, eps=1e-6, data_format="channels_first") for dim in dims
        ])
        self.apply(_init_conv1d_linear)

    def forward(self, x):
        features = []
        for downsample, stage, norm in zip(
            self.downsample_layers, self.stages, self.stage_norms
        ):
            x = stage(downsample(x))
            features.append(norm(x))
        return features


class ReconstructionDecoder1d(nn.Module):
    """UPer-style multi-stage decoder that reconstructs the input curves."""

    def __init__(self, in_channels, out_channels, channels=256):
        super().__init__()
        self.decode_head = UPerHead1d(
            in_channels=in_channels,
            num_classes=out_channels,
            channels=channels,
        )

    def forward(self, features, output_size):
        prediction = self.decode_head(features)
        if prediction.shape[-1] != output_size:
            prediction = F.interpolate(
                prediction, size=output_size, mode="linear", align_corners=False
            )
        return prediction


class MaskedWellLogAutoencoder(nn.Module):
    def __init__(
        self, in_chans=3, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
        decoder_channels=256, drop_path_rate=0.0, encoder_output_stride=8,
    ):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, in_chans, 1))
        nn.init.normal_(self.mask_token, std=0.02)
        self.encoder = ConvNeXt1dEncoder(
            in_chans=in_chans,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            encoder_output_stride=encoder_output_stride,
        )
        self.decoder = ReconstructionDecoder1d(
            in_channels=dims,
            out_channels=in_chans,
            channels=decoder_channels,
        )

    def forward(self, x, visible_mask):
        masked_input = torch.where(visible_mask, x, self.mask_token.expand_as(x))
        features = self.encoder(masked_input)
        return self.decoder(features, output_size=x.shape[-1])

    @staticmethod
    def masked_loss(prediction, target, visible_mask):
        masked = ~visible_mask
        if not masked.any():
            return prediction.sum() * 0.0
        return F.smooth_l1_loss(prediction[masked], target[masked])

    def encoder_state_dict(self):
        """Return keys accepted directly by ConvNeXt1dUPerSeg."""
        return {key: value.detach().cpu() for key, value in self.encoder.state_dict().items()}
