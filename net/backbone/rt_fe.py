# rt_fe.py --  Module containing the (real-time) feature-extraction network that will be used as the backbone
#              Paper: E-Net, https://arxiv.org/abs/1606.02147
#

import torch
import torch.nn as nn

# Custom module imports
from net.backbone.enet_blocks.init_blk import InitialConvBlock
from net.backbone.enet_blocks.asymmetric_blk import AsymmetricBottleNeck
from net.backbone.enet_blocks.downsampling_blk import DownsampleBottleNeck
from net.backbone.enet_blocks.upsampling_blk import UpsampleBottleNeck
from net.backbone.enet_blocks.bottleneck_blk import BottleNeck


def build_bottleneck_2(channels):
    """
    A function to build the bottleneck 2.X blocks
    Because there is a repeat, it is better to have it in the same place
    """
    conv_blk = nn.Sequential(
        BottleNeck(in_channels=channels, out_channels=channels, conv_1x1_channels=(channels // 2)),
        BottleNeck(in_channels=channels, out_channels=channels, conv_1x1_channels=(channels // 2), dilation=(2, 2)),
        AsymmetricBottleNeck(in_channels=channels, out_channels=channels, conv_1x1_channels=(channels // 2)),
        BottleNeck(in_channels=channels, out_channels=channels, conv_1x1_channels=(channels // 2), dilation=(4, 4)),
        BottleNeck(in_channels=channels, out_channels=channels, conv_1x1_channels=(channels // 2)),
        BottleNeck(in_channels=channels, out_channels=channels, conv_1x1_channels=(channels // 2), dilation=(8, 8)),
        AsymmetricBottleNeck(in_channels=channels, out_channels=channels, conv_1x1_channels=(channels // 2)),
        BottleNeck(in_channels=channels, out_channels=channels, conv_1x1_channels=(channels // 2), dilation=(16, 16)),
    )

    return conv_blk


class ENet_FeatureExtractor(nn.Module):
    """
    Class for the ENet model
    Paper: https://arxiv.org/abs/1606.02147
    """

    def __init__(self, device):
        super().__init__()

        self.device = device
        self.init_conv_blk = InitialConvBlock(in_channels=6)
        self.downsample_1 = DownsampleBottleNeck(in_channels=19, out_channels=64, dropout_p=0.01, device=device)

        # 4 bottleneck 1.X blocks
        self.bottleneck_1 = nn.Sequential(
            *[BottleNeck(in_channels=64, out_channels=64, conv_1x1_channels=32, dropout_p=0.01) for _ in range(4)]
        )

        self.downsample_2 = DownsampleBottleNeck(in_channels=64, out_channels=128, device=device)  # Using default dropout (0.1)
        self.bottleneck_2 = nn.Sequential(
            *[build_bottleneck_2(128) for _ in range(2)]
        )

        # Naming the upsample in reverse to know which downsample operation it undoes
        self.upsample_2 = UpsampleBottleNeck(in_channels=128, out_channels=64, conv_1x1_channels=64)
        self.bottleneck_4 = nn.Sequential(
            *[BottleNeck(in_channels=64, out_channels=64, conv_1x1_channels=32) for _ in range(2)]
        )

        self.upsample_1 = UpsampleBottleNeck(in_channels=64, out_channels=19, conv_1x1_channels=32)
        self.bottleneck_5 = BottleNeck(in_channels=19, out_channels=64, conv_1x1_channels=32)

        # Final layer -- Used a transposed convolution to upsample the image to the full resolution
        # NOTE: There is no final layer for this implementation. Also instead of 16 channels as output in
        #       self.bottleneck_5, 64 channels are used here (to be compliant with the Interpolation network)

    def forward(self, X):
        """
        X: (batch, channels, height, width)
        """
        y_init = self.init_conv_blk(X)

        y_ds_1, pool_idx_1 = self.downsample_1(y_init)
        y_bneck_1 = self.bottleneck_1(y_ds_1)

        y_ds_2, pool_idx_2 = self.downsample_2(y_bneck_1)
        y_bneck_2 = self.bottleneck_2(y_ds_2)

        y_us_2 = self.upsample_2(y_bneck_2, pool_idx_2)
        y_bneck_4 = self.bottleneck_4(y_us_2)

        y_us_1 = self.upsample_1(y_bneck_4, pool_idx_1)
        y_bneck_5 = self.bottleneck_5(y_us_1)

        return y_bneck_5
