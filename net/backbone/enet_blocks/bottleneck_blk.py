# bottleneck_blk.py -- Bottleneck block for the ENet model

import torch
import torch.nn as nn


class BottleNeck(nn.Module):
    """
    Default BottleNeck Convolution Block for the ENet Model, with optional dilation
    """

    def __init__(self, in_channels, out_channels, conv_1x1_channels, dilation=(1, 1), dropout_p=0.1):
        super().__init__()

        self.conv_blk = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=conv_1x1_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_features=conv_1x1_channels),
            nn.PReLU(),

            nn.Conv2d(in_channels=conv_1x1_channels, out_channels=conv_1x1_channels, kernel_size=(3, 3), padding='same', dilation=dilation),
            nn.BatchNorm2d(num_features=conv_1x1_channels),
            nn.PReLU(),

            nn.Conv2d(in_channels=conv_1x1_channels, out_channels=out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU(),

            nn.Dropout(p=dropout_p)
        )

    def forward(self, X):
        """
        X: (batch, 3, height, width)
        """
        y_conv = self.conv_blk(X)
        return y_conv



