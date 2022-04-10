# asymmetric_blk.py -- Asymmetric Bottleneck block for the ENet model

import torch
import torch.nn as nn


class AsymmetricBottleNeck(nn.Module):
    """
    Asymmetric BottleNeck Convolution Block for the ENet Model that uses (5, 1) and (1, 5) kernels
    """

    def __init__(self, in_channels, out_channels, conv_1x1_channels, dropout_p=0.1):
        super().__init__()

        self.conv_blk = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=conv_1x1_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_features=conv_1x1_channels),
            nn.PReLU(),

            nn.Conv2d(in_channels=conv_1x1_channels, out_channels=conv_1x1_channels, kernel_size=(5, 1), padding='same'),
            nn.BatchNorm2d(num_features=conv_1x1_channels),
            nn.PReLU(),

            nn.Conv2d(in_channels=conv_1x1_channels, out_channels=conv_1x1_channels, kernel_size=(1, 5), padding='same'),
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



