# init_blk.py -- Initial convolution block for the ENet model

import torch
import torch.nn as nn


class InitialConvBlock(nn.Module):
    """
    Initial Convolution Block for the ENet Model
    """

    def __init__(self, in_channels=6, out_channels=13):
        super().__init__()
        self.conv_blk = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.PReLU(),
            nn.BatchNorm2d(num_features=out_channels)
        )

        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, X):
        """
        X: (batch, 6, height, width)
        """
        y_conv = self.conv_blk(X)
        y_pool = self.max_pool(X)

        # Concatenate on the channel dimension
        # Convolution has 13 channels, and maxpool has 6 channels so the resulting channel dimension has 19 channels
        y_comb = torch.cat([y_conv, y_pool], dim=1)

        return y_comb



