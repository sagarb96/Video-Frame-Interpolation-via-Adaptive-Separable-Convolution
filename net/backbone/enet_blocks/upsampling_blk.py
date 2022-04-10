# upsampling_blk.py -- Upsampling Bottleneck block for the ENet model

import torch
import torch.nn as nn


class UpsampleBottleNeck(nn.Module):
    """
    UpSample BottleNeck Convolution Block for the ENet Model
    """

    def __init__(self, in_channels, out_channels, conv_1x1_channels, dropout_p=0.1):
        super().__init__()

        self.conv_blk = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=conv_1x1_channels, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(num_features=conv_1x1_channels),
            nn.PReLU(),

            nn.Conv2d(in_channels=conv_1x1_channels, out_channels=conv_1x1_channels, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=conv_1x1_channels),
            nn.PReLU(),

            nn.Conv2d(in_channels=conv_1x1_channels, out_channels=out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU(),

            nn.Dropout(p=dropout_p)
        )

        self.unpool = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        self.unpool_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))

    def forward(self, X, pool_indices):
        """
        X: (batch, 3, height, width)
        """

        y_unpool = self.unpool_conv(X)
        y_unpool = self.unpool(y_unpool, pool_indices)

        y_conv = self.conv_blk(X)

        return y_conv + y_unpool
