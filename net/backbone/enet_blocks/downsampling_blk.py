# downsampling_blk.py -- Downsampling block for the ENet model

import torch
import torch.nn as nn


class DownsampleBottleNeck(nn.Module):
    """
    Downsampling BottleNeck Convolution Block for the ENet Model
    """

    def __init__(self, in_channels, out_channels, device, dropout_p=0.1):
        super().__init__()
        self.device = device

        # Number of channels required to pad the max-pool operation
        self.n_pad_channels_req = out_channels - in_channels

        self.conv_blk = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(num_features=in_channels),
            nn.PReLU(),

            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=in_channels),
            nn.PReLU(),

            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU(),

            nn.Dropout(p=dropout_p)
        )

        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)

    def forward(self, X):
        """
        X: (batch, 3, height, width)
        """
        y_conv = self.conv_blk(X)
        y_pool, pool_indices = self.max_pool(X)

        # Now need to pad the channel dimension of max-pool with zeros so that
        # it matches with the output of the convolution block
        ip_shape = list(y_conv.shape)
        ip_shape[1] = self.n_pad_channels_req   # Change the channel dimension shape

        zero_pads = torch.zeros(ip_shape).to(self.device)

        y_pool_padded = torch.cat([y_pool, zero_pads], dim=1)

        # Need to do a sum of the (padded) pool layer and the convolution layer
        return y_pool_padded + y_conv, pool_indices



