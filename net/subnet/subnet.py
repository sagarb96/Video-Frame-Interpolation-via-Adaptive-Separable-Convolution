# subnet.py -- Module containing the branching subnet of the interpolation network
# NOTE: The implementation is same as the paper

import torch.nn as nn


class SubNet(nn.Module):
    """
    The branching network after the backbone feature-extracting network
    NOTE: Input channels must be 64
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),

            # Final upsample to match the input frame shape
            # The backbone network is responsible for upscaling upto half the original frame size
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same')
        ]

        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)
