# fe.py --  Module containing the feature-extraction network that will be used as the backbone
#           Paper: https://arxiv.org/abs/1708.01692
#
# Code Reference: https://github.com/sniklaus/sepconv-slomo

import torch
import torch.nn as nn


class DownSampleBlock(nn.Module):
    """
    Simple convolution block with ReLU activation for down-sampling

    Input size:  (batch, in_channels, height, width)
    Output size: (batch, out_channels, height//2, width//2)
    """
    def __init__(self, in_channels, out_channels):

        super().__init__()

        # Padding set to 'same' to ensure that the height/width remains the same as the input
        self.layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        ]

        # Now, down-sample the output by a factor of 2
        # Original paper (from their Github) uses Average Pooling, so using it here
        # NOTE: This is being written separately because we need to return the output BEFORE down-sampling
        #       for the skip-connection with up-sampling blocks
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), count_include_pad=False)

        # Build the network of convolutions
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        output = self.net(x)
        pooled_output = self.pool(output)

        return output, pooled_output


class UpSampleBlock(nn.Module):
    """
    Simple convolution block with ReLU activation for up-sampling

    Input size:  (batch, in_channels, height, width)
    Output size: (batch, out_channels, height*2, width*2)
    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        # Padding set to 'same' to ensure that the height/width remains the same as the input
        self.layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),

            # Now, up-sample the output by a factor of 2
            # Original paper (from their Github) uses BiLinear interpolation, so using it here
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        ]

        # Build the network of convolutions
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        output = self.net(x)
        return output


class FeatureExtractorNet(nn.Module):
    """
    Feature extraction network, based on the architecture provided in the above linked paper
    and their corresponding GitHub code
    """
    def __init__(self, device):

        super().__init__()

        # Series of down-sampling blocks
        self.ds_blk_1 = DownSampleBlock(6, 32)
        self.ds_blk_2 = DownSampleBlock(32, 64)
        self.ds_blk_3 = DownSampleBlock(64, 128)
        self.ds_blk_4 = DownSampleBlock(128, 256)
        self.ds_blk_5 = DownSampleBlock(256, 512)

        # Series of up-sampling blocks
        self.us_blk_5 = UpSampleBlock(512, 512)
        self.us_blk_4 = UpSampleBlock(512, 256)
        self.us_blk_3 = UpSampleBlock(256, 128)
        self.us_blk_2 = UpSampleBlock(128, 64)

    def forward(self, frames_comb):
        """
        Extracts the features from the two input (RGB) frames and returns it

        Input:  (batch, 3, height, width)
        Output: (batch, 64, height//2, width//2)

        NOTE: It is important to note that the height/width dimensions must be padded
              appropriately so that they are an exact multiple of 32. Or else, there will be dimension
              mismatch when doing the skip-connections (due to downsampling/upsampling layers)

              Why 32 ? There are 5 down-sampling blocks, so need input to be atleast multiple of 2^5 = 32
              They can be multiple of 64, 128 etc. as well, but multiple of 32 is the least we expect to avoid
              rounding-off when downsampling
        """

        # Down-sample the frames using the encoder net
        o1, ds_o1 = self.ds_blk_1(frames_comb)
        o2, ds_o2 = self.ds_blk_2(ds_o1)
        o3, ds_o3 = self.ds_blk_3(ds_o2)
        o4, ds_o4 = self.ds_blk_4(ds_o3)
        o5, ds_o5 = self.ds_blk_5(ds_o4)

        # Up-Sample the frames using the decoder net
        us_o5 = self.us_blk_5(ds_o5) + o5
        us_o4 = self.us_blk_4(us_o5) + o4
        us_o3 = self.us_blk_3(us_o4) + o3
        us_o2 = self.us_blk_2(us_o3) + o2

        # We stop here (no use for o1)
        return us_o2
