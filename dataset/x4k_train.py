# x4k_train.py -- Dataset class for X4K-1000FPS dataset

import pathlib
from torchvision.io import read_image
from torch.utils.data.dataloader import Dataset


class X4K1000FPS(Dataset):
    """
    Dataset loader for X4k-1000FPS dataset, scaled down by a factor of 8
    https://github.com/JihyongOh/XVFI#X4K1000FPS

    Convention:
        -> Each image is of PNG format
        -> There are sub-directories within the train directory that contains the corresponding images (i.e. frames)
        -> The frames are named sequentially in the order they appear in the video
    """

    def __init__(self, directory):
        self.dir_path = pathlib.Path(directory)
        self.folders = sorted(self.dir_path.glob('*/'))
        self.train_X = []                                   # Each training sample contains 2 images
        self.train_Y = []                                   # Each testing sample contains 1 image

        for folder in self.folders:
            imgs_list = sorted(folder.glob('*.png'))

            # Each image (i.e. frame) is named according to the actual frame no.
            # in the video. Take 3 consecutive frames and set the center frame as the target frame
            for i in range(len(imgs_list) - 2):
                train_x = (imgs_list[i], imgs_list[i+2])
                train_y = imgs_list[i+1]

                self.train_X.append(train_x)
                self.train_Y.append(train_y)

    def __len__(self):
        return len(self.train_X)

    def __getitem__(self, index):
        """ Reads the images at specified index """
        prev_frame_path, next_frame_path = self.train_X[index]
        target_frame_path = self.train_Y[index]

        # Normalize the frames
        prev_frame = read_image(str(prev_frame_path)) / 255.0
        next_frame = read_image(str(next_frame_path)) / 255.0
        target_frame = read_image(str(target_frame_path)) / 255.0

        return prev_frame, next_frame, target_frame
