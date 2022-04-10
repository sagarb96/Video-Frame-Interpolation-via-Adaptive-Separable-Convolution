# x4k_test.py -- Dataset class for X4K-1000FPS dataset

import pathlib
from torchvision.io import read_image
from torch.utils.data.dataloader import Dataset


class X4K1000FPS(Dataset):
    """
    Dataset for X4k-1000FPS dataset, scaled down by a factor of 8
    https://github.com/JihyongOh/XVFI#X4K1000FPS

    Note that each video is 1-second long

    Convention:
        -> Each image is of PNG format
        -> There shouldn't be any sub-folders (as they are directly ignored)
        -> Only the images (i.e. frames) of a specific video are present in the directory
        -> The frames are named sequentially in the order they appear in the video

    """

    def __init__(self, directory):
        self.dir_path = pathlib.Path(directory)
        self.test_X = []                                   # Each training sample contains 2 images
        self.test_Y = []                                   # Each testing sample contains 1 image

        imgs_list = sorted(self.dir_path.glob('*.png'))
        self.n_frames = len(imgs_list)

        # Each image (i.e. frame) is named according to the actual frame no.
        # in the video. Take 2 consecutive frames and produce the "never seen before" middle frame
        # This part is different from the training set
        # Here we are actually trying to find the "middle" frame
        for i in range(len(imgs_list) - 1):
            test_x = (imgs_list[i], imgs_list[i+1])
            self.test_X.append(test_x)

    def __len__(self):
        return len(self.test_X)

    def __getitem__(self, index):
        """ Reads the images at specified index """
        prev_frame_path, next_frame_path = self.test_X[index]

        # Normalize the frames
        prev_frame = read_image(str(prev_frame_path)) / 255.0
        next_frame = read_image(str(next_frame_path)) / 255.0

        return prev_frame, next_frame

    def get_video_id(self):
        """ Returns the name of the video folder """
        return self.dir_path.name

    def get_video_fps(self):
        """
        For this dataset, we produce one less than twice the number of frames as interpolated frames
        For example, if our frame list contains

                                    frames = [f0, f1, f2, ..., fn]

        output video would have frames = [f0, f'0, f1, f'1, f2, ..., f'(n-1), fn]
        where f'(i) is the interpolated frame between f(i) and f(i+1)
        """
        return self.n_frames, 2*self.n_frames - 1
