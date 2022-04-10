# main.py -- Module to build the networks and start the training/testing
#
# Usage:
#   python3 main.py --realTime {0 | 1} --train {0 | 1} --pretrained /path/to/pretrained/model --testPath /path/to/test/directory

import argparse
import pathlib
import sys
import torch.utils.data

# Custom module imports
import dataset.x4k_train as train_dataset
import dataset.x4k_test as test_dataset
import net.net as net
import start


# --------------------------------- CONSTANTS ---------------------------------
CHKPT_DIR_PATH = './checkpoints'  # The directory for storing model checkpoints
OUTPUT_DIR_PATH = './output'      # The directory for storing output
DATA_DIR_PATH = './dataset'       # Root directory of the dataset
TRAIN_DIR = 'train'               # Training set directory name

BATCH_SIZE = 16                   # No. of samples per batch
N_EPOCHS = 20                     # No. of epochs to train
CHKPT_EPOCHS = 2                  # Epochs after which model will be saved
# -----------------------------------------------------------------------------


def parse_args(args):
    """
    Builds the argument-parser and parses the arguments

    Returns:
        (realTime?, train?)
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--realTime', type=int, required=True, choices=(0, 1), help='Use neural-net made for real-time ?')
    arg_parser.add_argument('--train', type=int, required=True, choices=(0, 1), help='Train mode / Testing mode')
    arg_parser.add_argument('--pretrained', type=str, help='Path to the pretrained model', default=None)
    arg_parser.add_argument('--testPath', type=str, help='Path to test directory', default=None)

    # Parse the arguments and return the appropriate values
    got_args = arg_parser.parse_args(args)

    return got_args.realTime, got_args.train, got_args.pretrained, got_args.testPath


if __name__ == '__main__':
    real_time_mode, train_mode, pretrained_path, test_path = parse_args(sys.argv[1:])

    # Some sanity checks
    assert torch.cuda.is_available(), "[ERROR]: Need CUDA supported GPU or else it will not work :("
    assert train_mode or (pretrained_path is not None and test_path is not None), "[ERROR]: Testing mode requires pretrained model and test directory"

    device = torch.device('cuda:0')

    # Set the dataset path accordingly
    chkpt_dir_path = pathlib.Path(CHKPT_DIR_PATH)
    output_dir_path = pathlib.Path(OUTPUT_DIR_PATH)
    data_dir_path = pathlib.Path(DATA_DIR_PATH)

    # Build the network
    # TODO: Load the network from pre-defined weights if option specified
    network = net.InterpolationNet(real_time_mode, device)

    # Finally start the training/testing
    if train_mode:
        # Prepare the torch dataset object, and then the loader class
        dataset_path = data_dir_path / TRAIN_DIR
        dataset_obj = train_dataset.X4K1000FPS(dataset_path)
        data_loader = torch.utils.data.DataLoader(
            dataset_obj,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        start.train(network, data_loader, N_EPOCHS, CHKPT_EPOCHS, chkpt_dir_path, device, real_time_mode)
    else:
        dataset_path = test_path
        dataset_obj = test_dataset.X4K1000FPS(dataset_path)
        start.test(network, dataset_obj, output_dir_path, device, pretrained_path, real_time_mode)

