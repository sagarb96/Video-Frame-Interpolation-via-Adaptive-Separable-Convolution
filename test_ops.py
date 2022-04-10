# test_ops.py -- Test the computational complexities of the implemented models
#                for different input sizes

from ptflops import get_model_complexity_info
import torch
import net.net as net


# --------------------- CONSTANTS ---------------------
REAL_TIME_MODE = 1  # 0: Not real-time mode
INPUT_SIZE = (1, 3, 270, 512)
DEVICE = torch.device('cuda')
# -----------------------------------------------------


def vfi_constructor(input_shape):
    x = torch.ones(input_shape).to(DEVICE)
    return {
        'frame_prev': x,
        'frame_next': x
    }


if __name__ == '__main__':
    network = net.InterpolationNet(REAL_TIME_MODE, DEVICE).to(DEVICE)
    flops_count, params_count = get_model_complexity_info(
        network,
        INPUT_SIZE,
        as_strings=True,
        input_constructor=vfi_constructor,
        print_per_layer_stat=False)

    print('{:<30}  {:<8}'.format('Computational complexity: ', flops_count))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params_count))
