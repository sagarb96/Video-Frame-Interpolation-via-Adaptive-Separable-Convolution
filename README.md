# Real-Time Video Frame Interpolation via Adaptive Separable Convolution

## Abstract
Existing paper[4], although has reduced the memory requirements of [3] for high-resolution
(720p) videos, it still depends on the traditional encoder-decoder architecture based convolu-
tional neural-network (similar to SegNet[2]) for extracting the features. This makes it difficult
to be deployed in real-time scenarios because of the amount of computations needed to be performed 
to extract the features. We propose substituting this encoder-decoder architecture with
another one which is specifically meant for real-time tasks (for ex., ENet[1]) and try to quantify
the performance improvement while trading-off the quality of the output on a (spatially scaled-down) subset of 
X4K1000FPS dataset, used in XVFI [5].



## Setting Up the Environment
Ideally, set up a virtual environment using `venv` module (or any other way) and then install the dependencies
```
(my_virtual_env) $ pip3 install requirements.txt
```

> **NOTE:** There is hard-requirement for the system to have Nvidia GPU with CUDA support. Make sure to uninstall
> `cupy-cuda112==10.2.0` (which `requirements.txt` contains) if your system's CUDA toolkit's version differs and install
> the version that matches with the CUDA toolkit.
> Else the code will **not work**. This is a nasty issue, but there is no other choice.
>

## Training / Testing
Make sure you have installed the proper dependencies and have a CUDA supported Nvidia GPU, as mentioned above.
The training dataset we used is already present under `dataset/train` and the testing data is under `dataset/test`.

- To train the model, do the following
    ```
    (my_virtual_env) $ python3 main.py --realTime {0 | 1} --train 1
    ```
    where `--realTime` flag corresponds to whether to use real-time mode (value of `1`) or not. Enabling it will use ENet as
    the feature extraction network, else will use the default encoder-decoder network, as mentioned in the paper.
    
    > **NOTE:** To change number of epochs/batch-size, change `N_EPOCHS` / `BATCH_SIZE` inside `main.py`
    
- To test the network, ensure that you have downloaded the pretrained models. Do the following
    ```
    (my_virtual_env) $ python3 main.py --realTime {0 | 1} --train 0 --pretrained /path/to/pretrained/model --testPath /path/to/test/directory
    ```
    where `--pretrained` flag needs to set to the path of the pretrained model and `--testPath` needs to be set to the 
    path of the testing directory.
    
    The outputs will be present under `output/` directory. 
    
    > **NOTE:** We have provided our obtained outputs inside `output/from_our_own_dataset/`
    and `output/from_x4k_dataset/` directories.


## Regarding our Pre-Trained Models
The file sizes ( > 250MB) exceeded the max.limit (100MB) imposed by GitHub, hence had to be uploaded to
[Google Drive](https://drive.google.com/drive/folders/1a_Z2IXpHbaq4RedIkShaG4WPbdOuP7wn?usp=sharing). The drive contains
two files, which needs to be put under some directory. Ideally, put it under `checkpoints/` directory in this
repository's directory.
- `default_model.pt`, which corresponds to the model presented in the paper
- `realtime_model.pt`, which corresponds to the model presented in the paper, with ENet as the encoder-decoder

Use them for testing purposes, if needed. This is because testing needs a pretrained model.
> **NOTE:** The drive link can only be accessed using LDAP

## Code References
- https://github.com/sniklaus/sepconv-slomo
- https://github.com/HyeongminLEE/pytorch-sepconv

Although we had implemented the paper's model from scratch, we had borrowed (and modified) an entire file ([`sepconv.py`](https://github.com/HyeongminLEE/pytorch-sepconv/blob/master/sepconv.py))
which is written in CUDA + Python (we couldn't make sense out of it and this module is the last part of the network, which we kind
of feel like a discrepancy between the claim in the paper and the implementation). Similarly, some lines of code have
also been borrowed from the above two repositories, where we couldn't understand few things. Eitherways, we have
properly mentioned in the comments in our code, wherever necessary.


## Paper References
[1] Paszke, A., Chaurasia, A., Kim, S. and Culurciello, E., 2016. Enet: A deep neural network architecture for real-time 
semantic segmentation. arXiv preprint arXiv:1606.02147.

[2] V. Badrinarayanan, A. Kendall and R. Cipolla, ”SegNet: A Deep Convolutional Encoder-Decoder Architecture for 
Image Segmentation,” in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 12, pp. 2481-2495, 
1 Dec. 2017, doi: 10.1109/TPAMI.2016.2644615.

[3] S. Niklaus, L. Mai and F. Liu, ”Video Frame Interpolation via Adaptive Convolution,” 2017 IEEE Conference on Computer 
Vision and Pattern Recognition (CVPR), 2017, pp. 2270-2279, doi: 10.1109/CVPR.2017.244.

[4] Niklaus, S., Mai, L. and Liu, F., 2017. Video frame interpolation via adaptive separable convolution. 
In Proceedings of the IEEE International Conference on Computer Vision (pp. 261-270).

[5] H. Sim, J. Oh and M. Kim, ”XVFI: eXtreme Video Frame Interpolation,” 2021 IEEE/CVF International Conference on Computer 
Vision (ICCV), 2021, pp. 14469-14478, doi: 10.1109/ICCV48922.2021.01422.