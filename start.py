# start.py -- Module containing the code for training/testing

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.io import write_video


# -------------------------- CONSTANTS --------------------------
MODEL_CHKPT_PREFIX = 'model'    # Prefix for model checkpoint
VIDEO_OP_EXTENSION = 'mp4'      # Extension of output video
# ---------------------------------------------------------------


def build_chkpt_dict(model, epoch, optimizer, loss):
    """ Builds the dictionary for saving the model """

    return {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }


def train(model, data_loader, n_epochs, chkpt_epochs, chkpt_dir, device, real_time_mode):
    """ Starts the training of the model """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()

    # Start the training
    for e in range(n_epochs):
        epoch_loss = 0

        for prev_frame, next_frame, target_frame in tqdm.tqdm(data_loader):
            prev_frame = prev_frame.to(device)
            next_frame = next_frame.to(device)
            target_frame = target_frame.to(device)

            # Predict the middle frame using the model
            pred_frame = model(prev_frame, next_frame)

            # Compute the MSE loss between the two frames
            loss = mse_loss(pred_frame, target_frame)

            # Compute the gradients and back-propagate the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Add the loss to check the training progress
            epoch_loss += loss.cpu().item()

        # Save the model
        # Naming format: [0|1]_model_[e]_[loss].pt
        if (e+1) % chkpt_epochs == 0:
            chkpt_path = str(chkpt_dir / f'{real_time_mode}_{MODEL_CHKPT_PREFIX}_{e}_{str(round(epoch_loss, 3))}.pt')
            chkpt_dict = build_chkpt_dict(model, e, optimizer, epoch_loss)
            torch.save(chkpt_dict, chkpt_path)

        print(f'Epoch[{e}] loss: {epoch_loss}\n')


def test(model, dataset_obj, output_dir, device, chkpt_path, real_time_mode):
    """
    Starts the testing of the model and saves the outputs in the specified
    output directory
    """

    # NOTE: Assumption is made that the correct dictionary is loaded for the model
    #       Failing to do so would throw errors
    chkpt_dict = torch.load(chkpt_path)
    model.load_state_dict(chkpt_dict['model_state_dict'])
    model = model.to(device)

    net_id = 'enet' if real_time_mode else 'default'
    video_id = dataset_obj.get_video_id()                               # To identify between the two networks
    video_fps_true, video_fps_inter = dataset_obj.get_video_fps()       # Get the frame-rate of original and interpolated video

    inter_video_frames = []     # List storing the frames + interpolated video frames
    true_video_frames = []      # List storing the actual frames

    # Start the predictions
    # NOTE: Observe that data-loader is not being used here. This is to ensure chronological
    # ordering of frames and to know where to save

    model.eval()        # Set the model to inference mode

    for i in tqdm.tqdm(range(len(dataset_obj))):
        prev_frame, next_frame = dataset_obj[i]

        # NOTE: Need to add the batch dimension manually because we are not using data-loaders here
        prev_frame_b = prev_frame.unsqueeze(0).to(device)
        next_frame_b = next_frame.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_frame = model(prev_frame_b, next_frame_b)
            pred_frame = pred_frame.squeeze(0).cpu()        # Remove the batch dimension and bring to CPU

        # Permute the tensor axes in the format (Height, Width, Channels)
        # Currently the frames are in the format (Channels, Height, Width)
        # Need to ensure to produce correct output in write_video(...)

        prev_frame = pred_frame.permute((1, 2, 0))
        next_frame = next_frame.permute((1, 2, 0))
        pred_frame = pred_frame.permute((1, 2, 0))

        # Save the frames into appropriate lists
        # NOTE: The following check is to avoid duplicate entries
        if not inter_video_frames:
            inter_video_frames += [prev_frame, pred_frame, next_frame]
            true_video_frames += [prev_frame, next_frame]

        else:
            inter_video_frames += [pred_frame, next_frame]
            true_video_frames += [next_frame]

    # Convert the frames to tensor and rescale them to fit the range [0, 255]
    # Need to ensure that they are uint8 because write_video(...) expects them to be so
    inter_video_frames_t = (torch.stack(inter_video_frames) * 255).to(torch.uint8)
    true_video_frames_t = (torch.stack(true_video_frames) * 255).to(torch.uint8)

    # Save the output videos in the output directory
    true_video_path = str(output_dir / f'{video_id}_true.{VIDEO_OP_EXTENSION}')
    inter_video_path = str(output_dir / f'{video_id}_{net_id}_pred.{VIDEO_OP_EXTENSION}')

    write_video(inter_video_path, inter_video_frames_t, fps=video_fps_inter)
    write_video(true_video_path, true_video_frames_t, fps=video_fps_true)

    print(f'True Video written to : {true_video_path}')
    print(f'Interpolated video written to: {inter_video_path}')
