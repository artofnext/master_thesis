import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' #matplotlib do not work without it
import time

import torch.nn as nn
import torch
from models_conditional import UNET, Scheduler
from data_loader import SRDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from helpers import plot_sr_metrics

def train_sr(
        image_dir,
        crop_size=32,
        scale_factor=2,
        batch_size: int = 64,
        num_time_steps: int = 1000,
        num_epochs: int = 15,
        lr=2e-5,
        checkpoint_path: str = None,
        file_to_save: str = None,
):

    """
    Trains a super-resolution diffusion model (SR-DDPM).

    This function orchestrates the training process for a diffusion-based super-resolution
    model. It loads the dataset, initializes the UNet model, a custom scheduler, and an
    optimizer. It runs the training loop, periodically prints the loss, and saves a
    checkpoint of the model's weights and optimizer state at the end. It can also
    resume training from a previously saved checkpoint.

    Args:
        image_dir (str): Path to the directory containing the training images.
        crop_size (int, optional): The size of the cropped images. Defaults to 32.
        scale_factor (int, optional): The upscaling factor for super-resolution. Defaults to 2.
        batch_size (int, optional): Number of images per batch. Defaults to 64.
        num_time_steps (int, optional): Number of diffusion time steps. Defaults to 1000.
        num_epochs (int, optional): Number of training epochs. Defaults to 15.
        seed (int, optional): Random seed for reproducibility. Defaults to 23.
        lr (float, optional): Learning rate for the Adam optimizer. Defaults to 2e-5.
        checkpoint_path (str, optional): Path to a model checkpoint to resume training from.
                                         Defaults to None.
        file_to_save (str, optional): Path to a checkpoint file to save model. If None defaults to 'ddpm_checkpoints/ddpm_sr_checkpoint{time.time()}.pth'.
    """
    #  Set default path to save model's weights
    file_to_save_default = f'ddpm_checkpoints/ddpm_sr_checkpoint{time.time()}.pth'

    # Initialize the dataset and data loader.
    train_dataset = SRDataset(image_dir=image_dir, crop_size=crop_size, scale_factor=scale_factor, cfg_factor=0.2)
    # TODO debugging
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Number of classes: {train_dataset.num_classes}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    # Initialize the diffusion process scheduler and the UNet model.
    scheduler = Scheduler(num_time_steps=num_time_steps)
    model = UNET(num_heads=8, num_classes=train_dataset.num_classes).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss = []
    start_epoch = 1
    # Check for a checkpoint path to resume training.
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        train_loss = torch.load(checkpoint_path)['train_loss']
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = len(train_loss)
    # Initialize the  Mean Squared Error (MSE) loss function.
    criterion = nn.MSELoss(reduction='mean')

    # Initialize memory snapshot https://pytorch.org/blog/understanding-gpu-memory-1/
    # torch.cuda.memory._record_memory_history(
    #     max_entries= 10000
    # )
    # file_prefix = f"memory_snapshot_{time.time()}"

    # Main training loop.
    for i in range(num_epochs):
        total_loss = 0
        # Use tqdm to show a progress bar for the training loop.
        for bidx, (x, y, ci) in enumerate(tqdm(train_loader, desc=f"Epoch {i + start_epoch}/{num_epochs}")):
            x = x.cuda()
            y = y.cuda()
            ci = ci.cuda()

            # Sample a random time step 't' and a random noise 'e'.
            t = torch.randint(0, num_time_steps, (batch_size,))
            e = torch.randn_like(x, requires_grad=False)

            # Sample a random time step 't' and a random noise 'e'.
            a = scheduler.alpha[t].view(batch_size, 1, 1, 1).cuda()
            x = (torch.sqrt(a) * x) + (torch.sqrt(1 - a) * e)
            # Forward - predict the noise `e` from the noisy image `x`.
            output = model(x, t, y, ci)
            optimizer.zero_grad()
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Save memory snapshot
            # try:
            #     torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
            # except Exception as e:
            #     logger.error(f"Failed to capture memory snapshot {e}")

        # Calculate and print the average loss for the epoch.
        train_loss.append(total_loss / (len(train_dataset) / batch_size))
        print(f'Epoch {i + start_epoch} | Loss {total_loss / (len(train_dataset) / batch_size):.5f}')

        # Save model's weights for every 10-th epoch
        if (i+1) % 10 == 0:
            temp_checkpoint_file = file_to_save if file_to_save else file_to_save_default
            temp_file_name, temp_file_ext = temp_checkpoint_file.split(".")
            temp_file_name = f"{temp_file_name}_epoch-{i + start_epoch}.{temp_file_ext}"
            checkpoint = {
                'train_loss': train_loss,
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, temp_file_name)

    # Stop recording memory snapshot history.
    # torch.cuda.memory._record_memory_history(enabled=None)

    # Save final weights
    file_to_save = file_to_save if file_to_save is not None else file_to_save_default
    checkpoint = {
        'train_loss': train_loss,
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(checkpoint, file_to_save)

    # Plot the training loss over epochs.
    plot_sr_metrics(train_loss)




