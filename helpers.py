import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' #matplotlib do not work without it
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import random
from einops import rearrange
import matplotlib.pyplot as plt
from typing import List
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Dataset data loader
from data_loader import SRDataset
from torch.utils.data import DataLoader

def set_seed(seed: int = 42):  # you know why 42 :)
    """
    Sets the random seed for reproducibility.

    Args:
        seed (int): The seed value to be set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize(x):
    # Find the global min and max values
    x_min = x.min()
    x_max = x.max()

    # Perform min-max normalization
    normalized_x = (x - x_min) / (x_max - x_min)
    return normalized_x

def image_loader(image_dir, num_images, crop_size, scale_factor):
    # Load and log the first 10 elements from the DataLoader
    test_dataset = SRDataset(image_dir=image_dir, crop_size=crop_size, scale_factor=scale_factor, cfg_factor=0)
    test_loader = DataLoader(test_dataset, shuffle=True)
    # list of low-res image tensors
    lrs = []
    # list of hi-res image tensors
    hrs = []
    # list of true image classes
    clss =[]
    # print(f"First {num_images} elements from DataLoader:")
    for idx, (hr, lr, ci) in enumerate(test_loader):
        if idx >= num_images:
            break
        hrs.append(hr.squeeze(0))
        lrs.append(lr.squeeze(0))
        clss.append(ci)
    return lrs, hrs, clss

def show_images(low_res: torch.Tensor, hi_res: torch.Tensor, generated: torch.Tensor):
    """
    Displays four images: low-res, hi-res, bicubic upscaled, and generated.
    All images are shown in a single row with the same size.

    Args:
        low_res (torch.Tensor): The low-resolution image tensor.
        hi_res (torch.Tensor): The high-resolution image tensor.
        generated (torch.Tensor): The generated high-resolution image tensor.
    """
    # 1. Convert tensors to PIL Images for upscaling
    # The torchvision.transforms.ToPILImage() function is used to convert tensors to PIL images.
    to_pil = transforms.ToPILImage(mode='L')
    low_res_pil = to_pil(low_res.cpu())

    # 2. Upscale the low-res image using bicubic interpolation
    # The size of the hi-res image is used as the target size for the bicubic upscaling.
    bicubic_upscaler = transforms.Resize(size=hi_res.shape[1:], interpolation=transforms.InterpolationMode.BICUBIC)
    bicubic_upscaled = bicubic_upscaler(low_res_pil)

    # 3. Create a list of the four images
    # The generated image needs to be converted to a PIL Image as well.
    images = [low_res_pil, to_pil(hi_res.cpu()), bicubic_upscaled, to_pil(normalize(generated.cpu()))]
    # images = [low_res, hi_res, normalize(bicubic_upscaled), normalize(generated)]
    titles = ['LR image', 'HR image', 'Bicubic interpolated', 'SR image (generated)']

    # 4. Plot the images
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for i, (ax, img, title) in enumerate(zip(axes.flat, images, titles)):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=12)
        ax.axis('off')  # Hide the axes
    plt.tight_layout()
    plt.show()

def display_imgs(images: List):
    ncols = len(images)
    assert ncols <= 10, "You can only display at most 10 images."
    fig, axes = plt.subplots(1, ncols, figsize=(10, 1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x, cmap='gray')  # grayscale here
        ax.axis('off')
    plt.show()


def plot_sr_metrics(train_losses):
    """
    Plots the training loss over a number of epochs.

    This function takes a list of loss values recorded during the training process
    and generates a single line plot to visualize the model's learning progress.
    The plot shows the training loss on the y-axis and the epoch number on the x-axis.

    Args:
        train_losses (list): A list of loss values from the training set for each epoch.
    """
    num_epochs = len(train_losses)
    # Plot the training loss
    plt.figure(figsize=(18, 12))
    plt.subplot(2, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

def plot_classificator_metrics(train_losses, val_losses):
    """
    Plots a single diagram for Training vs. Validation Loss.

    Args:
        train_losses (list): A list of loss values from the training set for each epoch.
        val_losses (list): A list of loss values from the validation set for each epoch.
    """
    assert len(train_losses) == len(val_losses), f'Train Loss and Validation Loss must have the same length.'
    num_epochs = len(train_losses)

    # Create a single figure for the first diagram.
    plt.figure(figsize=(10, 6))

    # Plot Training and Validation Loss.
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Display the plot.
    plt.show()

def show_conf_matrix(seq_1, seq_2, title, indexes, columns, xlabel, ylabel, color='Blues'):
    assert len(seq_1) == len(seq_2), f'Data sequences must have the same length.'
    # Generated image classification confusion matrix
    # Define the new desired order for the labels
    new_labels_order = [True, False]
    gen_cm = confusion_matrix(
        seq_1,
        seq_2,
        labels=new_labels_order
    )
    gen_cm_df = pd.DataFrame(
        gen_cm,
        index=indexes,
        columns=columns
    )
    # Plotting the confusion matrix using Seaborn
    plt.figure(figsize=(4, 3))
    sns.heatmap(gen_cm_df, annot=True, fmt='d', cmap=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    return gen_cm_df

def load_json(json_path=None):
    if json_path is None:
        json_path = 'stats/stat_data.json'
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
    return data
