import os
import torch
from torch import cuda
from torchvision.transforms import ToPILImage
import helpers
import data_loader
from pathlib import Path
from einops import rearrange
import matplotlib.pyplot as plt
from models_conditional import UNET, Scheduler
from data_loader import DataClasses


def count_files_recursive(start_dir: str) -> int:
    """
    Recursively scans a directory and its subdirectories and counts the total
    number of files found.

    This function uses os.walk to efficiently traverse the directory structure.

    Args:
        start_dir: The path to the starting directory (string).

    Returns:
        The total count of files (integer).
    """
    if not os.path.isdir(start_dir):
        # Handle the case where the provided path is not a directory
        print(f"Error: The path '{start_dir}' is not a valid directory.")
        return 0

    file_count = 0

    for root, dirs, files in os.walk(start_dir):
        # The 'files' list contains all non-directory files in the current 'root' directory.
        file_count += len(files)
    return file_count


def generate(source_image_path, path_to_save, model_file):
    counter = 0

    class_folders = [
        'no',
        'v_mild',
        'mild',
        'moderate',
    ]

    num_steps = 500

    # Create data folder and class subfolders if not exist
    if not os.path.exists(path_to_save):
        print(f"Output folder '{path_to_save}' not found. Creating directory.")
        # if the OUTPUT_DIR directory is not present then create it.
        os.makedirs(path_to_save)
        for folder in class_folders:
            os.mkdir(path_to_save + '/' + folder)
            print(f'Created class folder {path_to_save + '/' + folder}')

    # Load images from dataset to make inference
    crop_size = 128
    print('source_image_path: ', source_image_path)
    num_images_to_generate = count_files_recursive(source_image_path)

    # lrs - Low-resolution images list
    # hrs - Hi-resolution images list
    # clss - List of ground true class of images
    # path - List of path to the images
    lrs, hrs, clss, path = helpers.image_loader(
        image_dir=source_image_path,
        num_images=num_images_to_generate,
        crop_size=crop_size,
        scale_factor=2,
    )

    print(f'Number of images: \t\t{len(lrs)}')
    print(f'Number of images to generate: \t{num_images_to_generate}')
    if num_images_to_generate > len(lrs):
        num_images_to_generate = len(lrs)
        print(
            f'Can\'t generate more images that DataLoader found for inference. Number of images to generate: {num_images_to_generate}')

    model = get_sr_model_state(model_file, num_classes=3)
    # creating scheduler
    scheduler = get_scheduler(num_steps)
    # Get "null" class for guideless inference
    null_class = torch.tensor([data_loader.DataClasses.get_null_class_index(data_loader.DataClasses)])

    # generating image examples (will take some time)
    for idx in range(num_images_to_generate):
        generated_image = generate_image(128, lrs[idx], null_class, model, scheduler, num_steps)

        # If the tensor is a batch (e.g., shape [B, C, H, W]), select one image
        if generated_image.dim() == 4:
            generated_image = generated_image[0]

        # Check if the tensor is grayscale (1 channel)
        if generated_image.shape[0] == 1:
            # Remove the channel dimension for grayscale conversion
            generated_image = generated_image.squeeze(0)

        # normalize
        generated_image = helpers.normalize(generated_image)

        # Convert the tensor to a PIL Image
        to_pil = ToPILImage()
        pil_image = to_pil(generated_image.cpu())

        # Specify the path to save the image
        p = Path(path[idx][0])
        save_image_path = path_to_save + '/' + str(p.parent.name) + '/' + str(p.name)

        # Save the PIL Image
        pil_image.save(save_image_path)

        counter += 1
        if idx % 100 == 0:
            print(f'Generated image \t{idx}')
            print(f'Generated images: \t{(100 * idx / num_images_to_generate):.2f} %')
        cuda.empty_cache()
    print(f'Finished generating images. Generated {counter} images, {(100 * counter / num_images_to_generate):.2f}')


def generate_image(img_size, lr, class_index, model, scheduler, num_time_steps):
    # torch.cuda.empty_cache()
    with torch.no_grad():
        z = torch.randn(1, 1, img_size, img_size)
        lr = lr.cuda()
        ci = class_index.cuda()
        for t in reversed(range(1, num_time_steps)):
            t = [t]
            temp = (scheduler.beta[t] / (
                    (torch.sqrt(1 - scheduler.alpha[t])) * (torch.sqrt(1 - scheduler.beta[t]))))
            z = (1 / (torch.sqrt(1 - scheduler.beta[t]))) * z - (temp * model(z.cuda(), t, lr, ci).cpu())
            e = torch.randn(1, 1, img_size, img_size)
            z = z + (e * torch.sqrt(scheduler.beta[t]))
        temp = scheduler.beta[0] / ((torch.sqrt(1 - scheduler.alpha[0])) * (torch.sqrt(1 - scheduler.beta[0])))
        x = (1 / (torch.sqrt(1 - scheduler.beta[0]))) * z - (temp * model(z.cuda(), [0], lr, ci).cpu())

    return x


def get_sr_model_state(checkpoint_path, num_classes=2):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = UNET(num_classes=num_classes).cuda()
    model.load_state_dict(checkpoint['weights'])
    return model

def get_scheduler(num_time_steps):
    return Scheduler(num_time_steps=num_time_steps)

def inference_sr(low_res_images: list,
                high_res_images: list = None,
                class_indexes: list = None,
                img_size: int = 32,
                checkpoint_path: str = None,
                num_time_steps: int = 500
                 ):
    data_classes = DataClasses()
    num_classes = len(data_classes.CLASSES) + 1  # + null class
    null_class_idx = data_classes.NULL_CLASS[1]
    model = get_sr_model_state(checkpoint_path, num_classes=num_classes)
    scheduler = get_scheduler(num_time_steps)

    assert len(low_res_images) == len(high_res_images) == len(class_indexes), "All lists must have the same length."

    with torch.no_grad():

        for i in range(len(low_res_images)):

            x = generate_image(img_size, low_res_images[i], class_indexes[i], model, scheduler, num_time_steps)
            x_null = generate_image(img_size, low_res_images[i], torch.tensor([null_class_idx]), model, scheduler, num_time_steps)
            lr = low_res_images[i]

            fig, axes = plt.subplots(1, 4, figsize=(10, 3))

            # Prepare low resolution image
            lr = rearrange(lr, 'c h w -> h w c').detach().numpy()
            axes[0].imshow(1 - lr, cmap='gray')
            axes[0].set_title("Low Res")
            axes[0].axis('off')

            # Prepare original high resolution image
            hr = rearrange(high_res_images[i], 'c h w -> h w c').detach().numpy()
            axes[1].imshow(1 - hr, cmap='gray')
            axes[1].set_title("High Res")
            axes[1].axis('off')

            # Prepare generated high resolution image
            x = rearrange(x.squeeze(0), 'c h w -> h w c').detach().numpy()
            axes[2].imshow(1 - x, cmap='gray')
            axes[2].set_title(f"{x.shape[0]}x{x.shape[1]} ci={class_indexes[i]}")
            axes[2].axis('off')

            x_null = rearrange(x_null.squeeze(0), 'c h w -> h w c').detach().numpy()
            axes[3].imshow(1 - x_null, cmap='gray')
            axes[3].set_title(f"{x_null.shape[0]}x{x_null.shape[1]} - {data_classes.NULL_CLASS[0]}")
            axes[3].axis('off')

            plt.tight_layout()
            plt.show()
