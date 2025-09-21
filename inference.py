import torch
from einops import rearrange
import matplotlib.pyplot as plt

from models_conditional import UNET, Scheduler
from data_loader import DataClasses
from helpers import normalize

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

    return normalize(x)


def get_sr_model_state(checkpoint_path, num_classes=2):
    checkpoint = torch.load(checkpoint_path)
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
