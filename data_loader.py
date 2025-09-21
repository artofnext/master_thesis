import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DataClasses:
    CLASSES = {
        'healthy': 0,
        'unhealthy': 1,
    }

    DATA_CLASSES = {
        'no': 0,
        'v_mild': 1,
        'mild': 2,
        'moderate': 3,
    }

    NULL_CLASS = ('null', 2)

    @staticmethod
    def get_null_class_index(cls):
        return cls.NULL_CLASS[1]

    @staticmethod
    def map_data_to_class(data_class: str) -> int:
        """
        Map a data class from DATA_CLASSES to the corresponding value in CLASSES.

        Args:
            data_class (str): The key from DATA_CLASSES.

        Returns:
            int: The corresponding value in CLASSES.
        """
        if data_class in DataClasses.DATA_CLASSES:
            value = DataClasses.DATA_CLASSES[data_class]
            if value == 0:  # 'no' corresponds to 'healthy'
                return DataClasses.CLASSES['healthy']
            else:  # Any other value corresponds to 'unhealthy'
                return DataClasses.CLASSES['unhealthy']
        else:
            raise ValueError(f"'{data_class}' is not a valid data class.")



# class SRDataset(Dataset):
#     def __init__(self, image_dir, crop_size=32, scale_factor=2):
#         self.image_dir = image_dir
#         self.crop_size = crop_size
#         self.scale_factor = scale_factor
#         self.files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
#         # Define transformations
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             # Add more transformations as needed, e.g., normalization, augmentation
#         ])
#
#     def __len__(self):
#         # Implement logic to count the number of images
#         return len(self.files)
#
#     def __getitem__(self, idx):
#         """
#         Retrieves a high-resolution and low-resolution image pair from the dataset
#         at the given index. The images are constructed from the original grayscale
#         image by applying respective scaling factors.
#
#         The low-resolution image is resized to the crop size divided by the scale
#         factor, while the high-resolution image retains the original crop size.
#         Both images undergo the specified transformations.
#
#         :param idx: Index of the data point to retrieve.
#         :type idx: int
#         :return: A tuple containing the high-resolution image and the low-resolution
#             image after applying the transformations.
#         :rtype: Tuple[torch.Tensor, torch.Tensor]
#         """
#         # Load the image
#         image_path = os.path.join(self.image_dir, self.files[idx])
#         # image = Image.open(image_path).convert('RGB')  # color image
#         raw_image = Image.open(image_path).convert('L')  # grayscale image
#
#         # Pad image to square
#         max_dim = max(raw_image.width, raw_image.height)
#         image = transforms.Pad((
#             max(0, (max_dim - raw_image.width) // 2),
#             max(0, (max_dim - raw_image.height) // 2)
#         ))(raw_image)
#
#         # Crop the image to fit the target
#         hr_image = transforms.Resize((self.crop_size, self.crop_size))(image)
#
#         # Create low-resolution and high-resolution images
#         lr_image = transforms.Resize(self.crop_size // self.scale_factor)(hr_image)
#         # hr_image = transforms.Resize(self.crop_size)(image)
#
#         # Apply transformations
#         lr_image = self.transform(lr_image)
#         hr_image = self.transform(hr_image)
#
#         # raw_image = self.transform(raw_image)
#
#         return hr_image, lr_image

class SRDataset(Dataset, DataClasses):
    def __init__(self, image_dir, crop_size=32, scale_factor=2, patch=4, cfg_factor=0.2):
        self.classes = self.__class__.DATA_CLASSES
        self.out_classes = self.__class__.CLASSES
        self.null_class = self.__class__.NULL_CLASS
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.is_cfg = cfg_factor > 0
        self.cfg_factor = cfg_factor  # "null" token probability cfg factor
        self.num_classes = len(self.out_classes) if not self.is_cfg else len(self.out_classes) + 1
        self.patch = patch if patch > 1 else 1

        self.files = [os.path.join(root, f) for root, _, files in os.walk(image_dir) for f in files if
                      os.path.isfile(os.path.join(root, f))]

        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),  # Augmentation: Random horizontal flip
        ])


    def __len__(self):
        # Implement logic to count the number of images
        return len(self.files) * self.patch

    def __getitem__(self, idx):

        # Load the image
        path = self.files[idx//self.patch]

        # Get image class
        last_dir = os.path.basename(os.path.dirname(path))

        class_index = self.__class__.map_data_to_class(last_dir)

        if self.is_cfg:
            # randomly "null" token index with cfg_factor probability
            class_index = class_index if random.random() > self.cfg_factor else self.null_class[1]

        image_path = os.path.join(self.image_dir, path)
        raw_image = Image.open(image_path).convert('L')  # grayscale image

        # Crop random region from the image at target size
        hr_image = transforms.RandomCrop(self.crop_size, pad_if_needed=True)(raw_image)

        # Apply transformations
        hr_image = self.transform(hr_image)

        # Create low-resolution image
        lr_image = transforms.Resize(self.crop_size // self.scale_factor)(hr_image)

        return (
            hr_image,
            lr_image,
            class_index,
        )


class ClassifierDataset(Dataset, DataClasses):
    def __init__(self, image_dir, crop_size=32, downscale=1):
        self.classes = self.__class__.DATA_CLASSES
        self.out_classes = self.__class__.CLASSES
        self.num_classes = len(self.out_classes)
        self.image_dir = image_dir
        self.crop_size = crop_size
        assert downscale >= 1
        self.downscale = downscale
        self.files = [os.path.join(root, f) for root, _, files in os.walk(image_dir) for f in files if
                      os.path.isfile(os.path.join(root, f))]

        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),  # Augmentation: Random horizontal flip
        ])


    def __len__(self):
        # Implement logic to count the number of images
        return len(self.files)

    def __getitem__(self, idx):

        # Load the image
        path = self.files[idx]

        # Get image class
        last_dir = os.path.basename(os.path.dirname(path))

        class_index = self.__class__.map_data_to_class(last_dir)
        image_path = os.path.join(self.image_dir, path)
        image = Image.open(image_path).convert('L')  # grayscale image

        # Apply transformations
        image = self.transform(image)

        if self.downscale > 1:
            lr_image = transforms.Resize(self.crop_size // self.downscale)(image)
        else:
            lr_image = image

        return (
            image,
            lr_image,
            class_index,
        )

