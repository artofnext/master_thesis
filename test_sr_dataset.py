import unittest
import os
import tempfile
from PIL import Image
# from torchvision.datasets import Dataset
from torchvision import transforms
import torch
from data_loader import SRDataset  # Assuming this is the file where the class resides


class TestSRDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to simulate image data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_dir = self.temp_dir.name

        # Simulate images spread across class subdirectories
        class_names = ['no', 'v_mild', 'mild', 'moderate', 'null']
        os.makedirs(self.image_dir, exist_ok=True)

        for class_name in class_names:
            class_subdir = os.path.join(self.image_dir, class_name)
            os.makedirs(class_subdir, exist_ok=True)

            # Create fake grayscale images in each directory (e.g., 10 per subdirectory)
            for i in range(10):
                image = Image.new('L', (128, 128), color=255)  # (L: grayscale, 128x128)
                image_path = os.path.join(class_subdir, f'{class_name}_{i}.png')
                image.save(image_path)

        # Initialize the SRDataset instance
        self.dataset = SRDataset(
            image_dir=self.image_dir,
            crop_size=32,
            scale_factor=2,
            patch=4,
            cfg_factor=0.2
        )

    def tearDown(self):
        # Cleanup the temporary directory after tests
        self.temp_dir.cleanup()

    def test_dataset_length(self):
        # Expected length: number of images x patch
        expected_length = len(self.dataset.files) * self.dataset.patch
        self.assertEqual(len(self.dataset), expected_length)

    def test_getitem(self):
        # Test the __getitem__ method for a single index
        idx = 0
        hr_image, lr_image, class_index = self.dataset[idx]

        # Check that the returned objects are tensors
        self.assertIsInstance(hr_image, torch.Tensor)
        self.assertIsInstance(lr_image, torch.Tensor)

        # Check tensor shapes
        self.assertEqual(hr_image.shape, torch.Size([1, self.dataset.crop_size, self.dataset.crop_size]))
        self.assertEqual(lr_image.shape, torch.Size([1, self.dataset.crop_size // self.dataset.scale_factor,
                                                     self.dataset.crop_size // self.dataset.scale_factor]))

        # Check that class_index is an integer and within valid class indices
        self.assertIsInstance(class_index, int)
        self.assertIn(class_index, [0, 1, 2,])

    def test_transformations(self):
        # Test if the defined transformations are applied
        idx = 0
        hr_image, _, _ = self.dataset[idx]  # Only need HR image for this test

        # Convert tensor back to PIL image and check if transformations are applied
        pil_image = transforms.ToPILImage()(hr_image)

        # Ensure the image dimensions match crop_size (32x32)
        self.assertEqual(pil_image.size, (self.dataset.crop_size, self.dataset.crop_size))

    def test_class_balancing(self):
        # Count the occurrences of each class in the dataset
        class_counts = {i: 0 for i in range(len(self.dataset.classes))}
        print("Class counts: ",class_counts)
        for _, _, class_index in self.dataset:
            class_counts[class_index] += 1
        print("Class counts values: ",class_counts.values())
        # Check if the distribution roughly matches the balancing factor (cfg_factor)
        total_samples = sum(class_counts.values())
        print("Total samples: ",total_samples)
        actual_distribution = {k: v / total_samples for k, v in class_counts.items()}
        print("Actual distribution: ",actual_distribution)
        # Assert that the actual distribution for all classes the same and for "null" resembles a balanced outcome within a certain tolerance
        cfgf = (1- self.dataset.cfg_factor) / len(self.dataset.classes)
        expected_factor = [cfgf] * (len(self.dataset.classes))
        expected_factor[len(expected_factor) - 1] = cfgf + self.dataset.cfg_factor
        print("Expected factor: ",expected_factor)
        for class_idx, actual_ratio in actual_distribution.items():
            self.assertAlmostEqual(actual_ratio, expected_factor[class_idx], delta=0.1)  # Tolerance of ±10%


if __name__ == '__main__':
    unittest.main()