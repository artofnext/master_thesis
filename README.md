# Master's Thesis Project:  Enhancing clinical diagnosis by reducing false-negative parameters using super resolution diffusion probabilistic model

This repository contains the source code for a master's thesis on **performing experiment with guided diffusion SR model to reduce false-negative diagnosis**.

The project uses a **Conda environment** for dependency management to ensure reproducibility and ease of setup.

-----

## Project Structure

The repository is organized as follows:

  * **`experiment.ipynb`**: Notebook main file for experiment (entry point).
  * **`experiment_runner.py`**: Contains function to run experiment process.
  * **`models_conditional.py`**: Contains the core guided diffusion model architecture.
  * **`trainer.py`**: Contains code for training guided diffusion model.
  * **`inference.py`**: Contains functions for guided diffusion model inference and image generation.
  * **`classificator.py`**: Contains the classificator model class and training function.
  * **`data_loader.py`**: Contains data loader classes for both models.
  * **`[classificator_checkpoints]/`**: Folder for saved classificator model's weights.
  * **`[ddpm_checkpoints]/`**: Folder for saved guided diffusion model's weights.
  * **`[stats]/`**: Folder for saved experiment statistics .json files.
  * **`helpers.py`**: Contains function for presenting results.
  * **`environment.yml`**: The Conda environment specification.
  * **`test_* .py`**: Test files.
  * **`README.md`**: This file.

-----

## Getting Started

### Prerequisites

You need **Conda** to manage the project environment. You can get it by installing [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### 1\. Clone the Repository

Clone the project from GitHub and navigate to the project directory.

```bash
git clone https://github.com/your-username/your-project-repo.git
cd your-project-repo
```

### 2\. Create and Activate the Conda Environment

Use the `environment.yml` file to create a new environment with all the necessary dependencies, including PyTorch.

```bash
conda env create -f environment.yml
conda activate [environment_name] # e.g., conda activate thesis-project
```

-----

## Usage

### Experiment 

To reproduce the experiment, run all cells in the experiment.ipynb jupyter notebook.

-----

## Code Overview
### Guided SR diffusion model

The code implements a **Guided SR diffusion model** based on a **U-Net** architecture. Key components include:

  * **`UNET`**: The primary model architecture, which includes all following blocks and layers.
  * **`ULayer`**: Main UNET block that forms downscale and upscale branches.
  * **`ResidualBlock`**: A fundamental building block that integrates periodical, conditional, and class embeddings.
  * **`Attention`**: A self-attention module to capture long-range dependencies in the feature maps.
  * **`Scheduler`**: Manages the noise schedule for the diffusion process.
  * **`ImageClassEmbedding`**: A module for injecting class information into the model using a learnable embedding.
  * **`PeriodicalEmbeddings`**: A module for inserting time-dependent information.
  * **`ConditionalEmbeddings`**: A module for providing low-resolution image information.
  * **`FiLM`**: A module to modulate normalization for class embedding.

### Image Classificator

Classificator is based on ResNet18 pre-trained model adjusted and fine-tuned to perform Alzheimer disease classification of the brain MRI images

  * **`ImageClassificationModel`**: The primary model architecture, which instantiate and adjust ResNet18 model.
  * **`trainer`**: Function orchestrates the training and validation process for an image classification model.
  * **`train`**: Function that performs one epoch of training for a given model.
  * **`validate`**: Function that performs model validation.
  * **`get_model_state`**: Function to load a pre-trained image classification model from a saved checkpoint file.
  * **`classify_image`**: Function that classifies a single image using the provided trained model.
  * **`classify_dataset`**: Function that classifies multiple images from a dataset using a pre-saved model.
  * **`classify_dataset_with_metrics`**: Function that classifies multiple images from a dataset using a pre-saved model and calculate TPR and FNR.
-----

## License

This project is licensed under the **[Your License Here]** License.

-----

## Acknowledgements

This project was completed as part of a master's thesis under the supervision of **Doctor João Pedro Oliveira, Associate Professor,
Iscte – Instituto Universitário de Lisboa**.