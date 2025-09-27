import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # matplotlib does not work without this line
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import time
from data_loader import ClassifierDataset
from helpers import plot_classificator_metrics

class ImageClassificationModel(nn.Module):
    """
    A custom image classification model using a pre-trained ResNet18 architecture.

    This model is specifically designed to handle single-channel (grayscale) images
    and classify them into a specified number of classes. It leverages transfer
    learning by initializing with pre-trained weights from ResNet18.

    Args:
        num_classes (int): The number of output classes for the final classification layer.
    """
    def __init__(self, num_classes):
        super(ImageClassificationModel, self).__init__()
        # Use a pre-trained ResNet model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Modify the first convolutional layer for one channel grayscale input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the final layer for classification
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def trainer(
        train_image_dir,
        val_image_dir,
        num_epochs=1,
        batch_size=32,
        crop_size=128,
        learning_rate=0.001,
        device='cpu',
        file_to_save=None,
        load_from=None,
):

    """
    Orchestrates the training and validation process for an image classification model.

    This function sets up the data loaders, initializes the model, loss function, and optimizer,
    and runs the main training loop over a specified number of epochs. It also tracks key
    performance metrics for both training and validation sets, plots the results, and saves
    the trained model as a checkpoint.

    Args:
        train_image_dir (str): Path to the directory containing training images.
        val_image_dir (str): Path to the directory containing validation images.
        num_epochs (int, optional): The number of epochs to train the model. Defaults to 1.
        batch_size (int, optional): The number of images per batch. Defaults to 32.
        crop_size (int, optional): The size to which images are cropped. Defaults to 128.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
        device (str, optional): The device to run the training on ('cpu' or 'cuda'). Defaults to 'cpu'.
        file_to_save (str, optional): Path to the file to save the trained model to. If None defaults to f'classificator_checkpoints/classificator_checkpoint_{time.time()}.pth'
        load_from (str, optional): Path to the file to load the trained model from. Defaults to None.
    """

    train_dataset = ClassifierDataset(image_dir=train_image_dir, crop_size=crop_size)
    val_dataset = ClassifierDataset(image_dir=val_image_dir, crop_size=crop_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the model, loss function, and optimizer
    model = ImageClassificationModel(num_classes=train_dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # load previously saved weights to continue training
    if load_from and os.path.exists(load_from):
        print(f"Loading model weights from {load_from}...")
        checkpoint = torch.load(load_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['weights'])
        # load epoch
        if 'train_losses' in checkpoint:
            epoch = len(checkpoint['train_losses'])
        # load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model weights loaded successfully.")
    else:
        print("No pre-trained model found or path not specified. Starting training from scratch.")


    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    train_tprs = []
    train_frns = []
    val_tprs = []
    val_frns = []

    # Main training process
    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_tpr, train_frn = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, val_tpr, val_frn = validate(model, val_loader, criterion, device)

        # Store metrics for plots
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        train_tprs.append(train_tpr)
        train_frns.append(train_frn)
        val_tprs.append(val_tpr)
        val_frns.append(val_frn)


        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Valid Loss: {val_loss:.4f}, Valid Accuracy: {val_accuracy:.4f}, "
              )

    plot_classificator_metrics(train_losses, val_losses)

    # Save the model
    checkpoint = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'weights': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    file_to_save = file_to_save if file_to_save is not None else f'classificator_checkpoints/classificator_checkpoint_{time.time()}.pth'
    torch.save(checkpoint, file_to_save)

def train(model, train_loader, criterion, optimizer, device):

    """
    Performs one epoch of training for a given model.

    This function sets the model to training mode, iterates through the training data,
    computes the loss, performs a backward pass, and updates the model's weights.
    It also calculates and returns the average loss, accuracy, True Positive Rate (TPR),
    and False Negative Rate (FNR) for the epoch.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): The DataLoader for the training dataset.
        criterion (nn.Module): The loss function (e.g., nn.CrossEntropyLoss).
        optimizer (optim.Optimizer): The optimizer for updating model weights.
        device (str): The device to run the training on ('cpu' or 'cuda').

    Returns:
        tuple: A tuple containing the average loss, accuracy, TPR, and FNR for the epoch.
    """
    # Set the model to training mode.
    model.train()
    # Initialize variables to track metrics over the epoch.
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    true_positive = 0
    false_negative = 0
    # Iterate over the training dataset.
    for images, _, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate the loss for the current batch.
        running_loss += loss.item()
        _, predicted_labels = torch.max(outputs, 1)

        # Accumulate correct predictions and total predictions for accuracy calculation.
        correct_predictions += (predicted_labels == labels).sum().item()
        total_predictions += labels.size(0)

        # Calculate TP (True Positives) and FN (False Negatives)
        for label in range(model.resnet.fc.out_features):  # For each class
            true_positive += ((predicted_labels == label) & (labels == label)).sum().item()
            false_negative += ((predicted_labels != label) & (labels == label)).sum().item()

    # Calculate loss and accuracy
    loss = running_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions
    # Calculate TPR and FNR
    tpr = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    fnr = false_negative / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    return loss, accuracy, tpr, fnr


def validate(model, val_loader, criterion, device):
    """
    Performs one epoch of validation for a given model.

    This function evaluates the model on a validation dataset without updating its weights.
    It sets the model to evaluation mode, iterates through the data, and computes
    key performance metrics such as loss, accuracy, True Positive Rate (TPR), and
    False Negative Rate (FNR). The function uses torch.no_grad() to disable
    gradient calculations, which saves memory and speeds up computation.

    Args:
        model (nn.Module): The neural network model to be validated.
        val_loader (DataLoader): The DataLoader for the validation dataset.
        criterion (nn.Module): The loss function (e.g., nn.CrossEntropyLoss).
        device (str): The device to run the validation on ('cpu' or 'cuda').

    Returns:
        tuple: A tuple containing the average loss, accuracy, TPR, and FNR for the epoch.
    """
    # Set the model to evaluation mode.
    model.eval()

    # Initialize variables to track metrics over the epoch.
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    true_positive = 0
    false_negative = 0

    # `torch.no_grad()` context manager disables gradient calculations.
    with torch.no_grad():
        for images, _, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Accumulate the loss for the current batch.
            running_loss += loss.item()
            # Get the predicted labels by finding the class with the highest probability.
            _, predicted_labels = torch.max(outputs, 1)
            # Accumulate correct predictions and total predictions for accuracy calculation.
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)

            # Calculate True Positives (TP) and False Negatives (FN) for each class.
            # This loop aggregates TP and FN across all classes in a multi-class setting.
            for label in range(model.resnet.fc.out_features):  # For each class
                true_positive += ((predicted_labels == label) & (labels == label)).sum().item()
                false_negative += ((predicted_labels != label) & (labels == label)).sum().item()

    # Calculate loss and accuracy
    loss = running_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions

    # Calculate TPR (True Positive Rate) = TP / (TP + FN) and FNR (False Negative Rate) = FN / (TP + FN)
    tpr = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    fnr = false_negative / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    return loss, accuracy, tpr, fnr


def get_model_state(model_path, num_classes=2, device='cpu'):
    """
    Loads a pre-trained image classification model from a saved checkpoint file.

    This function is designed to restore a model's state from a `.pth` file, which
    contains the model's weights and other metadata saved during training. It initializes
    a new model, loads the state dictionary, and sets the model to evaluation mode.

    Args:
        model_path (str): The file path to the saved model checkpoint.
        num_classes (int, optional): The number of output classes. Defaults to 2.
        device (str, optional): The device to load the model onto ('cpu' or 'cuda').
                               Defaults to 'cpu'.

    Returns:
        nn.Module: The loaded and re-initialized model, ready for inference.
    """
    # Load the checkpoint dictionary from the specified path.
    # The `map_location` argument ensures the model is loaded onto the correct device.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Initialize a new instance of the model.
    # The architecture must match the one used to create the checkpoint.
    model = ImageClassificationModel(num_classes=num_classes).to(device)

    # Load the saved weights (state dictionary) into the newly initialized model.
    # The 'weights' key hold the model's state_dict.
    model.load_state_dict(checkpoint['weights'])

    # Set the model to evaluation mode.
    model.eval()

    # Return the fully loaded and prepared model.
    return model

def classify_image(model, image, device='cpu'):
    """
    Classifies a single image using the provided trained model.

    This function takes a pre-processed image tensor, moves it to the specified
    device, and uses the model to perform a forward pass. It operates in
    `torch.no_grad()` context to ensure no gradients are computed, which is
    standard practice for inference. The function returns the predicted class
    index.

    Args:
        model (nn.Module): The trained neural network model.
        image (torch.Tensor): The pre-processed image tensor, with shape
                              (1, C, H, W), where C=1 for grayscale.
        device (str, optional): The device to run the inference on ('cpu' or 'cuda').
                               Defaults to 'cpu'.

    Returns:
        int: The index of the predicted class.
    """
    # Prepare the image
    # transform = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=1),  # Convert image to 1-channel grayscale
    #     transforms.Resize((128, 128)),  # Resize image to match the input size of the model
    #     transforms.ToTensor(),  # Convpoert image to tensor
    #     transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values to [0,1]
    # ])

    # image = Image.open(image_path)
    # image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        # Move the image tensor to the specified device.
        image = image.to(device)
        outputs = model(image)
        # Get the predicted class index by finding the maximum value along dimension 1.
        _, predicted_class = torch.max(outputs, 1)

    return predicted_class.item()


def classify_dataset(model_path, dataset, batch_size, num_classes, device='cpu'):
    """
    Classify multiple images from a dataset using a pre-saved model.

    Args:
        model_path (str): Path to the pre-saved model file (.pth).
        dataset (torch.utils.data.Dataset): Dataset object containing the images to classify.
        batch_size (int): Batch size for processing the dataset.
        num_classes (int): Number of output classes in the model.
        device (str): Device to use ('cpu' or 'cuda').

    Returns:
        list: List of predicted class indices for all images in the dataset.
    """
    # Step 1: Initialize the model
    model = ImageClassificationModel(num_classes=num_classes).to(device)

    # Step 2: Load the pre-trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    # Step 3: Create the data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Step 4: Perform inference
    predictions = []
    with torch.no_grad():
        for images, _ in data_loader:  # Assuming dataset provides (image, label) pairs
            images = images.to(device)
            outputs = model(images)
            _, predicted_classes = torch.max(outputs, 1)  # Get predicted class for each image
            predictions.extend(predicted_classes.cpu().tolist())  # Add batch predictions to the list

    return predictions


def classify_dataset_with_metrics(model_path, dataset, batch_size, num_classes, device='cpu'):
    """
    Classify multiple images from a dataset using a pre-saved model and calculate TPR and FNR.

    Args:
        model_path (str): Path to the pre-saved model file (.pth).
        dataset (torch.utils.data.Dataset): Dataset object containing images and their labels.
        batch_size (int): Batch size for processing the dataset.
        num_classes (int): Number of output classes in the model.
        device (str): Device to use ('cpu' or 'cuda').

    Returns:
        dict: Dictionary containing predictions, TPR, and FNR for each class.
    """
    # Step 1: Initialize the model
    model = ImageClassificationModel(num_classes=num_classes).to(device)

    # Step 2: Load the pre-trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    # Step 3: Create the data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Step 4: Initialize metrics
    predictions = []
    true_labels = []
    true_positive = [0] * num_classes
    false_negative = [0] * num_classes
    total_labels_per_class = [0] * num_classes

    # Step 5: Perform inference and calculate metrics
    with torch.no_grad():
        for images, labels in data_loader:  # Assuming dataset provides (image, label) pairs
            images, labels = images.to(device), labels.to(device)

            # Perform prediction
            outputs = model(images)
            _, predicted_classes = torch.max(outputs, 1)  # Get predicted class for each image

            # Collect predictions and labels
            predictions.extend(predicted_classes.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

            # Update metrics
            for label in range(num_classes):
                true_positive[label] += ((predicted_classes == label) & (labels == label)).sum().item()
                false_negative[label] += ((predicted_classes != label) & (labels == label)).sum().item()
                total_labels_per_class[label] += (labels == label).sum().item()

    # Step 6: Calculate TPR and FNR
    tpr = [tp / total if total > 0 else 0 for tp, total in zip(true_positive, total_labels_per_class)]
    fnr = [fn / total if total > 0 else 0 for fn, total in zip(false_negative, total_labels_per_class)]

    return {
        "predictions": predictions,
        "tpr": tpr,
        "fnr": fnr
    }
