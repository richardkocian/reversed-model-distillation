# --------------------------------------------
# File: train_model.py
# Description: Script for training models on datasets (CIFAR-10, Fashion-MNIST, California Housing)
#              without distillation.
# Author: Richard Kocián
# Created: 18.03.2025
# --------------------------------------------

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from configs import config
import os
import argparse

from data.datasets import get_loaders
from utils.set_seed import set_seed
from utils.test_model import test_model_classification, test_model_regression
from models.cifar import TeacherModelSmallCIFAR, TeacherModelMediumCIFAR, TeacherModelLargeCIFAR, StudentModelCIFAR
from models.fashion_mnist import TeacherModelSmallFashionMNIST, TeacherModelMediumFashionMNIST, TeacherModelLargeFashionMNIST, StudentModelFashionMNIST
from models.california_housing import TeacherModelSmallCALIFORNIA, TeacherModelMediumCALIFORNIA, TeacherModelLargeCALIFORNIA, StudentModelCALIFORNIA

def get_teacher_model(dataset, model_type):
    """
    Returns the appropriate teacher or student model based on the dataset and model type.

    :param dataset: str, the name of the dataset. Valid options are "cifar10", "fashion_mnist", and "california_housing".
    :param model_type: str, the type of model to return. Valid options are
                       "TeacherModelSmall", "TeacherModelMedium", "TeacherModelLarge", and "StudentModel".

    :return: nn.Module, the corresponding teacher or student model instance.
    :raises ValueError: If the combination of dataset and model type is not recognized.
    """
    if dataset == "cifar10":
        if model_type == "TeacherModelSmall":
            return TeacherModelSmallCIFAR()
        elif model_type == "TeacherModelMedium":
            return TeacherModelMediumCIFAR()
        elif model_type == "TeacherModelLarge":
            return TeacherModelLargeCIFAR()
        elif model_type == "StudentModel":
            return StudentModelCIFAR()
    elif dataset == "fashion_mnist":
        if model_type == "TeacherModelSmall":
            return TeacherModelSmallFashionMNIST()
        elif model_type == "TeacherModelMedium":
            return TeacherModelMediumFashionMNIST()
        elif model_type == "TeacherModelLarge":
            return TeacherModelLargeFashionMNIST()
        elif model_type == "StudentModel":
            return StudentModelFashionMNIST()
    elif dataset == "california_housing":
        if model_type == "TeacherModelSmall":
            return TeacherModelSmallCALIFORNIA()
        elif model_type == "TeacherModelMedium":
            return TeacherModelMediumCALIFORNIA()
        elif model_type == "TeacherModelLarge":
            return TeacherModelLargeCALIFORNIA()
        elif model_type == "StudentModel":
            return StudentModelCALIFORNIA()
    else:
        raise ValueError("Unknown combination of dataset and model")

def train_model_classification(train_loader, model, optimizer, criterion):
    """
    Trains a classification model using the provided data loader, optimizer, and loss function.

    :param train_loader: DataLoader, data loader for the training dataset.
    :param model: nn.Module, the classification model to be trained.
    :param optimizer: torch.optim.Optimizer, the optimizer used to update model parameters.
    :param criterion: nn.Module, the loss function used to compute the model's error.
    :return: list, the recorded training losses throughout the epochs.
    """
    epochs = config.EPOCHS
    model.train()
    training_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # Save every 100 mini-batches
                training_losses.append(running_loss / 100)
                running_loss = 0.0
    return training_losses

def train_model_reggresion(train_data, model, optimizer, criterion):
    """
    Trains a regression model using the provided data loader, optimizer, and loss function.

    :param train_data: tuple, a tuple containing training data and labels ([X_train, y_train]).
    :param model: nn.Module, the regression model to be trained.
    :param optimizer: torch.optim.Optimizer, the optimizer used to update model parameters.
    :param criterion: nn.Module, the loss function used to compute the model's error.
    :return: list, the recorded training losses throughout the epochs.
    """
    epochs = config.EPOCHS_REGRESSION
    X_train, y_train = train_data
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    model.train()
    training_losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        loss.backward()
        optimizer.step()

        training_losses.append(loss.item())
        #print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    return training_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model Without Distillation")
    parser.add_argument("--datasets-path", type=str, required=True, help="Path to the datasets folder")
    parser.add_argument("--output", type=str, required=True, help="Path to the outputs folder")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training (default: 64)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of worker threads for data loading (default: 4)")
    parser.add_argument("--seeds-file", type=str, required=True, help="Path to the seeds list txt file")
    parser.add_argument("--model", type=str, required=True,
                        choices=["TeacherModelSmall", "TeacherModelMedium", "TeacherModelLarge", "StudentModel"],
                        help="Model to train")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar10", "fashion_mnist", "california_housing"], help="Dataset")

    args = parser.parse_args()
    datasets_path = args.datasets_path
    outputs_path = args.output
    batch_size = args.batch_size
    num_workers = args.num_workers
    seeds_file = args.seeds_file
    dataset = args.dataset
    model_type = args.model

    device = torch.device("cuda" if config.USE_CUDA and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seeds = np.loadtxt(seeds_file, dtype=int).tolist()

    for run, seed in enumerate(seeds):
        print(f"Training Model {run + 1}/{len(seeds)} (seed: {seed})...")
        set_seed(seed)

        train_loader, test_loader = get_loaders(datasets_path=datasets_path, batch_size=batch_size,
                                                    num_workers=num_workers, dataset=dataset)
        model = get_teacher_model(dataset, model_type).to(device)
        student_optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        if dataset == "california_housing":
            criterion = nn.MSELoss()
            training_losses = train_model_reggresion(train_loader, model, student_optimizer, criterion)
            accuracy = test_model_regression(model, test_loader, device)
        else:
            criterion = nn.CrossEntropyLoss()
            training_losses = train_model_classification(train_loader, model, student_optimizer, criterion)
            accuracy = test_model_classification(model, test_loader, device)

        save_dir = f"{outputs_path}/model_seed_{seed}"
        print(f"Saving Model to {save_dir}...")
        os.makedirs(save_dir, exist_ok=True)
        np.savetxt(f"{save_dir}/training_losses.txt", training_losses)
        decimal_places = 6 if dataset == "california_housing" else 2
        np.savetxt(f"{save_dir}/accuracy.txt", [accuracy], fmt=f"%.{decimal_places}f")
        torch.save(model, os.path.join(save_dir, f"teacher_model.pth"))
