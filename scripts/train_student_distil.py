# --------------------------------------------
# File: train_student_distill.py
# Description: Script for training models on datasets (CIFAR-10, Fashion-MNIST, California Housing)
#              with reversed model distillation.
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
import torch.nn.functional as F
import gc
import time
import csv

from data.datasets import get_loaders
from utils.set_seed import set_seed
from utils.test_model import test_model_classification, test_model_regression, test_model_fgsm_classification, test_model_fgsm_regression
from models.cifar import StudentModelCIFAR
from models.fashion_mnist import StudentModelFashionMNIST
from models.california_housing import StudentModelCALIFORNIA

def get_student_model(dataset):
    """
    Returns the appropriate student model based on the dataset.

    :param dataset: str, the name of the dataset. Valid options are "cifar10", "fashion_mnist", and "california_housing".

    :return: nn.Module, the corresponding teacher or student model instance.
    :raises ValueError: If the combination of dataset and model type is not recognized.
    """
    if dataset == "cifar10":
        return StudentModelCIFAR()
    elif dataset == "fashion_mnist":
        return StudentModelFashionMNIST()
    elif dataset == "california_housing":
        return StudentModelCALIFORNIA()
    else:
        raise ValueError("Unknown combination of dataset and model")

def save_results(dataset, save_dir, training_losses, accuracy, student_model, save_model):
    """
    Saves the training results, including losses, accuracy, and the trained model, to the specified directory.

    :param dataset: str, the name of the dataset ("california_housing", "cifar10", "fashion_mnist").
    :param save_dir: str, the directory where the results should be saved.
    :param training_losses: list, the list of training losses recorded during model training.
    :param accuracy: float, the accuracy of the trained model.
    :param student_model: nn.Module, the trained student model to be saved.
    :param save_model: bool, whether to save the trained model to disk. If True, the model is saved
                       as `student_model.pth` in the `save_dir`.
    """
    decimal_places = 6 if dataset == "california_housing" else 2
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(f"{save_dir}/training_losses.txt", training_losses)
    np.savetxt(f"{save_dir}/accuracy.txt", [accuracy], fmt=f"%.{decimal_places}f")
    if save_model:
        torch.save(student_model, os.path.join(save_dir, f"student_model.pth"))

def train_student_distill_classification(train_loader, student_model, teacher_model, optimizer, criterion, switch_epoch, alpha):
    """
    Trains a classification model using reversed model distillation, combining both hard and soft losses
    during the training process.

    The training starts with distillation, where the student model learns from the teacher model (soft loss)
    until the switch_epoch. After that, the model trains using only the hard loss (classification error).

    :param train_loader: DataLoader, the data loader for the training dataset.
    :param student_model: nn.Module, the student model to be trained.
    :param teacher_model: nn.Module, the teacher model providing soft targets for distillation.
    :param optimizer: torch.optim.Optimizer, the optimizer used for updating the model parameters.
    :param criterion: nn.Module, the loss function used for computing the model's error.
    :param switch_epoch: int, epoch from which the training switches to using 100% hard loss (classification error).
    :param alpha: float, the weight parameter for the distillation loss. During distillation,
                  (100-alpha)% of the loss is soft (from teacher model), and alpha% is hard (from true labels).

    :return: list, the recorded training losses throughout the epochs.
    """
    epochs = config.EPOCHS
    student_model.train()
    teacher_model.eval()
    training_losses = []

    for epoch in range(epochs):
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            student_outputs = student_model(inputs)

            if epoch < switch_epoch:
                # Until switch_epoch use also teacher model for computing loss
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)

                loss_soft = F.kl_div(F.log_softmax(student_outputs, dim=1),
                                     F.softmax(teacher_outputs, dim=1),
                                     reduction="batchmean")
                loss_hard = criterion(student_outputs, targets)
                loss = alpha * loss_hard + (1 - alpha) * loss_soft
            else:
                loss = criterion(student_outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:  # Save every 100 mini-batches
                training_losses.append(running_loss / 100)
                running_loss = 0.0
    return training_losses

def train_student_distill_regression(train_data, student_model, teacher_model, optimizer, criterion, switch_epoch, alpha):
    """
    Trains a regression model using reversed model distillation, combining both hard and soft losses
    during the training process.

    The training starts with distillation, where the student model learns from the teacher model (soft loss)
    until the switch_epoch. After that, the model trains using only the hard loss (regression error).

    :param train_data: tuple, a tuple containing training data and labels ([X_train, y_train]).
    :param student_model: nn.Module, the student model to be trained.
    :param teacher_model: nn.Module, the teacher model providing soft targets for distillation.
    :param optimizer: torch.optim.Optimizer, the optimizer used for updating the model parameters.
    :param criterion: nn.Module, the loss function used for computing the model's error.
    :param switch_epoch: int, epoch from which the training switches to using 100% hard loss (regression error).
    :param alpha: float, the weight parameter for the distillation loss. During distillation,
                  (100-alpha)% of the loss is soft (from teacher model), and alpha% is hard (from true labels).

    :return: list, the recorded training losses throughout the epochs.
    """
    epochs = config.EPOCHS_REGRESSION
    X_train, y_train = train_data
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    student_model.train()
    teacher_model.eval()
    training_losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        student_outputs = student_model(X_train_tensor)
        if epoch < switch_epoch:
            with torch.no_grad():
                teacher_outputs = teacher_model(X_train_tensor)

            loss_soft = criterion(student_outputs, teacher_outputs)
            loss_hard = criterion(student_outputs, y_train_tensor)
            loss = alpha * loss_hard + (1 - alpha) * loss_soft
        else:
            loss = criterion(student_outputs, y_train_tensor)

        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())
    return training_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Student Model")
    parser.add_argument("--datasets-path", type=str, required=True, help="Path to the datasets folder")
    parser.add_argument("--teacher-path", type=str, required=True, help="Path to teacher model")
    parser.add_argument("--output", type=str, required=True, help="Path to the outputs folder")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training (default: 64)")
    parser.add_argument("--num-workers", type=int, default=10,
                        help="Number of worker threads for data loading (default: 10)")
    parser.add_argument("--seeds-file", type=str, required=True, help="Path to the seeds list txt file")
    parser.add_argument("--alpha", type=float, default=0.6, help="Alpha parameter (default: 0.6)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar10", "fashion_mnist", "california_housing"], help="Dataset")
    parser.add_argument("--save-model", action="store_true", help="Save trained models (.pth)")

    args = parser.parse_args()
    datasets_path = args.datasets_path
    outputs_path = args.output
    batch_size = args.batch_size
    num_workers = args.num_workers
    teacher_path = args.teacher_path
    seeds_file = args.seeds_file
    alpha = args.alpha
    dataset = args.dataset
    save_model = args.save_model

    device = torch.device("cuda" if config.USE_CUDA and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seeds = np.loadtxt(seeds_file, dtype=int).tolist()

    teacher_model = torch.load(teacher_path, weights_only=False)

for run, seed in enumerate(seeds):
    print(f"Starting run {run + 1}/{len(seeds)} (seed: {seed})...")
    results = {}
    switch_epoch_accuracies = []

    if dataset == "california_housing":
        epochs = config.EPOCHS_REGRESSION
        switch_epochs = 15
        step_size = epochs // switch_epochs
        for i in range(1, switch_epochs + 1):
            set_seed(seed)
            train_loader, test_loader = get_loaders(datasets_path=datasets_path, batch_size=batch_size,
                                                    num_workers=num_workers, dataset=dataset)
            switch_epoch = i * step_size
            print(f"Training Student with Distillation, switch_epoch = {switch_epoch}")
            student_model = get_student_model(dataset).to(device)
            student_optimizer = optim.Adam(student_model.parameters(), lr=config.LEARNING_RATE)
            criterion = nn.MSELoss()
            training_losses = train_student_distill_regression(train_loader, student_model, teacher_model, student_optimizer,
                                                    criterion, switch_epoch, alpha)
            accuracy = test_model_regression(student_model, test_loader, device)
            save_dir = f"{outputs_path}/seed_{seed}/switch_epoch_{switch_epoch}"
            save_results(dataset, save_dir, training_losses, accuracy, student_model, save_model)

            with open(os.path.join(save_dir, "fgsm_results.csv"), mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["epsilon", "accuracy"])

                for epsilon in [0, 0.005, 0.01, 0.05]:
                    print(f"Running FGSM test with epsilon: {epsilon}...")
                    fgsm_accuracy = test_model_fgsm_regression(student_model, test_loader, device, epsilon)
                    writer.writerow([epsilon, fgsm_accuracy])
            del train_loader, test_loader
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.1)
    else:
        switch_epochs = config.EPOCHS
        for switch_epoch in range(1, switch_epochs + 1):
            set_seed(seed)
            train_loader, test_loader = get_loaders(datasets_path=datasets_path, batch_size=batch_size,
                                                    num_workers=num_workers, dataset=dataset)
            print(f"Training Student with Distillation with switch_epoch = {switch_epoch}")

            student_model = get_student_model(dataset).to(device)
            student_optimizer = optim.Adam(student_model.parameters(), lr=config.LEARNING_RATE)

            criterion = nn.CrossEntropyLoss()
            training_losses = train_student_distill_classification(train_loader, student_model, teacher_model, student_optimizer, criterion, switch_epoch, alpha)
            accuracy = test_model_classification(student_model, test_loader, device)
            save_dir = f"{outputs_path}/seed_{seed}/switch_epoch_{switch_epoch}"
            save_results(dataset, save_dir, training_losses, accuracy, student_model, save_model)

            with open(os.path.join(save_dir, "fgsm_results.csv"), mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["epsilon", "accuracy"])

                for epsilon in [0, 0.005, 0.01, 0.05]:
                    print(f"Running FGSM test with epsilon: {epsilon}...")
                    fgsm_accuracy = test_model_fgsm_classification(student_model, test_loader, device, epsilon)
                    writer.writerow([epsilon, fgsm_accuracy])

            del train_loader, test_loader
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.1)
