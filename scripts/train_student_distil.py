import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import config
import os
import argparse
import torch.nn.functional as F

from datasets.datasets import get_loaders
from set_seed import set_seed
from test_model import test_model, test_model_regression
from models.cifar import StudentModelCIFAR
from models.fashion_mnist import StudentModelFashionMNIST
from models.california_housing import StudentModelCALIFORNIA

def get_student_model(dataset):
    if dataset == "cifar10":
        return StudentModelCIFAR()
    elif dataset == "fashion_mnist":
        return StudentModelFashionMNIST()
    elif dataset == "california_housing":
        return StudentModelCALIFORNIA()
    else:
        raise ValueError("Unknown combination of dataset and model")

def save_results(dataset, seed, outputs_path, switch_epoch, training_losses, accuracy, student_model):
    decimal_places = 6 if dataset == "california_housing" else 2
    save_dir = f"{outputs_path}/seed_{seed}/switch_epoch_{switch_epoch}"
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(f"{save_dir}/training_losses.txt", training_losses)
    np.savetxt(f"{save_dir}/accuracy.txt", [accuracy], fmt=f"%.{decimal_places}f")
    if args.save_model:
        torch.save(student_model, os.path.join(save_dir, f"student_model.pth"))

parser = argparse.ArgumentParser(description="Train Student Model")
parser.add_argument("--datasets-path", type=str, required=True, help="Path to the datasets folder")
parser.add_argument("--teacher-path", type=str, required=True, help="Path to teacher model")
parser.add_argument("--output", type=str, required=True, help="Path to the outputs folder")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training (default: 64)")
parser.add_argument("--num-workers", type=int, default=10, help="Number of worker threads for data loading (default: 10)")
parser.add_argument("--seeds-file", type=str, required=True, help="Path to the seeds list txt file")
parser.add_argument("--alpha", type=float, default=0.6, help="Alpha parameter (default: 0.6)")
parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "fashion_mnist", "california_housing"], help="Dataset")
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

device = torch.device("cuda" if config.USE_CUDA and torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

seeds = np.loadtxt(seeds_file, dtype=int).tolist()

teacher_model = torch.load(teacher_path, weights_only=False)

def train_student_distill(train_loader, student_model, teacher_model, optimizer, criterion, switch_epoch, alpha):
    epochs = config.EPOCHS
    student_model.train()
    teacher_model.eval()
    training_losses = []

    for epoch in range(epochs):
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Student outputs
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
def train_student_distill_regression(train_loader, student_model, teacher_model, optimizer, criterion, switch_epoch, alpha):
    epochs = config.EPOCHS_REGRESSION
    X_train, y_train = train_loader
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

            save_results(dataset, seed, outputs_path, switch_epoch, training_losses, accuracy, student_model)
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
            training_losses = train_student_distill(train_loader, student_model, teacher_model, student_optimizer, criterion, switch_epoch, alpha)
            accuracy = test_model(student_model, test_loader, device)

            save_results(dataset, seed, outputs_path, switch_epoch, training_losses, accuracy, student_model)
