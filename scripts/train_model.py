import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import config
import os
import argparse

from datasets.datasets import get_loaders
from test_model import test_model_california
from set_seed import set_seed
from test_model import test_model
from models.cifar import TeacherModelSmallCIFAR, TeacherModelMediumCIFAR, TeacherModelLargeCIFAR, StudentModelCIFAR
from models.fashion_mnist import TeacherModelSmallFashionMNIST, TeacherModelMediumFashionMNIST, TeacherModelLargeFashionMNIST, StudentModelFashionMNIST
from models.california_housing import TeacherModelMediumCALIFORNIA, StudentModelCALIFORNIA

def get_teacher_model(dataset, model_type):
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
            raise ValueError("Unknown combination of dataset and model")
        elif model_type == "TeacherModelMedium":
            return TeacherModelMediumCALIFORNIA()
        elif model_type == "TeacherModelLarge":
            raise ValueError("Unknown combination of dataset and model")
        elif model_type == "StudentModel":
            return StudentModelCALIFORNIA()
    else:
        raise ValueError("Unknown combination of dataset and model")

parser = argparse.ArgumentParser(description="Train Model Without Distillation")
parser.add_argument("--datasets-path", type=str, required=True, help="Path to the datasets folder")
parser.add_argument("--output", type=str, required=True, help="Path to the outputs folder")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training (default: 64)")
parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads for data loading (default: 4)")
parser.add_argument("--seeds-file", type=str, required=True, help="Path to the seeds list txt file")
parser.add_argument("--model", type=str, required=True, choices=["TeacherModelSmall", "TeacherModelMedium", "TeacherModelLarge", "StudentModel"], help="Model to train")
parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "fashion_mnist", "california_housing"], help="Dataset")

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

epochs = config.EPOCHS

def train_model(train_loader, model, optimizer, criterion):
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


for run, seed in enumerate(seeds):
    print(f"Training Model {run + 1}/{len(seeds)} (seed: {seed})...")

    set_seed(seed)
    train_loader, test_loader = get_loaders(datasets_path=datasets_path, batch_size=batch_size,
                                                    num_workers=num_workers, dataset=dataset)

    model = get_teacher_model(dataset, model_type).to(device)
    print(model)
    student_optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    if dataset == "california_housing":
        criterion = nn.MSELoss()
        training_losses = train_model(train_loader, model, student_optimizer, criterion)
        accuracy = test_model_california(model, test_loader, device)
    else:
        criterion = nn.CrossEntropyLoss()
        training_losses = train_model(train_loader, model, student_optimizer, criterion)
        accuracy = test_model(model, test_loader, device)

    save_dir = f"{outputs_path}/model_seed_{seed}"
    print(f"Saving Model to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(f"{save_dir}/training_losses.txt", training_losses)
    np.savetxt(f"{save_dir}/accuracy.txt", [accuracy], fmt="%.2f")
    torch.save(model, os.path.join(save_dir, f"teacher_model.pth"))
