import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import config
import os
import models.student_cifar
import argparse

from torch.utils.data import DataLoader
from datasets.cifar10 import get_cifar10
from test_model import test_model

parser = argparse.ArgumentParser(description='Train Student Model Without Distillation')
parser.add_argument('--datasets-path', type=str, required=True, help='Path to the datasets folder')
parser.add_argument('--output', type=str, required=True, help='Path to the outputs folder')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training (default: 64)')
parser.add_argument('--num-workers', type=int, default=4, help='Number of worker threads for data loading (default: 4)')

num_students = 15

args = parser.parse_args()
datasets_path = args.datasets_path
outputs_path = args.output
batch_size = args.batch_size
num_workers = args.num_workers

device = torch.device("cuda" if config.USE_CUDA and torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

epochs = config.EPOCHS

torch.backends.cudnn.benchmark = True

train_dataset, test_dataset = get_cifar10(datasets_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=config.PIN_MEMORY, persistent_workers=config.PERSISTENT_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                         pin_memory=config.PIN_MEMORY, persistent_workers=config.PERSISTENT_WORKERS)


def train_student(model, optimizer, criterion):
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


for i in range(num_students):
    student_model = models.student_cifar.StudentModelCIFAR().to(device)  # Pick teacher model
    teacher_optimizer = optim.Adam(student_model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"Training Student Model {i + 1}...")
    training_losses = train_student(student_model, teacher_optimizer, criterion)
    print(f"Teacher Model {i + 1} testing:")
    accuracy = test_model(student_model, test_loader, device)

    save_dir = f"{outputs_path}/models/student_model_cifar_{i + 1}"
    print(f"Saving Student Model {i + 1} to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(f"{save_dir}/training_losses.txt", training_losses)
    np.savetxt(f"{save_dir}/accuracy.txt", [accuracy], fmt="%.2f")
    torch.save(student_model, os.path.join(save_dir, f"student_model.pth"))
