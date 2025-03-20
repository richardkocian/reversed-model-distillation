import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import config
import os
import models.student_cifar
import models.teacher_cifar
import argparse
import torch.nn.functional as F

from datasets.cifar10 import get_cifar10_loaders
from set_seed import set_seed
from test_model import test_model


parser = argparse.ArgumentParser(description='Train Student Model')
parser.add_argument('--datasets-path', type=str, required=True, help='Path to the datasets folder')
parser.add_argument('--teacher-path', type=str, required=True, help='Path to teacher model')
parser.add_argument('--output', type=str, required=True, help='Path to the outputs folder')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training (default: 64)')
parser.add_argument('--num-workers', type=int, default=4, help='Number of worker threads for data loading (default: 4)')
parser.add_argument('--seeds-file', type=str, required=True, help='Path to the seeds list txt file')
parser.add_argument('--alpha', type=float, default=0.6, help='Alpha parameter (default: 0.6)')

args = parser.parse_args()
datasets_path = args.datasets_path
outputs_path = args.output
batch_size = args.batch_size
num_workers = args.num_workers
teacher_path = args.teacher_path
seeds_file = args.seeds_file
alpha = args.alpha



seeds = np.loadtxt(seeds_file, dtype=int).tolist()
epochs = config.EPOCHS

teacher_model = torch.load(teacher_path, weights_only=False)

def train_student_distill(train_loader, student_model, teacher_model, optimizer, criterion, switch_epoch, alpha):
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
                                     reduction='batchmean')
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

for run, seed in enumerate(seeds):
    print(f"Starting run {run + 1}/{len(seeds)} (seed: {seed})...")
    results = {}
    switch_epoch_accuracies = []

    set_seed(seed)
    device = torch.device("cuda" if config.USE_CUDA and torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    train_loader, test_loader = get_cifar10_loaders(datasets_path=datasets_path,batch_size=batch_size,num_workers=num_workers)

    for switch_epoch in range(epochs):
        print(f"Training Student with Distillation with switch_epoch = {switch_epoch+1}")

        student_model = models.student_cifar.StudentModelCIFAR().to(device)  # Pick student model
        student_optimizer = optim.Adam(student_model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        training_losses = train_student_distill(train_loader, student_model, teacher_model, student_optimizer, criterion, switch_epoch+1, alpha)
        accuracy = test_model(student_model, test_loader, device)

        save_dir = f"{outputs_path}/seed_{seed}/switch_epoch_{switch_epoch+1}"
        os.makedirs(save_dir, exist_ok=True)
        np.savetxt(f"{save_dir}/training_losses.txt", training_losses)
        np.savetxt(f"{save_dir}/accuracy.txt", [accuracy], fmt="%.2f")
