import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import config
import os
import argparse

from datasets.cifar10 import get_cifar10_loaders
from set_seed import set_seed
from test_model import test_model
from models.teacher_cifar import TeacherModelMedium, TeacherModelTiny, TeacherModelLarge
from models.student_cifar import StudentModel


parser = argparse.ArgumentParser(description="Train Model Without Distillation")
parser.add_argument("--datasets-path", type=str, required=True, help="Path to the datasets folder")
parser.add_argument("--output", type=str, required=True, help="Path to the outputs folder")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training (default: 64)")
parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads for data loading (default: 4)")
parser.add_argument("--seeds-file", type=str, required=True, help="Path to the seeds list txt file")
parser.add_argument("--model", type=str, required=True, choices=["TeacherModelMedium", "TeacherModelSmall", "TeacherModelLarge", "StudentModel"], help="Model to train")

args = parser.parse_args()
datasets_path = args.datasets_path
outputs_path = args.output
batch_size = args.batch_size
num_workers = args.num_workers
seeds_file = args.seeds_file

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
    set_seed(seed)
    train_loader, test_loader = get_cifar10_loaders(datasets_path=datasets_path, batch_size=batch_size,
                                                    num_workers=num_workers)

    model_dict = {
        "TeacherModelMedium": TeacherModelMedium,
        "TeacherModelTiny": TeacherModelTiny,
        "TeacherModelLarge": TeacherModelLarge,
        "StudentModel": TeacherModelLarge
    }

    model_class = model_dict[args.model]
    model = model_class().to(device)
    
    student_optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"Training Model {run + 1}/{len(seeds)} (seed: {seed})...")
    training_losses = train_model(train_loader, model, student_optimizer, criterion)
    accuracy = test_model(model, test_loader, device)

    save_dir = f"{outputs_path}/{args.model}_cifar_seed_{seed}"
    print(f"Saving Model to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(f"{save_dir}/training_losses.txt", training_losses)
    np.savetxt(f"{save_dir}/accuracy.txt", [accuracy], fmt="%.2f")
