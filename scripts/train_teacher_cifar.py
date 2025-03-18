import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import config
import os
import models.teacher_cifar

from torch.utils.data import DataLoader
from datasets.cifar10 import get_cifar10
from test_model import test_model

device = torch.device("cuda" if config.USE_CUDA and torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

epochs = config.EPOCHS

num_teachers = 10

torch.backends.cudnn.benchmark = True

train_dataset, test_dataset = get_cifar10()
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS,
                          pin_memory=config.PIN_MEMORY, persistent_workers=config.PERSISTENT_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS,
                         pin_memory=config.PIN_MEMORY, persistent_workers=config.PERSISTENT_WORKERS)


def train_teacher(model, optimizer, criterion):
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


for i in range(num_teachers):
    teacher_model = models.teacher_cifar.TeacherModel1().to(device)  # Pick teacher model
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"Training Teacher Model {i + 1}...")
    training_losses = train_teacher(teacher_model, teacher_optimizer, criterion)
    print(f"Teacher Model {i + 1} testing:")
    accuracy = test_model(teacher_model, test_dataset, device)

    save_dir = f"models/teacher_model_cifar_{i + 1}"
    print(f"Saving Teacher Model {i + 1} to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt("training_losses.txt", training_losses)
    np.savetxt("accuracy.txt", accuracy)
    torch.save(teacher_model, os.path.join(save_dir, f"teacher_model.pth"))
