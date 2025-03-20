from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import random
import numpy as np
import config


def get_cifar10_datasets(datasets_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(root=datasets_path, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=datasets_path, train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_cifar10_loaders(datasets_path, batch_size, num_workers):
    train_dataset, test_dataset = get_cifar10_datasets(datasets_path)

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=config.PIN_MEMORY, persistent_workers=config.PERSISTENT_WORKERS, worker_init_fn=seed_worker,generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=config.PIN_MEMORY, persistent_workers=config.PERSISTENT_WORKERS, worker_init_fn=seed_worker,generator=g)
    return train_loader, test_loader
