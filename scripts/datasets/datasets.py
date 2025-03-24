from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config


def get_cifar10_datasets(datasets_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(root=datasets_path, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=datasets_path, train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def get_mnist_datasets(datasets_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(root=datasets_path, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=datasets_path, train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def get_loaders(datasets_path, batch_size, num_workers, dataset):
    if dataset == "cifar10":
        train_dataset, test_dataset = get_cifar10_datasets(datasets_path)
    elif dataset == "fashion_mnist":
        train_dataset, test_dataset = get_mnist_datasets(datasets_path)
    else:
        raise ValueError("Unknown dataset!")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=config.PIN_MEMORY, persistent_workers=config.PERSISTENT_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=config.PIN_MEMORY, persistent_workers=config.PERSISTENT_WORKERS)
    return train_loader, test_loader
