from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
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

def get_california_housing_datasets():
    data = fetch_california_housing()
    X, y = data.data, data.target.reshape(-1, 1)

    # Rozdělení datasetu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizace dat
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    # Převod na PyTorch tensory
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # DataLoadery
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    return train_dataset, test_dataset

def get_loaders(datasets_path, batch_size, num_workers, dataset):
    if dataset == "cifar10":
        train_dataset, test_dataset = get_cifar10_datasets(datasets_path)
    elif dataset == "fashion_mnist":
        train_dataset, test_dataset = get_mnist_datasets(datasets_path)
    elif dataset == "california_housing":
        train_dataset, test_dataset = get_california_housing_datasets()
    else:
        raise ValueError("Unknown dataset!")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=config.PIN_MEMORY, persistent_workers=config.PERSISTENT_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=config.PIN_MEMORY, persistent_workers=config.PERSISTENT_WORKERS)
    return train_loader, test_loader
