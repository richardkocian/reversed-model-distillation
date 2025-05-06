# --------------------------------------------
# File: datasets.py
# Description: Functions for loading and preprocessing datasets,
#              including CIFAR-10, Fashion-MNIST, and California Housing.
#              Provides DataLoader objects for training and testing.
# Author: Richard Kocián
# Created: 18.03.2025
# --------------------------------------------

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from configs import config


def get_cifar10_datasets(datasets_path):
    """
    Loads the CIFAR-10 training and test datasets with preprocessing.

    :param datasets_path: Path to the directory where datasets will be downloaded or are already stored.
    :return: A tuple (train_dataset, test_dataset) of torchvision.datasets.CIFAR10 instances.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(root=datasets_path, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=datasets_path, train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def get_mnist_datasets(datasets_path):
    """
    Loads the Fashion-MNIST training and test datasets with preprocessing.

    :param datasets_path: Path to the directory where datasets will be downloaded or are already stored.
    :return: A tuple (train_dataset, test_dataset) of torchvision.datasets.FashionMNIST instances.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(root=datasets_path, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=datasets_path, train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def get_california_housing_datasets():
    """
    Loads and preprocesses the California Housing dataset.

    :return: A tuple of two lists:
        - The first list contains the preprocessed training data: [X_train, y_train].
        - The second list contains the preprocessed testing data: [X_test, y_test].
    """
    data = fetch_california_housing()
    X, y = data.data, data.target.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    return [X_train, y_train], [X_test, y_test]

def get_loaders(datasets_path, batch_size, num_workers, dataset):
    """
    Returns data loaders for classification datasets or numpy arrays for regression datasets.

    For classification tasks, it returns DataLoader instances for both training and test datasets.
    For regression tasks (e.g., California Housing), it returns numpy arrays containing training
    and test data (features and targets) as tuples.

    :param datasets_path: str, path to the dataset.
    :param batch_size: int, number of samples per batch.
    :param num_workers: int, number of subprocesses to use for data loading.
    :param dataset: str, name of the dataset ("california_housing", "cifar10", "fashion_mnist").

    :return:
        - For classification datasets (CIFAR-10, Fashion MNIST), returns a tuple of
          DataLoader instances (train_loader, test_loader).
        - For regression datasets (California Housing), returns a tuple of numpy arrays
          ([X_train, y_train], [X_test, y_test]).
    """
    if dataset == "california_housing":
        train_loader, test_loader = get_california_housing_datasets()
        return train_loader, test_loader

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
