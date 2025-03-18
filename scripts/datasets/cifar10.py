from torchvision import datasets, transforms
from scripts.config import DATASETS_PATH


def get_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(root=DATASETS_PATH, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=DATASETS_PATH, train=False, download=True, transform=transform)

    return train_dataset, test_dataset
