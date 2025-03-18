from torchvision import datasets, transforms


def get_cifar10(datasets_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(root=datasets_path, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=datasets_path, train=False, download=True, transform=transform)

    return train_dataset, test_dataset
