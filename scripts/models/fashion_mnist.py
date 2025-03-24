import torch.nn as nn
import torch.nn.functional as F

class StudentModelFashionMNIST(nn.Module):
    def __init__(self, input_size=28):  # input_size=32 for CIFAR-10
        super(StudentModelFashionMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        fc_input_dim = (input_size // 4) * (input_size // 4) * 128

        self.fc1 = nn.Linear(fc_input_dim, 256)
        self.fc2 = nn.Linear(256, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class TeacherModelSmallFashionMNIST(nn.Module):
    def __init__(self, input_size=28):
        super(TeacherModelSmallFashionMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        fc_input_dim = (input_size // 2) * (input_size // 2) * 2

        self.fc1 = nn.Linear(fc_input_dim, 16)
        self.fc2 = nn.Linear(16, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class TeacherModelMediumFashionMNIST(nn.Module):
    def __init__(self, input_size=28):
        super(TeacherModelMediumFashionMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        fc_input_dim = (input_size // 2) * (input_size // 2) * 4

        self.fc1 = nn.Linear(fc_input_dim, 32)
        self.fc2 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class TeacherModelLargeFashionMNIST(nn.Module):
    def __init__(self, input_size=28):
        super(TeacherModelLargeFashionMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3, padding=1)  # Přidaná vrstva
        self.pool = nn.MaxPool2d(2, 2)

        fc_input_dim = (input_size // 2) * (input_size // 2) * 16

        self.fc1 = nn.Linear(fc_input_dim, 64)  # Větší hidden layer
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
