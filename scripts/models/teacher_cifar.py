import torch.nn as nn
import torch.nn.functional as F

class TeacherModel1(nn.Module):
    def __init__(self, input_size=32): # input_size = 32 for CIFAR-10
        super(TeacherModel1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        fc_input_dim = (input_size // 2) * (input_size // 2) * 16

        self.fc1 = nn.Linear(fc_input_dim, 64)
        self.fc2 = nn.Linear(64, 10)  # 10 classes for CIFAR-10
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class TeacherModelTiny(nn.Module):
    def __init__(self, input_size=32):
        super(TeacherModelTiny, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)  # Jen 4 filtry
        self.pool = nn.MaxPool2d(2, 2)

        fc_input_dim = (input_size // 2) * (input_size // 2) * 4

        self.fc1 = nn.Linear(fc_input_dim, 16)  # Extrémně malá hidden layer
        self.fc2 = nn.Linear(16, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class TeacherModelLarge(nn.Module):
    def __init__(self, input_size=32):
        super(TeacherModelLarge, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Přidaná vrstva
        self.pool = nn.MaxPool2d(2, 2)

        fc_input_dim = (input_size // 2) * (input_size // 2) * 32

        self.fc1 = nn.Linear(fc_input_dim, 128)  # Větší hidden layer
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)