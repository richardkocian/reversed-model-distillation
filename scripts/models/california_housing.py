import torch.nn as nn
import torch.nn.functional as F

class StudentModelCALIFORNIA(nn.Module):
    def __init__(self):
        super(StudentModelCALIFORNIA, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class TeacherModelMediumCALIFORNIA(nn.Module):
    def __init__(self):
        super(TeacherModelMediumCALIFORNIA, self).__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
