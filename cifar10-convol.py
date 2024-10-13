import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# TODO try with mean = (0.4914, 0.4822, 0.4465) and std = (0.2023, 0.1994, 0.2010)

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, num_workers=12, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, num_workers=12, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 3x32x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1) # in_channels=RGB, kernel_size = size of filters
        # 16x32x32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 16x16x16
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 32x16x16
        # pool --
        # 32x8x8
        self.fc1 = nn.Linear(in_features=32 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# TODO momentum=0.9 znamená, že se při aktualizaci váh bude brát v úvahu z 90 % směr předchozí aktualizace. To může pomoci váhy posouvat správným směrem rychleji a bez zbytečných oscilací.

epochs = 20
print(next(model.parameters()).device)
for epoch in range(epochs):
    running_loss = 0.0
    print(f"[{epoch + 1}] ztráta: {running_loss / 500:.3f}")
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # Získání vstupů; data je seznam [vstupy, štítky]
        images, labels = data

        images, labels = images.to(device), labels.to(device)

        # Nulování gradientů
        optimizer.zero_grad()

        # Forward průchod
        outputs = model(images)

        # Výpočet ztráty
        loss = criterion(outputs, labels)

        # Backward průchod (výpočet gradientů)
        loss.backward()

        # Aktualizace vah
        optimizer.step()

        # Tisknutí statistiky o ztrátě
        running_loss += loss.item()
        # if i % 50 == 49:  # Každých 500 batchů
        #     print(f"[{epoch + 1}, {i + 1}] ztráta: {running_loss / 500:.3f}")
        #     running_loss = 0.0

print('Trénování dokončeno')

model.eval()
correct = 0
total = 0
count = 10
with torch.no_grad():  # Deaktivujeme výpočet gradientů, abychom šetřili paměť
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        for i in range(len(predicted)):
            if count == 0:
                break
            if predicted[i] != labels[i]:
                plt.imshow(images[i].cpu().numpy().transpose((1, 2, 0)) / 2 + 0.5)
                plt.title(f'Predicted: {classes[predicted[i].item()]} ({predicted[i].item()}), Actual: {classes[labels[i].item()]} ({labels[i].item()})')
                plt.axis('off')  # Skrytí os
                plt.show()
                count -= 1
                break
        correct += (predicted == labels).sum().item()

print(f"Správnost na testovacích datech: {100 * correct / total:.2f}%")

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print(f'Správnost u třídy {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
