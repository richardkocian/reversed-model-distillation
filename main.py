import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Definice transformací pro normalizaci obrázků
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# Pokud bych věděl, že je v datasetu více bílých pixelů než černých, tak je vhodně zvolit průměr a odchylku na menší číslo?
# Není výhodné projít celý dataset a podle toho nastavit hodnotu průměru a odchylky?

# Načtení trénovacích a testovacích dat
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

image, label = train_dataset[0]
# Zobrazení obrázku pomocí matplotlib
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f'Label: {label}')
plt.show()

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Vstupní vrstva s 128 neurony
        self.fc2 = nn.Linear(128, 64)       # Skrytá vrstva s 64 neurony
        self.fc3 = nn.Linear(64, 10)        # Výstupní vrstva s 10 neurony (pro každou číslici)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Převedení 28x28 obrázku na 1D tensor
        x = F.relu(self.fc1(x))  # Aktivace ReLU pro první vrstvu
        x = F.relu(self.fc2(x))  # Aktivace ReLU pro druhou vrstvu
        x = self.fc3(x)          # Výstupní vrstva (bez aktivace, používá se softmax níže)
        return x

# Inicializace modelu
model = SimpleNN()

criterion = nn.CrossEntropyLoss() # softmax
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
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

        # Sledování ztráty
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

correct = 0
total = 0

model.eval()
# Deaktivace gradientů pro testování (abychom šetřili paměť)
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        # for i in range(len(predicted)):
        #     if predicted[i] != labels[i]:
        #         # Pokud předpověď není shodná se skutečným labelem, zobraz obrázek
        #         plt.imshow(images[i].squeeze(), cmap='gray')
        #         plt.title(f'Predicted: {predicted[i].item()}, Actual: {labels[i].item()}')
        #         plt.show()
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")


