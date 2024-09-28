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
