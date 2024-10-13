import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence

torch.backends.cudnn.benchmark = True

# Nastavení zařízení
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Načtení datasetu IMDB
train_data, test_data = IMDB(root='.data')

# Použití základního tokenizeru
tokenizer = get_tokenizer('basic_english')

# Počítání výskytů slov a vytvoření slovníku
from collections import Counter

print(f"Number of training samples: {len(list(train_data))}")
print(f"Number of test samples: {len(list(test_data))}")
counter = Counter()
for _, line in train_data:
    counter.update(tokenizer(line))

# Přidáme speciální tokeny do slovníku
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common())}  # Posuneme o 2 kvůli speciálním tokenům
vocab[PAD_TOKEN] = 0  # Padding token bude na indexu 0
vocab[UNK_TOKEN] = 1  # Neznámý token bude na indexu 1


# Pipeline pro text a label
def text_pipeline(text):
    return [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenizer(text)]


label_pipeline = lambda label: 1 if label == 'pos' else 0


# Dataset třída
class IMDBDataset(Dataset):
    def __init__(self, data):
        self.data = list(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, text = self.data[idx]
        return label_pipeline(label), torch.tensor(text_pipeline(text), dtype=torch.long)


# Vytvoření datasetů pro trénování a testování
train_dataset = IMDBDataset(train_data)
test_dataset = IMDBDataset(test_data)


# Funkce pro dávkování a padding
def collate_batch(batch):
    labels, texts = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.float32)

    # Převádíme texty na tensory a přidáváme padding
    texts = [t.clone().detach().long() for t in texts]
    texts = pad_sequence(texts, batch_first=True, padding_value=vocab[PAD_TOKEN])  # Padding na hodnotu pro PAD_TOKEN
    return labels, texts


# DataLoadery pro trénink a testování
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_batch, pin_memory=True, num_workers=12)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_batch, pin_memory=True, num_workers=12)


# Definice RNN modelu
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=vocab[PAD_TOKEN])  # Přidáváme padding_idx
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        out = self.fc(rnn_out[:, -1, :])  # Používáme výstup z posledního časového kroku
        return out


# Parametry modelu
vocab_size = len(vocab)
embed_size = 128
hidden_size = 128
output_size = 1  # Protože máme binární klasifikaci

# Inicializace modelu
model = SimpleRNN(vocab_size, embed_size, hidden_size, output_size).to(device)

# Ztrátová funkce a optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trénovací smyčka
num_epochs = 2

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for labels, texts in train_loader:
        labels, texts = labels.to(device), texts.to(device)

        # Reset gradientů
        optimizer.zero_grad()

        # Předpověď
        outputs = model(texts)

        # Výpočet ztráty
        loss = criterion(outputs.squeeze(), labels)

        # Backpropagace a aktualizace váh
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

# Vyhodnocení modelu
model.eval()
correct, total = 0, 0
correctly_classified_samples = []
wrongly_classified_samples = []

with torch.no_grad():
    for labels, texts in test_loader:
        labels, texts = labels.to(device), texts.to(device)

        outputs = model(texts)
        predictions = torch.round(torch.sigmoid(outputs.squeeze()))
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Uložení správně klasifikovaných vzorků
        for i in range(len(predictions)):
            if predictions[i] == labels[i]:  # Kontrola, zda byla předpověď správná
                # Přidání textu a příslušného štítku do seznamu
                correctly_classified_samples.append((texts[i].cpu(), labels[i].cpu()))
            else:
                wrongly_classified_samples.append((texts[i].cpu(), labels[i].cpu()))


        print(f"correct = {correct}")
        print(f"total = {total}")
        print(f'Accuracy: {correct / total}')

print(f'Accuracy: {correct / total}')

i = 0
print("CORRECT :)")
for text_tensor, label_tensor in correctly_classified_samples:
    # Převod tokenů zpět na text
    if i <= 5:
        text = ' '.join([list(vocab.keys())[list(vocab.values()).index(token.item())] for token in text_tensor if token.item() not in [0, 1]])  # Ignorování PAD a UNK tokenů
        label = "Positive" if label_tensor.item() == 1 else "Negative"
        print(f'Label: {label} | Text: {text}')
        i = i + 1

i = 0
print("WRONG :(")
for text_tensor, label_tensor in wrongly_classified_samples:
    # Převod tokenů zpět na text
    if i <= 5:
        text = ' '.join([list(vocab.keys())[list(vocab.values()).index(token.item())] for token in text_tensor if token.item() not in [0, 1]])  # Ignorování PAD a UNK tokenů
        label = "Positive" if label_tensor.item() == 1 else "Negative"
        correct_label = "Negative" if label_tensor.item() == 1 else "Positive"
        print(f'Label was: {label} but should be {correct_label} | Text: {text}')
        i = i + 1
