import torch
import torch.nn as nn
import torch.nn.functional as F

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Accuracy: {accuracy}%')
    return accuracy

def test_model_regression(model, test_loader, device):
    X_test, y_test = test_loader
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        outputs = model(X_test_tensor)
        loss = criterion(outputs, y_test_tensor)
        total_loss += loss.item()

    print(f"Test Loss: {total_loss:.6f}")
    return total_loss

def test_model_fgsm(model, test_loader, device):
    model.eval()
    correct = 0
    correct_fgsm = 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

        mask = predicted.eq(targets)
        if mask.sum().item() == 0:
            continue

        inputs_filtered = inputs[mask]
        targets_filtered = targets[mask]

        data_fgsm = fgsm_attack(model, inputs_filtered, targets_filtered, 0.1)

        with torch.no_grad():
            outputs_fgsm = model(data_fgsm)
        _, predicted_fgsm = outputs_fgsm.max(1)

        correct_fgsm += predicted_fgsm.eq(targets_filtered).sum().item()


    accuracy = 100. * correct / len(test_loader.dataset)
    accuracy_fgsm = 100. * correct_fgsm / len(test_loader.dataset)
    print(f'Accuracy: {accuracy}%')
    print(f'Accuracy_fgsm: {accuracy_fgsm}%')
    return accuracy


def fgsm_attack(model, images, labels, epsilon):
    images = images.clone().detach().requires_grad_(True)

    # Forward pass
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)

    # Backward pass
    model.zero_grad()
    loss.backward()

    data_grad = images.grad.data
    perturbed_images = images + epsilon * data_grad.sign()

    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images