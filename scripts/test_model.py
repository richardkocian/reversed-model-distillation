import torch
import torch.nn as nn

def test_model_classification(model, test_loader, device):
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

def test_model_fgsm_classification(model, test_loader, device, epsilon):
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

        criterion = nn.CrossEntropyLoss()
        data_fgsm = get_fgsm_data(model, inputs_filtered, targets_filtered, epsilon, criterion)
        data_fgsm = torch.clamp(data_fgsm, -1, 1)

        with torch.no_grad():
            outputs_fgsm = model(data_fgsm)
        _, predicted_fgsm = outputs_fgsm.max(1)

        correct_fgsm += predicted_fgsm.eq(targets_filtered).sum().item()

    accuracy_fgsm = 100. * correct_fgsm / len(test_loader.dataset)
    print(f'Accuracy FGSM: {accuracy_fgsm}%')
    return accuracy_fgsm

def test_model_fgsm_regression(model, test_loader, device, epsilon):
    X_test, y_test = test_loader
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    model.eval()
    total_loss = 0.0
    total_loss_fgsm = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        outputs = model(X_test_tensor)
        loss = criterion(outputs, y_test_tensor)
        total_loss += loss.item()

    criterion = nn.MSELoss()
    data_fgsm = get_fgsm_data(model, X_test_tensor, y_test_tensor, epsilon, criterion)
    data_fgsm = torch.clamp(data_fgsm, X_test_tensor.min(), X_test_tensor.max())

    with torch.no_grad():
        outputs_fgsm = model(data_fgsm)
        loss_fgsm = criterion(outputs_fgsm, y_test_tensor)
        total_loss_fgsm += loss_fgsm.item()

    print(f"FGSM Loss: {total_loss_fgsm}")
    return total_loss_fgsm


def get_fgsm_data(model, test_data, labels, epsilon, criterion):
    test_data = test_data.clone().detach().requires_grad_(True)

    outputs = model(test_data)
    loss = criterion(outputs, labels)

    model.zero_grad()
    loss.backward()

    data_grad = test_data.grad.data
    perturbed_data = test_data + epsilon * data_grad.sign()

    return perturbed_data
