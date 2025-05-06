# --------------------------------------------
# File: testing_model.py
# Description: Functions for evaluating models, including accuracy and loss under FGSM attack.
# Author: Richard Kocián
# Created: 18.03.2025
# --------------------------------------------

import torch
import torch.nn as nn

def test_model_classification(model, test_loader, device):
    """
    Evaluate the classification accuracy of a trained model on the test dataset.

    This function sets the model to evaluation mode, iterates over the test dataset,
    performs inference on the input data, and computes the accuracy by comparing
    the model's predictions with the true labels.

    :param model: The trained model to be evaluated.
    :param test_loader: DataLoader for the test dataset.
    :param device: The device (CPU or GPU) on which the model and data are loaded.

    :return: The accuracy of the model on the test dataset as a percentage.
    """
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

def test_model_regression(model, test_data, device):
    """
    Evaluate the regression model on the test dataset by computing the mean squared error (MSE) loss.

    This function sets the model to evaluation mode, performs inference on the test dataset,
    and computes the loss by comparing the model's predictions with the true values (targets).

    :param model: The trained regression model to be evaluated.
    :param test_data: tuple, a tuple containing testing data and labels ([X_train, y_train]).
    :param device: The device (CPU or GPU) on which the model and data are loaded.

    :return: The total loss (MSE) on the test dataset.
    """
    X_test, y_test = test_data
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
    """
    Evaluate the classification accuracy of a model on the test dataset under FGSM attack.

    This function sets the model to evaluation mode, iterates over the test dataset,
    performs inference on the input data, and computes the accuracy both on clean inputs
    and adversarial examples generated using the Fast Gradient Sign Method (FGSM).

    :param model: The trained model to be evaluated.
    :param test_loader: DataLoader for the test dataset.
    :param device: The device (CPU or GPU) on which the model and data are loaded.
    :param epsilon: The perturbation magnitude for the FGSM attack.

    :return: The accuracy of the model on adversarial examples as a percentage.
    """
    model.eval()
    correct = 0
    correct_fgsm = 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

        # Perform FGSM attack only on the test images that was correctly classified:
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

def test_model_fgsm_regression(model, test_data, device, epsilon):
    """
    Evaluate the regression model on the test dataset under FGSM attack by computing the mean squared error (MSE) loss.

    This function sets the model to evaluation mode, generates adversarial examples using the Fast Gradient Sign Method (FGSM),
    and computes the loss both on clean inputs and adversarially perturbed inputs.

    :param model: The trained regression model to be evaluated.
    :param test_data: tuple, a tuple containing testing data and labels ([X_train, y_train]).
    :param device: The device (CPU or GPU) on which the model and data are loaded.
    :param epsilon: The perturbation magnitude for the FGSM attack.

    :return: The total MSE loss (both clean and FGSM) on the test dataset.
    """
    X_test, y_test = test_data
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
    """
    Generate adversarial examples using the Fast Gradient Sign Method (FGSM).

    This function computes the gradient of the loss with respect to the input data
    and generates adversarial perturbations by adding the sign of the gradient to the input.

    :param model: The trained model used to compute gradients.
    :param test_data: The input data to be perturbed.
    :param labels: The true labels for the input data.
    :param epsilon: The magnitude of the perturbation to be added.
    :param criterion: The loss function used for computing gradients (typically MSE or CrossEntropy).

    :return: The perturbed data, adversarial examples generated using FGSM.
    """
    test_data = test_data.clone().detach().requires_grad_(True)

    outputs = model(test_data)
    loss = criterion(outputs, labels)

    model.zero_grad()
    loss.backward()

    data_grad = test_data.grad.data
    perturbed_data = test_data + epsilon * data_grad.sign()

    return perturbed_data
