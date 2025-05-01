import argparse
import torch
import config
from test_model import test_model_fgsm_classification, test_model_fgsm_regression
from datasets.datasets import get_loaders
import os
import csv

parser = argparse.ArgumentParser(description="Run FGSM Attack")
parser.add_argument("--datasets-path", type=str, required=True, help="Path to the datasets folder")
parser.add_argument("--models-path", type=str, required=True, help="Path to trained models")
parser.add_argument("--output", type=str, required=True, help="Path to the outputs folder")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training (default: 64)")
parser.add_argument("--num-workers", type=int, default=10, help="Number of worker threads for data loading (default: 10)")
parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "fashion_mnist", "california_housing"], help="Dataset")

args = parser.parse_args()
datasets_path = args.datasets_path
outputs_path = args.output
batch_size = args.batch_size
num_workers = args.num_workers
modesl_path = args.models_path
dataset = args.dataset

device = torch.device("cuda" if config.USE_CUDA and torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

train_loader, test_loader = get_loaders(datasets_path=datasets_path, batch_size=batch_size, num_workers=num_workers,
                                            dataset=dataset)

epsilons = [0, 0.005, 0.01, 0.05]

for dirpath, dirnames, filenames in os.walk(modesl_path):
    for filename in filenames:
        if filename.endswith(".pth"):
            model_path = os.path.join(dirpath, filename)
            model = torch.load(str(model_path), weights_only=False)

            csv_path = os.path.join(dirpath, "fgsm_results.csv")
            print(f"Saving results to: {csv_path}...")

            with open(csv_path, mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["epsilon", "accuracy"])

                for epsilon in epsilons:
                    print(f"Running FGSM test with epsilon: {epsilon}...")
                    if dataset == "california_housing":
                        fgsm_accuracy = test_model_fgsm_regression(model, test_loader, device, epsilon)
                    else:
                        fgsm_accuracy = test_model_fgsm_classification(model, test_loader, device, epsilon)
                    writer.writerow([epsilon, fgsm_accuracy])
