import argparse
import torch
import config
import numpy as np
from test_model import test_model_fgsm, test_model
from datasets.datasets import get_loaders

from set_seed import set_seed

parser = argparse.ArgumentParser(description="Run FGSM Attack")
parser.add_argument("--datasets-path", type=str, required=True, help="Path to the datasets folder")
parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
parser.add_argument("--output", type=str, required=True, help="Path to the outputs folder")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training (default: 64)")
parser.add_argument("--num-workers", type=int, default=10, help="Number of worker threads for data loading (default: 10)")
parser.add_argument("--seeds-file", type=str, required=True, help="Path to the seeds list txt file")
parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "fashion_mnist", "california_housing"], help="Dataset")

args = parser.parse_args()
datasets_path = args.datasets_path
outputs_path = args.output
batch_size = args.batch_size
num_workers = args.num_workers
model_path = args.model_path
seeds_file = args.seeds_file
dataset = args.dataset

device = torch.device("cuda" if config.USE_CUDA and torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

seeds = np.loadtxt(seeds_file, dtype=int).tolist()

model = torch.load(model_path, weights_only=False)

for run, seed in enumerate(seeds):
    print(f"Starting run {run + 1}/{len(seeds)} (seed: {seed})...")
    set_seed(seed)
    train_loader, test_loader = get_loaders(datasets_path=datasets_path, batch_size=batch_size, num_workers=num_workers,
                                            dataset=dataset)
    test_model(model,train_loader, device)
