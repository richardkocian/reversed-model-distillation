# --------------------------------------------
# File: set_seed.py
# Description: Function for setting random seeds, ensuring reproducibility
#              of experiments in PyTorch and NumPy.
# Author: Richard Kocián
# Created: 18.03.2025
# --------------------------------------------

import numpy as np
import torch
import random

def set_seed(seed):
    """
    Set the random seed for NumPy, Python's random module, and PyTorch to ensure reproducibility.

    :param seed: The seed value to set for the random number generators.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    print(f"Seed {seed} has been set!")
