# --------------------------------------------
# File: config.py
# Description: Configuration settings for training models,
#              including hyperparameters and GPU settings.
# Author: Richard Kocián
# Created: 18.03.2025
# --------------------------------------------

LEARNING_RATE = 0.001
EPOCHS = 15
EPOCHS_REGRESSION = 1500

USE_CUDA = True  # Set to False to run on CPU.
PIN_MEMORY = True # Recommended when using a GPU.
PERSISTENT_WORKERS = False
