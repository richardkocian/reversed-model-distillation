DATASETS_PATH = "/scratch/xkocia19/datasets" # If the dataset is missing, it will be downloaded automatically.

BATCH_SIZE = 2048 # Reduce this on less powerful machines.
NUM_WORKERS = 16 # Adjust based on CPU capabilities.
LEARNING_RATE = 0.001
EPOCHS = 15

USE_CUDA = True  # Set to False to run on CPU.
PIN_MEMORY = True # Recommended when using a GPU.
PERSISTENT_WORKERS = True # Keeps DataLoader workers alive between epochs.
