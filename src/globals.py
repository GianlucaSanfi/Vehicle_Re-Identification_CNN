
import torch

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image size (in dataset)
IMG_HEIGHT = 256
IMG_WIDTH = 128

# Training hyperparameters
BATCH_P = 16     # identities per batch
BATCH_K = 4      # images per identity
FEAT_DIM = 512   # feature embedding size
EPOCHS = 20
LR = 3e-4
MARGIN = 0.3     # triplet margin
