import torch
from .visual import save_training_plot

# Definiamo DEVICE qui una volta per tutte
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
