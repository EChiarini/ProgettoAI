"""
Global utilities and hardware configuration for the Reinforcement Learning project.
"""
import torch
from .visual import save_training_plot

# Define the hardware DEVICE here globally for the entire project
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")