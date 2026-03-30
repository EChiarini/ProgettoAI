from pathlib import Path

# Number of episodes for the training process. 
DEFAULT_TRAIN_EPISODES = 100000

# Time delay (in seconds) between frames during the testing/inference phase 
# to make the agent's behavior human-observable.
DEFAULT_TEST_DELAY = 0.1

# Maximum number of steps allowed per episode. 
# This prevents infinite loops and limits the duration of suboptimal trajectories.
STEP_LIMIT = 1000

# Size of the rolling window used to calculate the moving average of rewards.
SCORES_WINDOW_SIZE = 1000

# Frequency (in episodes) at which the model's weights are saved.
SAVE_CHECKPOINT_EVERY = 1000

# Default filenames for the optimized model, performance plots, and trajectory analysis.
DEFAULT_MODEL_FILENAME = "best_model.pth"
DEFAULT_GRAPH_FILENAME = "grafico_finale.png"
REPORT_FILENAME = "heatmap_finale"

# Toggle for automated PDF report generation upon training completion.
GEN_REPORT = True

# Training Mode selector:
# 0: Standard training (from scratch).
# 1: Fine-tuning/Transfer Learning (loads pre-existing weights).
TRAINING_MODE = 0

# Directory path management 
CHECKPOINTS_PATH = Path("models")   # Storage for neural network weight files (.pth)
RESULTS_PATH = Path("results")      # Storage for logs, plots, and heatmaps