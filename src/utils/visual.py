import matplotlib.pyplot as plt
import numpy as np
import os
from main_costants import SAVE_CHECKPOINT_EVERY

def save_training_plot(scores, filename="training_plot.png", window_size=SAVE_CHECKPOINT_EVERY):
    """
    Creates and saves a plot of the reward progression.
    
    Args:
        scores (list): The complete list of scores for each episode.
        filename (str): The path where the image will be saved.
        window_size (int): The window size for the moving average.
    """
    # Set the figure size
    plt.figure(figsize=(10, 6))
    
    
    # Calculation and plotting of moving average (Trend)
    # If we have enough data to calculate the moving average
    if len(scores) >= window_size:
        # Calculate moving average using convolution for efficiency
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        
        # The X axis for the moving average must be shifted to align properly
        # (it starts from the 'window_size' episode)
        x_axis = np.arange(window_size - 1, len(scores))
        
        plt.plot(x_axis, moving_avg, color='blue', linewidth=2, label=f'Media Mobile ({window_size} ep)')

    # Labels and Title
    plt.title('Andamento Addestramento DQN')
    plt.xlabel('Episodio')
    plt.ylabel('Reward Totale (Score)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the file
    plt.savefig(filename)
    print(f"Grafico salvato in: {filename}")
    
    # Close the figure to free up memory
    plt.close()