import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), 'src'))

from env.track_env import TrackEnv
from env.track_costants import *

def visualize():
    env = TrackEnv(render_mode=None)
    matrix = env.matrix
    checkpoints = env._checkpoints
    
    print(f"Trovati {len(checkpoints)} checkpoint.")
    
    plt.figure(figsize=(12, 12))

    cmap = sns.color_palette("Greys", as_cmap=True)
    sns.heatmap(matrix, cmap=cmap, cbar=False, square=True)
    

    colors = sns.color_palette("hsv", len(checkpoints))
        
    for i, (name, coords) in enumerate(checkpoints.items()):
        coords = np.array(coords)
        if len(coords) > 0:

            rows = coords[:, 0]
            cols = coords[:, 1]
            
            plt.scatter(cols + 0.5, rows + 0.5, 
                        color=colors[i], 
                        label=name, 
                        s=50, 
                        edgecolor='black',
                        alpha=0.8)

            center_x = np.mean(cols) + 0.5
            center_y = np.mean(rows) + 0.5
            plt.text(center_x, center_y, str(i+1), 
                     color='white', 
                     fontweight='bold', 
                     ha='center', va='center',
                     path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    plt.title("Track Layout & Checkpoints")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    output_path = Path("results/checkpoints_map.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Immagine salvata in: {output_path}")

if __name__ == "__main__":
    visualize()
