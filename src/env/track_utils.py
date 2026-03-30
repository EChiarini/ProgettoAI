import pandas as pd
import numpy as np
import copy
import os
from pathlib import Path
from .track_costants import get_default_track_path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import copy

def argwhere(matrix, value):
  """
  Searches for a specific value in a matrix and returns a list of coordinates 
  [row, column] for all occurrences.
  """
  l = list()
  x_max,y_max=matrix.shape
  for x in range(x_max):
    for y in range(y_max):
    #  print(f"CHHHHHHHH{matrix[x,y]}")
      if matrix[x,y] == value:
        l.append([x,y])

  return l

def build_track(fileName = get_default_track_path()):
    """Reads the track CSV file and converts it into a NumPy matrix."""
    df = pd.read_csv(fileName, header=None, sep=',')
    df = df.astype(float)
    matrice_circuito = df.to_numpy()
    print(f"Dimensioni matrice:{matrice_circuito.shape}")
    return matrice_circuito

def count_numpy_list(list_numpy, param):
    """Counts how many times a specific point (param) appears in a list of coordinates."""
    cont = 0
    for x in list_numpy:
        if x[0] == param[0] and x[1] == param [1]:
            cont = cont + 1

    return cont

def crea_matrice_distanze(percorsoFile, direzione):
    """
    Distance mapping function.
    Implements a Breadth-First Search (BFS) algorithm to calculate the distance 
    of each single asphalt cell from the finish line.
    """
    df = pd.read_csv(percorsoFile, sep = ',', header = None)
    matrice_distanze = df.to_numpy().copy()
    traguardo = argwhere(matrice_distanze,0.3)

    larghezza, altezza=matrice_distanze.shape

    for x in range(larghezza):
        for y in range(altezza):
            if matrice_distanze[x,y] != 0.0:
                matrice_distanze[x,y] = -2

    a = matrice_distanze

    index = copy.deepcopy(traguardo)

    for i in index:
        a[i[0],i[1]] = 0

    match direzione:
        case "destra":
            slider=[0,1]
        case "sinistra":
            slider=[0,-1]
        case "basso":
            slider=[1,0]
        case "alto":
            slider=[-1,0]
        case _:
            slider=[0,0]


    for i in index:
        i[0]=i[0]+slider[0]
        i[1]=i[1]+slider[1]
        a[i[0],i[1]]=1
    #print(f"Starting line {index}")

    while len(index) != 0:
        i = index.pop(0)
        # We need to figure out how to avoid moving backward from the finish line
        if (i[1]-1) >= 0 and a[i[0],i[1]-1] == -2 and traguardo.count([i[0],i[1]-1]) == 0:
            a[i[0],i[1]-1] = a[i[0],i[1]]+1
            index.append([i[0],i[1]-1])

        if (i[1]+1) < altezza and a[i[0],i[1]+1] == -2  and traguardo.count([i[0],i[1]+1]) == 0:
            a[i[0],i[1]+1] = a[i[0],i[1]]+1
            index.append([i[0],i[1]+1])

        if (i[0]-1) >= 0 and a[i[0]-1,i[1]] == -2  and traguardo.count([i[0]-1,i[1]]) == 0:
            a[i[0]-1,i[1]] = a[i[0],i[1]]+1
            index.append([i[0]-1,i[1]])

        if (i[0]+1) < larghezza and a[i[0]+1,i[1]] == -2  and traguardo.count([i[0]+1,i[1]]) == 0:
            a[i[0]+1,i[1]] = a[i[0],i[1]]+1
            index.append([i[0]+1,i[1]])

    # Convert to DataFrame
    traguardo_np = np.array(traguardo)
    righe = traguardo_np[:, 0]
    cols = traguardo_np[:, 1]

    # Apply the maximum value
    valore_max = np.max(a) + 1
    a[righe, cols] = valore_max
    df = pd.DataFrame(a)   

    out_dir = Path("../data/track_distance/")
    out_dir.mkdir(parents=True, exist_ok=True)

    nome_circuito = Path(percorsoFile).stem
    df.to_csv(out_dir / f"{nome_circuito}_distance.csv", index=False, header=False)

    return a

def crea_matrice_centro(matrice_circuito):
    """
    Creates a matrix where the value of each piece of asphalt represents 
    its linear distance from the center of the map.
    """
    altezza, larghezza = matrice_circuito.shape
    matrice_centro = np.zeros((altezza, larghezza), dtype=float)
    
    punti_pista = np.argwhere(matrice_circuito != 0.0)
    if len(punti_pista) > 0:
        min_x, min_y = punti_pista.min(axis=0)
        max_x, max_y = punti_pista.max(axis=0)
        
        centro_x = (min_x + max_x) / 2.0
        centro_y = (min_y + max_y) / 2.0
        
        for x in range(altezza):
            for y in range(larghezza):
                if matrice_circuito[x, y] != 0.0:
                    matrice_centro[x, y] = np.sqrt((x - centro_x)**2 + (y - centro_y)**2)
                    
    return matrice_centro



def salva_heatmap_csv(heatmap_dict, filename, shape):
    """
    Converts the dictionary { (r, c): count } into a CSV matrix.
   
    Args:
        heatmap_dict: The dictionary containing coordinates (tuples) and visitation counts.
        filename: The path where to save the file (e.g., "results/heatmap.csv").
        shape: The dimensions of the original matrix (e.g., (60, 60)).
    """
    # Create an empty matrix (all zeros) with the track's dimensions
    grid = np.zeros(shape, dtype=int)


    # Fill the matrix using the dictionary coordinates
    for coord, count in heatmap_dict.items():
        # Coord is a tuple (row, column)
        r, c = coord
       
        # Safety check to avoid crashes in case of out-of-bounds coordinates
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            grid[r, c] = count
   
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)


    # Save to file
    # fmt='%d' is used to save integers
    np.savetxt(filename, grid, delimiter=",", fmt='%d')
   
    print(f"Heatmap successfully saved in: {filename}")



def salva_heatmap_immagine(heatmap_dict, filename, track_matrix):
    """Generates and saves a visual heatmap representation."""
    shape = track_matrix.shape
    
    # Create the grid of visits
    grid_visits = np.zeros(shape, dtype=float)
    for coord, count in heatmap_dict.items():
        r, c = coord
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            grid_visits[r, c] = count

    # Replace zeros (unvisited road) with a very small number (e.g., 0.01).
    # This prevents LogNorm from crashing and rendering them transparent.
    grid_visits[grid_visits == 0] = 0.01

    # Create the Mask for the walls (True = Wall)
    wall_mask = (track_matrix == 0.0)

    # Calculate the maximum value for the scale
    max_val = np.max(grid_visits)
    if max_val < 1: max_val = 1

    # Configure the custom colormap
    # We take "inferno" and set the color for values "below the minimum" (under) to BLACK.
    my_cmap = copy.copy(plt.get_cmap("inferno"))
    my_cmap.set_under('black') 

    # Plot setup
    plt.figure(figsize=(10, 10))
    ax = plt.axes()
    ax.set_facecolor("lightgray") # Masked WALLS will show this color

    # Draw the heatmap
    sns.heatmap(grid_visits, 
                cmap=my_cmap,       
                mask=wall_mask,         # Walls become transparent -> The Gray underneath is visible
                cbar=True,            
                square=True, 
                xticklabels=False, 
                yticklabels=False,
                linewidths=0.0,
                # Scala Logaritmica:
                # vmin=1: Tutto ciò che è >= 1 usa la scala colori (Rosso/Giallo)
                # Tutto ciò che è < 1 (il nostro 0.01) usa il colore "under" (Nero)
                norm=LogNorm(vmin=1, vmax=max_val) 
                )

    plt.title("Heatmap: Grigio=Muro | Nero=Non Visitato | Colori=Visitato")
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150)
    plt.close()
    
    print(f" Heatmap Logaritmica (con bordi visibili) salvata in: {filename}")