import pandas as pd
import numpy as np
import copy
import os
from pathlib import Path
from .track_costants import get_default_track_path
import matplotlib.pyplot as plt
import seaborn as sns

def argwhere(matrix, value):
  l = list()
  x_max,y_max=matrix.shape
  for x in range(x_max):
    for y in range(y_max):
    #  print(f"CHHHHHHHH{matrix[x,y]}")
      if matrix[x,y] == value:
        l.append([x,y])

  return l


def build_track(fileName = get_default_track_path()):
    df = pd.read_csv(fileName, header=None, sep=',')
    df = df.astype(float)
    matrice_circuito = df.to_numpy()
    print(f"Dimensioni matrice:{matrice_circuito.shape}")
    return matrice_circuito

def count_numpy_list(list_numpy, param):
    cont = 0
    for x in list_numpy:
        if x[0] == param[0] and x[1] == param [1]:
            cont = cont + 1

    return cont

def crea_matrice_distanze(percorsoFile, direzione):
    df = pd.read_csv(percorsoFile, sep = ',', header = None)
    matrice_distanze = df.to_numpy().copy()
    traguardo = argwhere(matrice_distanze,0.3)

    larghezza, altezza=matrice_distanze.shape

    for x in range(larghezza):
        for y in range(altezza):
            if matrice_distanze[x,y] != 0.0:
                matrice_distanze[x,y] = -2

    a = matrice_distanze

    index = copy.deepcopy(traguardo) #secondo me ha più senso calcolare la distanza dalla prima cella del traguardo disponibile rispetto ad una (list([[35,50]]))

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
    #print(f"linea iniziale {index}")

    while len(index) != 0:
        i = index.pop(0)
        #bisogna capire come non fare andare all'indietro del traguardo
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

    # 2. Convertiamo in DataFrame
    df = pd.DataFrame(a)

    out_dir = Path("../data/track_distance/")
    out_dir.mkdir(parents=True, exist_ok=True)

    nome_circuito = Path(percorsoFile).stem
    df.to_csv(out_dir / f"{nome_circuito}_distance.csv", index=False, header=False)

    return a





def salva_heatmap_csv(heatmap_dict, filename, shape):
    """
    Converte il dizionario { (r, c): conteggio } in una matrice CSV.
   
    Args:
        heatmap_dict: Il dizionario contenente le coordinate (tuple) e i valori.
        filename: Il percorso dove salvare il file (es. "results/heatmap.csv").
        shape: Le dimensioni della matrice originale (es. (60, 60)).
    """
    # 1. Crea una matrice vuota (tutti zeri) delle dimensioni della pista
    grid = np.zeros(shape, dtype=int)


    # 2. Riempie la matrice usando le coordinate del dizionario
    for coord, count in heatmap_dict.items():
        # Assumiamo coord sia una tupla (riga, colonna)
        r, c = coord
       
        # Controllo di sicurezza per evitare crash se ci sono coordinate strane
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            grid[r, c] = count
   
    # 3. Crea la cartella se non esiste
    os.makedirs(os.path.dirname(filename), exist_ok=True)


    # 4. Salva su file
    # fmt='%d' serve a salvare numeri interi (niente virgole decimali 0.00)
    np.savetxt(filename, grid, delimiter=",", fmt='%d')
   
    print(f"✅ Heatmap salvata correttamente in: {filename}")



def salva_heatmap_immagine(heatmap_dict, filename, track_matrix):
    """
    Genera una heatmap dove:
    - I muri sono GRIGI.
    - La strada non visitata è NERA/VIOLA SCURO.
    - La strada visitata è COLORATA (Arancione/Giallo).
    
    Args:
        heatmap_dict: Dizionario delle visite.
        filename: Dove salvare l'immagine.
        track_matrix: La matrice originale della pista (serve per trovare i muri).
    """
    shape = track_matrix.shape
    
    # 1. Crea la griglia delle visite (inizializzata a 0)
    grid_visits = np.zeros(shape, dtype=float)
    for coord, count in heatmap_dict.items():
        r, c = coord
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            grid_visits[r, c] = count

    # 2. Crea una "Maschera" per i muri
    # True dove c'è il muro (valore 0 nella tua matrice), False dove c'è la strada
    wall_mask = (track_matrix == 0.0)

    # 3. Setup Grafico
    plt.figure(figsize=(10, 10))
    
    # Impostiamo il colore di sfondo della figura a GRIGIO (questo sarà il colore dei muri)
    ax = plt.axes()
    ax.set_facecolor("lightgray") 

    # 4. Disegna la Heatmap
    # mask=wall_mask -> Dice a seaborn di rendere TRASPARENTI i pixel dei muri.
    # Essendo trasparenti, si vedrà il grigio sotto.
    sns.heatmap(grid_visits, 
                cmap="inferno",       # Colori dal nero al giallo
                mask=wall_mask,       # Nascondi i muri (mostra sfondo grigio)
                cbar=True,            # Barra laterale
                square=True, 
                xticklabels=False, 
                yticklabels=False,
                linewidths=0.0,       # Nessuna linea tra i pixel per fluidità
                vmin=0)               # Fissa il minimo a 0

    plt.title("Heatmap: Grigio=Fuori | Nero=Pista ")
    
    # 5. Salva
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150) # dpi più alto per qualità migliore
    plt.close()
    
    print(f"🔥 Heatmap con bordi salvata in: {filename}")