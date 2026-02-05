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
    shape = track_matrix.shape
    
    # 1. Crea la griglia delle visite
    grid_visits = np.zeros(shape, dtype=float)
    for coord, count in heatmap_dict.items():
        r, c = coord
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            grid_visits[r, c] = count

    # 2. TRUCCO PER LA SCALA LOGARITMICA:
    # Sostituisci gli zeri (strada non visitata) con un numero piccolissimo (es. 0.01).
    # In questo modo LogNorm non "esplode" e non li rende trasparenti.
    grid_visits[grid_visits == 0] = 0.01

    # 3. Crea la Maschera per i muri (True = Muro)
    wall_mask = (track_matrix == 0.0)

    # 4. Calcola il massimo per la scala
    max_val = np.max(grid_visits)
    if max_val < 1: max_val = 1

    # 5. Configura la Colormap Personalizzata
    # Prendiamo "inferno" e impostiamo il colore per i valori "sotto il minimo" (under) a NERO.
    my_cmap = copy.copy(plt.get_cmap("inferno"))
    my_cmap.set_under('black') 

    # 6. Setup Grafico
    plt.figure(figsize=(10, 10))
    ax = plt.axes()
    ax.set_facecolor("lightgray") # I MURI mascherati mostreranno questo colore

    # 7. Disegna
    sns.heatmap(grid_visits, 
                cmap=my_cmap,       
                mask=wall_mask,         # I muri diventano trasparenti -> Si vede il Grigio sotto
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
    
    print(f"🔥 Heatmap Logaritmica (con bordi visibili) salvata in: {filename}")