import pandas as pd
import numpy as np
import copy
import os
from pathlib import Path
from .track_costants import get_default_track_path

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