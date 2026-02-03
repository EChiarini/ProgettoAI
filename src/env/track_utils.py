import pandas as pd
import copy
import os
from pathlib import Path

def argwhere(matrix, value):
  l = list()
  x_max,y_max=matrix.shape
  for x in range(x_max):
    for y in range(y_max):

      if matrix[x,y] == value:
        l.append([x,y])

  return l


def build_track(fileName = os.getcwd() + "/data/tracks/track_imola.csv"):
    df = pd.read_csv(fileName, header=None, sep=',')
    df = df.astype(int)
    matrice_circuito = df.to_numpy()
    print(f"Dimensioni matrice:{matrice_circuito.shape}")
    return matrice_circuito

def crea_matrice_distanze(percorsoFile, direzione):
    df = pd.read_csv(percorsoFile, sep = ',', header = None)
    matrice_distanze = df.to_numpy().copy()
    traguardo = argwhere(matrice_distanze,2)

    larghezza, altezza=matrice_distanze.shape

    for x in range(larghezza):
        for y in range(altezza):
            if matrice_distanze[x,y] != -1:
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
    print(f"linea iniziale {index}")

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

    # 3. Salviamo in CSV
    Path("../data/track_distance/").mkdir(parents=True, exist_ok=True) # crea la cartella se non esiste, evita errore
    nome_circuito = percorsoFile.split("/")[-1].split(".")[0]
    df.to_csv(f'../data/track_distance/{nome_circuito}_distance.csv', index=False, header=False)

    return a