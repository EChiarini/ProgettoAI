import pandas as pd
import copy
from pathlib import Path
from .track_constants import (
  get_default_track_path,
  TRACK_FINISH_VALUE,
  TRACK_OFFROAD_VALUE,
  TRACK_UNKNOWN_VALUE,
)

def argwhere(matrix, value):
  l = list()
  x_max,y_max=matrix.shape
  for x in range(x_max):
    for y in range(y_max):

      if matrix[x,y] == value:
        l.append([x,y])

  return l


def build_track(fileName=None):
  if fileName is None:
    fileName = get_default_track_path()
  df = pd.read_csv(fileName, header=None, sep=',')
  df = df.astype(int)
  matrice_circuito = df.to_numpy()
  print(f"Dimensioni matrice:{matrice_circuito.shape}")
  return matrice_circuito

def crea_matrice_distanze(percorsoFile, direzione):
    df = pd.read_csv(percorsoFile, sep = ',', header = None)
    matrice_distanze = df.to_numpy().copy()
    traguardo = argwhere(matrice_distanze, TRACK_FINISH_VALUE)

    larghezza, altezza=matrice_distanze.shape

    for x in range(larghezza):
        for y in range(altezza):
            if matrice_distanze[x,y] != TRACK_OFFROAD_VALUE:
                matrice_distanze[x,y] = TRACK_UNKNOWN_VALUE

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
        if (i[1]-1) >= 0 and a[i[0],i[1]-1] == TRACK_UNKNOWN_VALUE and traguardo.count([i[0],i[1]-1]) == 0:
            a[i[0],i[1]-1] = a[i[0],i[1]]+1
            index.append([i[0],i[1]-1])

        if (i[1]+1) < altezza and a[i[0],i[1]+1] == TRACK_UNKNOWN_VALUE  and traguardo.count([i[0],i[1]+1]) == 0:
            a[i[0],i[1]+1] = a[i[0],i[1]]+1
            index.append([i[0],i[1]+1])

        if (i[0]-1) >= 0 and a[i[0]-1,i[1]] == TRACK_UNKNOWN_VALUE  and traguardo.count([i[0]-1,i[1]]) == 0:
            a[i[0]-1,i[1]] = a[i[0],i[1]]+1
            index.append([i[0]-1,i[1]])

        if (i[0]+1) < larghezza and a[i[0]+1,i[1]] == TRACK_UNKNOWN_VALUE  and traguardo.count([i[0]+1,i[1]]) == 0:
            a[i[0]+1,i[1]] = a[i[0],i[1]]+1
            index.append([i[0]+1,i[1]])

    # 2. Convertiamo in DataFrame
    df = pd.DataFrame(a)

    # 3. Salviamo in CSV
    output_dir = Path("../data/track_distance/")
    output_dir.mkdir(parents=True, exist_ok=True)

    nome_circuito = Path(percorsoFile).stem
    df.to_csv(output_dir / f"{nome_circuito}_distance.csv", index=False, header=False)

    return a
