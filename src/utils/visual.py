import matplotlib.pyplot as plt
import numpy as np
import os

def save_training_plot(scores, filename="training_plot.png", window_size=50):
    """
    Crea e salva un grafico dell'andamento dei reward.
    
    Args:
        scores (list): La lista completa dei punteggi per ogni episodio.
        filename (str): Il percorso dove salvare l'immagine.
        window_size (int): L'ampiezza della finestra per la media mobile.
    """
    # Imposta la dimensione della figura
    plt.figure(figsize=(10, 6))
    
    
    # 2. Calcolo e Plot della Media Mobile (Trend)
    # Se abbiamo abbastanza dati per calcolare la media
    if len(scores) >= window_size:
        # Calcola la media mobile usando la convoluzione (molto veloce)
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        
        # L'asse X per la media mobile deve essere spostato per allinearsi
        # (parte dall'episodio 'window_size')
        x_axis = np.arange(window_size - 1, len(scores))
        
        plt.plot(x_axis, moving_avg, color='blue', linewidth=2, label=f'Media Mobile ({window_size} ep)')

    # Etichette e Titolo
    plt.title('Andamento Addestramento DQN')
    plt.xlabel('Episodio')
    plt.ylabel('Reward Totale (Score)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Salva il file
    plt.savefig(filename)
    print(f"Grafico salvato in: {filename}")
    
    # Chiudi la figura per liberare memoria
    plt.close()
