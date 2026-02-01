import argparse
import os
import torch
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np

from env.track_env import TrackEnv
from agents.agent import Agent
from utils.visual import save_training_plot
from utils import DEVICE

def run_training(number_episodes):
    env = TrackEnv(render_mode="human")
    state_shape = env.observation_space["agent_view"].shape

    state_size = env.observation_space["agent_view"].shape[0]
    number_actions = env.action_space.n
    print('State shape: ', state_shape)
    print('State size: ', state_size)
    print('Number of actions: ', number_actions)

    Path("../models/checkpoints").mkdir(parents=True, exist_ok=True) # crea la cartella se non esiste, evita errore

    interpolation_parameter = 1e-3
    step_limit = 1000
    step_count=0

    pilota = Agent(state_size, number_actions, number_episodes)

    # Liste per tenere traccia dei punteggi
    scores = []
    scores_window = [] # Per la media mobile (es. ultimi 100 episodi)

    # Loop principale
    loop = tqdm(range(number_episodes))

    for i_episode in loop:
        state, _ = env.reset()
        score = 0 # Punteggio dell'episodio corrente
        step_count = 0

        terminated = False
        truncated = False

        while not (terminated or truncated) and step_count < step_limit:

            # 1. Scelta Azione
            action = pilota.select_action(state)

            # 2. Step nell'ambiente
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            
            

            # 3. Addestramento (Memorizza e impara)
            pilota.step(state, action, reward, next_state, done)

            # 4. Aggiornamento variabili
            state = next_state
            score += reward
            step_count += 1

        # --- Fine Episodio ---
        pilota.update_epsilon()

        # Salviamo i punteggi
        scores.append(score)
        scores_window.append(score)
        if len(scores_window) > 100: scores_window.pop(0) # Teniamo solo gli ultimi 100

        # Aggiorniamo la barra di caricamento con le info utili
        loop.set_description(f"Ep: {i_episode+1} | Score: {score:.2f} | Avg Score: {np.mean(scores_window):.2f} | Epsilon: {pilota.epsilon:.3f}")

        # (Opzionale) Salviamo il modello se otteniamo un buon risultato o ogni 100 episodi
        if (i_episode + 1) % 100 == 0:
            torch.save(pilota.q_net.state_dict(), os.getcwd() + f'/models/checkpoints/checkpoint_ep_{i_episode+1}.pth')


    Path("../results/").mkdir(parents=True, exist_ok=True) # crea la cartella se non esiste, evita errore
    save_training_plot(scores, filename = os.getcwd() + "/results/grafico_finale.png")

    env.close()


def run_testing(model_path, num_episodes=5, delay=0.1):
    """
    Carica un modello addestrato e lo visualizza in azione.
    
    Args:
        model_path (str): Il percorso del file .pth (es. 'checkpoint_ep_500.pth')
        num_episodes (int): Quanti episodi di test vuoi vedere.
        delay (float): Secondi di pausa tra un frame e l'altro (per rallentare l'azione).
    """
    
    # 1. Risolvi il percorso del file modello (accetta path assoluti o solo il nome)
    if os.path.isabs(model_path):
        model_fullpath = model_path
    else:
        model_fullpath = os.path.join(os.getcwd(), "models", "checkpoints", model_path)

    if not os.path.exists(model_fullpath):
        print(f"ERRORE: Il file '{model_fullpath}' non esiste.")
        return

    print(f"Caricamento modello da: {model_fullpath}...")

    # 2. Crea l'ambiente in modalità 'human' per il rendering
    #    Nota: view_size deve essere uguale a quello usato in training (7)
    env_test = TrackEnv(render_mode="human")
    
    view_size = 7 
    action_size = env_test.action_space.n

    # 3. Istanzia l'agente (la struttura deve essere IDENTICA a quella del training)
    #    Non ci serve la memoria o l'optimizer qui, ma la classe Agent li crea comunque.
    pilota_test = Agent(view_size, action_size, 1)

    # 4. Carica i pesi nella rete (q_net)
    #    map_location serve se hai addestrato su GPU ma testi su CPU
    state_dict = torch.load(model_fullpath, map_location=torch.device(DEVICE))
    pilota_test.q_net.load_state_dict(state_dict)
    
    # Imposta la rete in modalità valutazione (disattiva dropout, batchnorm, ecc.)
    pilota_test.q_net.eval() 

    # 5. Imposta Epsilon a 0 -> Solo sfruttamento (Exploitation), niente esplorazione
    pilota_test.epsilon = 0.0

    step_limit = 300

    # --- CICLO DI TEST ---
    for i in range(num_episodes):
        state, _ = env_test.reset()
        score = 0
        step = 0
        done = False
        
        print(f"\n--- Inizio Episodio di Test {i+1} ---")
        
        while not done:
            # Renderizza la scena
            env_test.render()
            
            # Scegli l'azione (sarà sempre la migliore secondo la rete)
            action = pilota_test.select_action(state)
            
            # Esegui l'azione
            next_state, reward, terminated, truncated, _ = env_test.step(action)
            
            done = terminated or truncated
            state = next_state
            score += reward
            step += 1
            
            # Rallenta un po' per permettere all'occhio umano di seguire
            time.sleep(delay)
            
            # Sicurezza per evitare loop infiniti se l'agente si blocca
            if step > step_limit:
                print("Loop troppo lungo, interrompo episodio.")
                break

        print(f"Episodio {i+1} terminato. Punteggio Totale: {score:.2f}")

    env_test.close()
    print("Test completato.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--ep", type=int, default=10000, help="numero di episodi per il training (default: 10000)")
    parser.add_argument("--mod", type=int, default=100, help="nome del file .pth (default: 100)")
    args = parser.parse_args()

    if args.mode == "train":
        print(f"training con {args.ep} episodi")
        run_training(args.ep)
        
    elif args.mode == "test":
        print(f"testo con {args.mod}")
        run_testing(f"checkpoint_ep_{args.mod}.pth")