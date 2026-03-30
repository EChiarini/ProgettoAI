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
from main_costants import *
from env.track_costants import VIEW_SIZE

from env.track_utils import salva_heatmap_csv, salva_heatmap_immagine
from utils.report_generator import genera_report
import agents.agent_costants  as agent_costants
import env.track_costants as track_costants
from agents.network import Network
import inspect

def run_training(number_episodes):
    """
    Orchestrates the Reinforcement Learning training loop.
    
    Args:
        number_episodes (int): Total number of training episodes to execute.
    """
    # Initialize the track environment and retrieve observation/action space dimensions
    env = TrackEnv(render_mode="human")
    state_shape = env.observation_space["agent_view"].shape
    state_size = env.observation_space["agent_view"].shape[0]
    number_actions = env.action_space.n
    print('State shape: ', state_shape)
    print('State size: ', state_size)
    print('Number of actions: ', number_actions)

    # Ensure directories for model checkpoints and results exist
    CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)


    step_limit = DEFAULT_TRAIN_EPISODES
    step_count=0
    
    # Initialize the Deep Q-Network Agent
    pilota = Agent(state_size, number_actions, number_episodes)

    # Metrics tracking for performance analysis
    scores = []
    scores_window = [] # Sliding window for calculating moving average (performance trend)
    best_avg_score = -float('inf')

    # Main execution loop
    loop = tqdm(range(number_episodes))
    start = time.perf_counter()

    # Fine-Tuning Module
    if TRAINING_MODE == 1:
        print(f"--- FINE TUNING ATTIVATO ---")
        search_path = Path("models/fine_tuning_model")
        found_files = list(search_path.glob("*.pth"))
        pilota.fine_tuning_mode = True
        
        if found_files:
            model_path = str(found_files[0])
            print(f"Trovato: {model_path}")
            # Load pre-trained weights and synchronize the target network
            pilota.load_model(model_path)
            pilota.target_net.load_state_dict(pilota.q_net.state_dict())
            print(">>> TARGET NET SINCRONIZZATA MANUALMENTE <<<")
        else:
            raise FileNotFoundError(f"Nessun file .pth trovato in {search_path}!")
        

    # Main Training Loop
    for i_episode in loop:
        # Reset environment for a new episode
        state, _ = env.reset(options={"direzione":"destra"})
        score = 0 # Cumulative reward for the current episode
        step_count = 0

        terminated = False
        truncated = False

        while not (terminated or truncated) and step_count < step_limit:

            # Action selection: Execute Epsilon-Greedy policy
            action = pilota.select_action(state)

            # Environment interaction: Execute action and observe transition
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Training step: Store experience and perform backpropagation
            pilota.step(state, action, reward, next_state, done)

            # State transition update
            state = next_state
            score += reward
            step_count += 1

        # Post-Episode
        # Decay exploration rate (epsilon)
        pilota.update_epsilon()

        # Update scoring metrics and moving average
        scores.append(score)
        scores_window.append(score)
        if len(scores_window) > SCORES_WINDOW_SIZE: scores_window.pop(0) # Keep only the last window size elements
        avg_score = np.mean(scores_window)

        # Update progress bar with real-time telemetry
        loop.set_description(f"Ep: {i_episode+1} | Score: {score:.2f} | Avg Score: {np.mean(scores_window):.2f} | Epsilon: {pilota.epsilon:.3f}")

        # Save the model if it achieves a new high score
        if score > best_avg_score:
            best_avg_score = score
            torch.save(pilota.q_net.state_dict(), CHECKPOINTS_PATH / DEFAULT_MODEL_FILENAME)

        # Periodic checkpointing for long-term recovery
        if (i_episode + 1) % SAVE_CHECKPOINT_EVERY == 0:
            torch.save(pilota.q_net.state_dict(), CHECKPOINTS_PATH / f"cp_{i_episode+1}.pth")


    # Save final training plot (Learning Curve)
    save_training_plot(scores, filename = RESULTS_PATH / DEFAULT_GRAPH_FILENAME)

    end = time.perf_counter()

    elapsed = end - start

    # Calculate and format execution time
    ore = int(elapsed // 3600)
    minuti = int((elapsed % 3600) // 60)
    secondi = elapsed % 60

    total_time = f"Tempo impiegato: {ore} h {minuti} m {secondi:.2f} s"

    print(f"Punteggio Totale: {score:.2f}")

    # Analytical Data Export
    # Save trajectory heatmap as raw CSV data
    salva_heatmap_csv(
    env.trajectory_heat_map,
    f"{RESULTS_PATH}/{REPORT_FILENAME}.csv",
    env.matrix.shape
    )

    # Generate spatial heatmap visualization
    print("Generazione grafico Heatmap...")
    salva_heatmap_immagine(
    env.trajectory_heat_map, 
    f"{RESULTS_PATH}/{REPORT_FILENAME}.png", 
    env.matrix)

    # Define paths for the automated report generator
    grafico_path = os.path.join(os.getcwd(), RESULTS_PATH, DEFAULT_GRAPH_FILENAME)
    heatmap_path = os.path.join(os.getcwd(), RESULTS_PATH, f"{REPORT_FILENAME}.png")
    source_code_step = inspect.getsource(TrackEnv.step)
    
    # Generate comprehensive PDF report
    if GEN_REPORT:
        try:
            genera_report(
                    args.ep, 
                    grafico_path, 
                    heatmap_path, 
                    agent_costants, 
                    track_costants,
                    Network,          
                    source_code_step,
                    total_time, 
                    TRAINING_MODE
                )
        except Exception as e:
            print(f"Errore generazione report: {e}")
            import traceback
            traceback.print_exc() # Prints the full stack trace if generation fails

    env.close()


def run_testing(model_path, delay=DEFAULT_TEST_DELAY):
    """
    Loads a pre-trained model for inference and visual evaluation.
    
    Args:
        model_path (str): Path to the .pth model file.
        delay (float): Time delay (seconds) between frames to slow down visualization.
    """

    # Resolve model file path (accepts absolute or relative paths)
    if os.path.isabs(model_path):
        model_fullpath = model_path
    else:
        model_fullpath = os.path.join(os.getcwd(), CHECKPOINTS_PATH, model_path)

    print(model_fullpath)

    if not os.path.exists(model_fullpath):
        print(f"ERRORE: Il file '{model_fullpath}' non esiste.")
        return

    print(f"Caricamento modello da: {model_fullpath}...")

    # Initialize environment in evaluation mode
    env_test = TrackEnv(render_mode="human", is_testing=True)
    
    view_size = VIEW_SIZE
    action_size = env_test.action_space.n

    # Instantiate Agent for inference
    pilota_test = Agent(view_size, action_size, 1)

    # Load trained weights into the policy network
    state_dict = torch.load(model_fullpath, map_location=torch.device(DEVICE))
    pilota_test.q_net.load_state_dict(state_dict)
    
   # Set network to evaluation mode (disables Dropout/BatchNorm if present)
    pilota_test.q_net.eval() 

    # Disable exploration: Set Epsilon to 0 for pure exploitation
    pilota_test.epsilon = 0.0

    step_limit = STEP_LIMIT

    # Inference Loop
    state, _ = env_test.reset(options={"direzione":"destra"})
    score = 0
    step = 0
    done = False

    
   
        
    while not done:
        # Render the current state
        env_test.render()
        
        # Policy selection (Optimal action according to the learned Q-values)
        action = pilota_test.select_action(state)
        
        # Execute action in environment
        next_state, reward, terminated, truncated, _ = env_test.step(action)
        
        done = terminated or truncated
        state = next_state
        print(reward)
        score += reward
        step += 1
        
        # Slow down for human observation
        time.sleep(delay)

        
        # Safety break to prevent infinite loops if the agent gets stuck
        if step > step_limit:
            print("Loop troppo lungo, interrompo episodio.")
            break


    env_test.render()
    print(f"Score: {score:.2f} | Steps: {step} | Progresso: {env_test._progresso}/{env_test.numero_checkpoints}")
    time.sleep(2)

    env_test.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--ep", type=int, default=DEFAULT_TRAIN_EPISODES, help="numero di episodi per il training (default: 10000)")
    parser.add_argument("--mod", type=str, default=DEFAULT_MODEL_FILENAME, help="file per il testing (default: best_model.pth)")
    args = parser.parse_args()

    if args.mode == "train":
        print(f"training con {args.ep} episodi")
        run_training(args.ep)
        
    elif args.mode == "test":
        print(f"testo con {args.mod}")
        run_testing(args.mod)