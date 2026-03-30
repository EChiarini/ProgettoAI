import torch
import torch.optim as optim
import random
import os
import numpy as np
import torch.nn.functional as F
from collections import deque
from .network import Network
from utils import DEVICE

from .agent_costants import *


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    Implementation of Experience Replay to break temporal correlation between samples.
    """
    def __init__(self, capacity, batch_size, seed=DEFAULT_SEED):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.seed = random.seed(seed)


    def add(self, state, action, reward, next_state, done):
        """Adds a new experience transition to the buffer."""
        self.memory.append((state, action, reward, next_state, done))


    def sample(self):
        """Randomly samples a batch of experiences for the learning step."""
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones


    def __len__(self):
        """Returns the current size of internal memory."""
        return len(self.memory)

class Agent:
    """
    The DRL Agent interacting with the environment using the DQN algorithm.
    It manages two networks (Local and Target) to stabilize learning.
    """
    def __init__(self, view_size, action_size, number_episodes, seed=DEFAULT_SEED):

        self.action_size = action_size
        self.view_size = view_size
        self.fine_tuning_mode = False
        
        # The input is the flattened grid of the agent's local view
        self.input_dim = view_size * view_size 
        
        # Policy Network: Used to select actions and trained via backpropagation
        self.q_net = Network(self.input_dim, action_size, seed).to(DEVICE)

        # Target Network: Used to provide stable Q-value targets for the loss function
        self.target_net = Network(self.input_dim, action_size, seed).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval() # Set to evaluation mode (no gradient computation)
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE, batch_size=MINIBATCH_SIZE, seed=seed)
        
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.t_step = 0

        # Dynamic calculation of Epsilon Decay based on the total number of episodes
        eps_type = EPSYLON_TYPE[EPSYLON_IS]

        match eps_type:
            case "lineare": 
                self.epsilon_decay = 1/(number_episodes*EPSILON_DECAY_PORTION)
            case "esponenziale": 
                T = max(1, int(number_episodes * EPSILON_DECAY_PORTION))
                self.epsilon_decay = (self.epsilon_min / self.epsilon) ** (1 / T)
            case _: raise ValueError(f"Invalid EPSYLON_TYPE: {eps_type}")


    def _preprocess_state(self, state_dict):
        """Flattens the observation grid into a vector suitable for the MLP input."""
        if isinstance(state_dict, dict):
            view = state_dict["agent_view"]
            return view.flatten()
        elif isinstance(state_dict, (list, tuple)):
            batch_views = [s["agent_view"] for s in state_dict]
            return np.array(batch_views).reshape(len(batch_views), -1)


    def update_epsilon(self):
        """Reduces exploration probability (epsilon) over time."""
        eps_type = EPSYLON_TYPE[EPSYLON_IS]

        match eps_type:
            case "lineare": self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
            case "esponenziale": self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            case _: raise ValueError(f"Invalid EPSYLON_TYPE: {eps_type}")


    def select_action(self, state):
        """
        Implements the Epsilon-Greedy policy.
        - Exploratory action with probability Epsilon.
        - Exploitative (Greedy) action with probability 1 - Epsilon.
        """
        processed_state = self._preprocess_state(state)
        state_t = torch.from_numpy(processed_state).float().unsqueeze(0).to(DEVICE)
        
        self.q_net.eval()
        with torch.no_grad():
            action_values = self.q_net(state_t)
        self.q_net.train()

        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


    def step(self, state, action, reward, next_state, done):
        """Saves transition to Replay Buffer and triggers the learning step periodically."""
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1 
        
        # Learn from experience every TARGET_UPDATE_EVERY steps
        if self.t_step % TARGET_UPDATE_EVERY == 0:
            # Logic to handle both Training from scratch and Fine-Tuning
            soglia = FINE_TUNING_WARMUP if self.fine_tuning_mode else self.memory.batch_size
            
            if len(self.memory) > soglia:
                self.learn()

        # Hard Update: Periodically synchronize the Target Network weights with the Local Network
        if self.t_step % 1000 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


    def learn(self):
        """
        Performs a gradient descent step based on the Bellman Equation
        """
        states, actions, rewards, next_states, dones = self.memory.sample()

        states_np = self._preprocess_state(states)
        next_states_np = self._preprocess_state(next_states)

        states_t = torch.tensor(states_np).float().to(DEVICE)
        next_states_t = torch.tensor(next_states_np).float().to(DEVICE)
        
        actions_t = torch.tensor(actions).long().unsqueeze(1).to(DEVICE)
        rewards_t = torch.tensor(rewards).float().unsqueeze(1).to(DEVICE)
        dones_t = torch.tensor(dones).float().unsqueeze(1).to(DEVICE)

        # Get current Q-values for the actions taken
        q_expected = self.q_net(states_t).gather(1, actions_t)
        
        # Compute the target Q-values using the Target Network
        with torch.no_grad():
            q_next = self.target_net(next_states_t).detach().max(1)[0].unsqueeze(1)
            q_target = rewards_t + (self.gamma * q_next * (1 - dones_t))
        
        # Loss minimization (Mean Squared Error)
        loss = F.mse_loss(q_expected, q_target)
        
  
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping to prevent Exploding Gradients and ensure training stability
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

    def load_model(self, file_path):
        """Loads pre-trained weights for evaluation or fine-tuning."""
        if os.path.isfile(file_path):

            checkpoint = torch.load(file_path, map_location=DEVICE)
            self.q_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(self.q_net.state_dict())
        
            print(f"--- Model loaded successfully from: {file_path} ---")
        else:
            print(f"ERROR: Checkpoint file not found at {file_path}")