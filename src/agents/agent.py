import torch
import torch.optim as optim
import random
import numpy as np
import torch.nn.functional as F
from collections import deque
from .network import Network
from utils import DEVICE

from .agent_costants import *

class ReplayBuffer:

    def __init__(self, capacity, batch_size, seed=DEFAULT_SEED):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.seed = random.seed(seed)


    def add(self, state, action, reward, next_state, done):
        # Salviamo tutto come numpy array o semplici float
        self.memory.append((state, action, reward, next_state, done))


    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        # Scompattiamo il batch
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones


    def __len__(self):
        return len(self.memory)

class Agent:

    def __init__(self, view_size, action_size, number_episodes, seed=DEFAULT_SEED):

        self.action_size = action_size
        self.view_size = view_size
        
        # Calcoliamo la dimensione totale dell'input appiattito
        self.input_dim = view_size * view_size 
        
        # Spostiamo le reti sul DEVICE corretto
        self.q_net = Network(self.input_dim, action_size, seed).to(DEVICE)
        self.target_net = Network(self.input_dim, action_size, seed).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE, batch_size=MINIBATCH_SIZE, seed=seed)
        
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.t_step = 0

        eps_type = EPSYLON_TYPE[EPSYLON_IS]

        match eps_type:
            case "lineare": 
                self.epsilon_decay = 1/(number_episodes*EPSILON_DECAY_PORTION)
            case "esponenziale": 
                T = max(1, int(number_episodes * EPSILON_DECAY_PORTION))
                self.epsilon_decay = (self.epsilon_min / self.epsilon) ** (1 / T)
            case _: raise ValueError(f"EPSYLON_TYPE non valido: {eps_type}")


    def _preprocess_state(self, state_dict):
        """
        Prende un dizionario (o lista di dizionari) ed estrae/appiattisce i dati
        in un unico array numpy o tensore.
        """
        if isinstance(state_dict, dict):
            view = state_dict["agent_view"]
            return view.flatten()
            
        elif isinstance(state_dict, (list, tuple)):
            batch_views = [s["agent_view"] for s in state_dict]
            return np.array(batch_views).reshape(len(batch_views), -1)


    def update_epsilon(self):
        eps_type = EPSYLON_TYPE[EPSYLON_IS]

        match eps_type:
            case "lineare": self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
            case "esponenziale": self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            case _: raise ValueError(f"EPSYLON_TYPE non valido: {eps_type}")


    def select_action(self, state):
        # 1. Preprocess: estrae e appiattisce -> (49,)
        processed_state = self._preprocess_state(state)
        
        # 2. Trasforma in Tensor, aggiungi batch dim -> (1, 49) E SPOSTA SU DEVICE
        # AGGIUNTO .to(DEVICE) QUI SOTTO
        state_t = torch.from_numpy(processed_state).float().unsqueeze(0).to(DEVICE)
        
        self.q_net.eval()
        with torch.no_grad():
            action_values = self.q_net(state_t)
        self.q_net.train()

        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy()) # .cpu() serve per portarlo fuori dalla GPU se necessario
        else:
            return random.choice(np.arange(self.action_size))


    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % TARGET_UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > self.memory.batch_size:
                self.learn()


    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample()

        # 1. Preprocessiamo il batch
        states_np = self._preprocess_state(states)
        next_states_np = self._preprocess_state(next_states)

        # 2. Conversione in Tensor E SPOSTAMENTO SU DEVICE
        # AGGIUNTO .to(DEVICE) A TUTTI I TENSORI
        states_t = torch.tensor(states_np).float().to(DEVICE)
        next_states_t = torch.tensor(next_states_np).float().to(DEVICE)
        
        actions_t = torch.tensor(actions).long().unsqueeze(1).to(DEVICE)
        rewards_t = torch.tensor(rewards).float().unsqueeze(1).to(DEVICE)
        dones_t = torch.tensor(dones).float().unsqueeze(1).to(DEVICE)

        # --- Calcolo della loss ---
        q_expected = self.q_net(states_t).gather(1, actions_t)
        
        with torch.no_grad():
            q_next = self.target_net(next_states_t).detach().max(1)[0].unsqueeze(1)
            q_target = rewards_t + (self.gamma * q_next * (1 - dones_t))
        
        loss = F.mse_loss(q_expected, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
