# Seed for random number generators to ensure reproducibility
DEFAULT_SEED = 42

# Optimization step size for the neural network weight updates
LEARNING_RATE = 0.0001

# Number of experience tuples sampled from the replay buffer per learning step
MINIBATCH_SIZE = 100

# Maximum capacity of the experience replay buffer
REPLAY_BUFFER_SIZE = 200000

# Gamma parameter: defines the importance of future rewards (0 = myopic, 1 = far-sighted)
DISCOUNT_FACTOR = 0.99

# Starting value of Epsilon for the epsilon-greedy action selection (1.0 = 100% random exploration)
EPSILON_START = 1.0

# Minimum value Epsilon can reach, ensuring a baseline level of exploration even late in training
EPSILON_MIN = 0.005

# Fraction of total training episodes over which Epsilon decays to its minimum value
EPSILON_DECAY_PORTION = 0.8

# Frequency (in steps) at which the local network performs a gradient descent learning step
TARGET_UPDATE_EVERY = 4

# Available Epsilon decay strategies
EPSYLON_TYPE = {
    1: "lineare",       # Linear decay
    2: "esponenziale"   # Exponential decay
}

# Selected Epsilon decay strategy (2 = Exponential)
EPSYLON_IS = 2

# Number of initial steps to purely populate the Replay Buffer before triggering learning during fine-tuning
FINE_TUNING_WARMUP = 20000