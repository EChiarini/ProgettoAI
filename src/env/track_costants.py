import os

# Track representation values within the grid
TRACK_FINISH_VALUE = 0.3
TRACK_CURB_VALUE = 0.1
TRACK_ROAD_VALUE = 0.2
TRACK_OFFROAD_VALUE = 0.0
TRACK_UNKNOWN_VALUE = -0.1

# Rendering configuration
RENDER_FPS = 30 
WINDOW_SIZE_PX = 800

# Agent view and general track parameters
VIEW_SIZE = 16
ROAD_WIDTH = 7
NUM_CHECKPOINTS = 7
ACTION_SPACE_SIZE = 4

# Movement mode selection: 
# "simple" (4 actions, 1-cell step) or "velocity" (9 actions, acceleration-based)
MOVEMENT_MODE = "simple"  # "simple" | "velocity" | "simple_aggiornato"

# Velocity mode specific constants
VELOCITY_MAX_SPEED = 4
VELOCITY_ACTION_SPACE_SIZE = 9

# Observation space bounds for the agent's view matrix
OBS_LOW = TRACK_UNKNOWN_VALUE
OBS_HIGH = TRACK_FINISH_VALUE

# Reward shaping constants
OFFROAD_REWARD = -50
STEP_PENALTY = -0.01
CHECKPOINT_REWARD = 30
CURB_REWARD = -0.20
FINISH_REWARD = 100
REPEAT_PENALTY = -0.5
BACKWARD_PENALTY = 1
CENTER_WEIGHT = 0.2

# Track and asset file configurations
DEFAULT_TRACK_FILENAME = "track_imola.csv"
DEFAULT_TRACK_RELATIVE_PATH = os.path.join("data", "tracks", DEFAULT_TRACK_FILENAME)
DEFAULT_DISTANCE_DIRECTION = "destra"
DEFAULT_TRACK_MUSIC = "coconut_mall.mp3"

def get_default_track_path():
    """Returns the absolute path to the default track CSV file."""
    return os.path.join(os.getcwd(), DEFAULT_TRACK_RELATIVE_PATH)


# Anti-aliasing (SSAA: render at Nx resolution, smoothscale down)
RENDER_SCALE = 5

# Visual Configuration (Nightlife Futuristic Theme)
COLOR_GRASS = (134, 22, 119)             # Dark purple background
COLOR_GRASS_DARK = (87, 12, 77)        # Darker purple variation
COLOR_GRASS_LIGHT = (0, 20, 68)       # Lighter purple variation
COLOR_ROAD = (22, 22, 30)              # Very dark asphalt with blue tint
COLOR_ROAD_LIGHT = (28, 28, 38)        # Asphalt texture variation
COLOR_ROAD_LINE = (0, 255, 200)        # Neon cyan road edge line
COLOR_GRAVEL = (30, 25, 40)            # Dark purple gravel
COLOR_KERB = (100, 210, 235)           # Light cyan/sky blue kerb
COLOR_FINISH_CHECKER_1 = (255, 255, 255)
COLOR_FINISH_CHECKER_2 = (15, 15, 20)

# Checkpoint marker colors
COLOR_CHECKPOINT = (0, 255, 180)        # Neon cyan-green
COLOR_CHECKPOINT_GLOW = (0, 200, 255)   # Electric blue glow

# Kart entity colors
COLOR_KART_BODY = (0, 90, 200)         # Racing blue
COLOR_KART_ACCENT = (255, 255, 255)    # White accent stripe
COLOR_KART_WHEEL = (30, 30, 30)        # Dark rubber
COLOR_KART_WHEEL_RIM = (160, 160, 170) # Alloy rim
COLOR_DRIVER_HELMET = (220, 220, 225)  # White/silver helmet
COLOR_DRIVER_VISOR = (40, 40, 45)      # Dark visor

# Heads-Up Display (HUD) colors
COLOR_HUD_TEXT = (0, 255, 200)          # Neon cyan text
COLOR_HUD_ACCENT = (255, 0, 100)       # Neon pink accent
COLOR_HUD_STROKE = (0, 0, 0)
COLOR_HUD_BG = (10, 10, 20, 180)       # Dark semi-transparent panel

# Font path
FONT_PATH = os.path.join("data", "Orbitron.ttf")