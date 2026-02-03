import os

# Track representation values
TRACK_FINISH_VALUE = 2
TRACK_CURB_VALUE = 0
TRACK_ROAD_VALUE = 1
TRACK_OFFROAD_VALUE = -1
TRACK_UNKNOWN_VALUE = -2

# Rendering
RENDER_FPS = 10
WINDOW_SIZE_PX = 800

# Agent view / track parameters
VIEW_SIZE = 7
ROAD_WIDTH = 5
NUM_CHECKPOINTS = 7
ACTION_SPACE_SIZE = 4

# Observation space bounds
OBS_LOW = TRACK_UNKNOWN_VALUE
OBS_HIGH = TRACK_FINISH_VALUE

# Rewards
OFFROAD_REWARD = -1000
STEP_PENALTY = -1
CHECKPOINT_REWARD = 1000

# Track files
DEFAULT_TRACK_FILENAME = "track_imola.csv"
DEFAULT_TRACK_RELATIVE_PATH = os.path.join("data", "tracks", DEFAULT_TRACK_FILENAME)
DEFAULT_DISTANCE_DIRECTION = "destra"


def get_default_track_path():
    return os.path.join(os.getcwd(), DEFAULT_TRACK_RELATIVE_PATH)
