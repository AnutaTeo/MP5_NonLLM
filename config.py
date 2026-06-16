from pathlib import Path

ENV_ID = "CarRacing-v3"

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
VIDEOS_DIR = Path("videos")
LOG_DIR = Path("logs")

MODEL_PATH = MODEL_DIR / "ppo_carracing_final"

# Recommended: 300k minimum, 500k for best results
TOTAL_TIMESTEPS = 300000

EVAL_EPISODES = 5
FRAME_STACK = 4

# We use continuous environment, but we transform it into discrete combined actions.
USE_CUSTOM_DISCRETE_ACTIONS = True

#paralelizare
N_ENVS = 4