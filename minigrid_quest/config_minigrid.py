from pathlib import Path

BASE=Path(__file__).parent

MODEL_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"

LEVELS = [
    {
        "name": "Level 1 - Empty Room 5x5",
        "env_id": "MiniGrid-Empty-5x5-v0",
        "model_path": MODEL_DIR / "ppo_empty5x5_v2",
        "timesteps": 50_000,
        "seed": 11,
        "target_type": "goal",
    },
    {
        "name": "Level 2 - Empty Room 8x8",
        "env_id": "MiniGrid-Empty-8x8-v0",
        "model_path": MODEL_DIR / "ppo_empty8x8_v2",
        "timesteps": 100_000,
        "seed": 22,
        "target_type": "goal",
    },
    {
        "name": "Level 3 - Lava Gap 5x5",
        "env_id": "MiniGrid-LavaGapS5-v0",
        "model_path": MODEL_DIR / "ppo_lavagap5x5_v2",
        "timesteps": 250_000,
        "seed": 33,
        "target_type": "goal",
    },
    {
        "name": "Level 4 - Lava Gap 6x6",
        "env_id": "MiniGrid-LavaGapS6-v0",
        "model_path": MODEL_DIR / "ppo_lavagap6x6_v2",
        "timesteps": 400_000,
        "seed": 44,
        "target_type": "goal",
    },
    {
        "name": "Level 5 - Door Key 5x5",
        "env_id": "MiniGrid-DoorKey-5x5-v0",
        "model_path": MODEL_DIR / "ppo_doorkey5x5_v2",
        "timesteps": 500_000,
        "seed": 55,
        "target_type": "key_door_goal",
    },
    {
        "name": "Level 6 - Door Key 6x6",
        "env_id": "MiniGrid-DoorKey-6x6-v0",
        "model_path": MODEL_DIR / "ppo_doorkey6x6_v2",
        "timesteps": 800_000,
        "seed": 66,
        "target_type": "key_door_goal",
    },
    {
        "name": "Level 7 - Door Key 8x8 Final Challenge",
        "env_id": "MiniGrid-DoorKey-8x8-v0",
        "model_path": MODEL_DIR / "ppo_doorkey8x8_v2",
        "timesteps": 1_000_000,
        "seed": 77,
        "target_type": "key_door_goal",
    },
        {
        "name": "Level 8 - Lava Gap 8x8 Final Challenge",
        "env_id": "MiniGrid-LavaGapS7-v0",
        "model_path": MODEL_DIR / "ppo_lavagap8x8_v2",
        "timesteps": 800_000,
        "seed": 88,
        "target_type": "goal",
    },
        {
        "name": "Level 9 - Door Key 16x16 Mega Challenge",
        "env_id": "MiniGrid-DoorKey-16x16-v0",
        "model_path": MODEL_DIR / "ppo_doorkey16x16_v2",
        "timesteps": 1_500_000,
        "seed": 111,
        "target_type": "key_door_goal",
    },
    {
        "name": "Level 10 - Four Rooms Maze",
        "env_id": "MiniGrid-FourRooms-v0",
        "model_path": MODEL_DIR / "ppo_fourrooms_v2",
        "timesteps": 700_000,
        "seed": 88,
        "target_type": "goal",
    },
    {
        "name": "Level 11 - Simple Crossing 9x9",
        "env_id": "MiniGrid-SimpleCrossingS9N1-v0",
        "model_path": MODEL_DIR / "ppo_simplecrossing9_v2",
        "timesteps": 700_000,
        "seed": 99,
        "target_type": "goal",
    },
    {
        "name": "Level 12 - Hard Maze 15x15",
        "env_id": "MiniGrid-HardMaze-15x15-v0",
        "model_path": MODEL_DIR / "ppo_hardmaze15x15_v2",
        "timesteps": 800_000,
        "seed": 150,
        "target_type": "goal",
    },
    {
        "name": "Level 13 - Lava Maze 15x15",
        "env_id": "MiniGrid-LavaMaze-15x15-v0",
        "model_path": MODEL_DIR / "ppo_lavamaze15x15_v2",
        "timesteps": 1_000_000,
        "seed": 151,
        "target_type": "goal",
        "planner_start": True,
    },
    
]

EVAL_EPISODES = 10
SLEEP_TIME = 0.12
MAX_STEPS_PER_ATTEMPT = 250
MAX_ATTEMPTS_PER_LEVEL = 3