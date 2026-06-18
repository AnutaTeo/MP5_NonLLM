from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from config_minigrid import LEVELS, MODEL_DIR, RESULTS_DIR
from minigrid_env_utils import make_minigrid_env


def train_level(level_index):
    level = LEVELS[level_index]

    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    env_id = level["env_id"]
    model_path = level["model_path"]
    total_timesteps = level["timesteps"]
    seed = level["seed"]
    target_type = level["target_type"]

    print(f"Training {level['name']}")
    print(f"Environment: {env_id}")
    print(f"Timesteps: {total_timesteps}")
    print(f"Seed: {seed}")
    print(f"Target type: {target_type}")

    env = make_minigrid_env(
        env_id,
        seed=seed,
        target_type=target_type,
        shaped_reward=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=str(MODEL_DIR),
        name_prefix=f"checkpoint_level_{level_index + 1}",
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=8,
        gamma=0.99,
        ent_coef=0.02,
        tensorboard_log="./logs_minigrid/",
        seed=seed,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    model.save(str(model_path))
    print(f"Saved model to {model_path}.zip")

    env.close()


def main():
    print("MiniGrid Quest - Train Level")
    print("----------------------------")

    for i, level in enumerate(LEVELS):
        print(f"{i + 1}. {level['name']} ({level['env_id']})")

    choice = int(input("Choose level to train: ")) - 1

    if choice < 0 or choice >= len(LEVELS):
        print("Invalid choice.")
        return

    train_level(choice)


if __name__ == "__main__":
    main()