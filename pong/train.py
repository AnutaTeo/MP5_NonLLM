import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from pong_env import PongEnv

MODEL_DIR = "./pong/models/"
LOG_DIR = "./logs/"
TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 32


def main():
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = make_vec_env(PongEnv, n_envs=N_ENVS)
    eval_env = make_vec_env(PongEnv, n_envs=1)

    device = "cpu" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000 // N_ENVS,
        save_path=MODEL_DIR,
        name_prefix="pong_ppo",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=200_000 // N_ENVS,
        n_eval_episodes=10,
        deterministic=True,
    )

    model = PPO(
        policy="MlpPolicy",   # pong_env uses a vector obs, not pixels
        env=env,
        device=device,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        tensorboard_log=LOG_DIR,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    model.save(MODEL_DIR + "pong_ppo_final")
    print("Done. Model saved.")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
