from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from config import ENV_ID, FRAME_STACK, MODEL_PATH, USE_CUSTOM_DISCRETE_ACTIONS, N_ENVS
from utils import make_vec_carracing_env, ensure_dirs


# Dacă ai deja 300k și vrei să ajungi la ~500k,
# mai antrenezi încă 200k.
EXTRA_TIMESTEPS = 200_000


def main():
    ensure_dirs()

    env = make_vec_carracing_env(
        env_id=ENV_ID,
        render_mode=None,
        use_custom_discrete_actions=USE_CUSTOM_DISCRETE_ACTIONS,
        frame_stack=FRAME_STACK,
        n_envs=N_ENVS
    )

    model = PPO.load(str(MODEL_PATH), env=env)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./models/",
        name_prefix="ppo_carracing_continued"
    )

    model.learn(
        total_timesteps=EXTRA_TIMESTEPS,
        callback=checkpoint_callback,
        reset_num_timesteps=False,
        progress_bar=False
    )

    model.save(str(MODEL_PATH))
    print(f"Continued model saved to {MODEL_PATH}.zip")

    env.close()


if __name__ == "__main__":
    main()