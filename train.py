from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from config import ENV_ID, TOTAL_TIMESTEPS, FRAME_STACK, MODEL_PATH, USE_CUSTOM_DISCRETE_ACTIONS, N_ENVS
from utils import make_vec_carracing_env, ensure_dirs


def main():
    ensure_dirs()

    env = make_vec_carracing_env(
        env_id=ENV_ID,
        render_mode=None,
        use_custom_discrete_actions=USE_CUSTOM_DISCRETE_ACTIONS,
        frame_stack=FRAME_STACK,
        n_envs=N_ENVS
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./models/",
        name_prefix="ppo_carracing_checkpoint"
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,

        #sa nu exploreze exagerat ca se invartea continuu
        ent_coef=0.001,

        tensorboard_log="./logs/"
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=False
    )

    model.save(str(MODEL_PATH))
    print(f"Model saved to {MODEL_PATH}.zip")

    env.close()


if __name__ == "__main__":
    main()