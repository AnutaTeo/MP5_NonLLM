from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from config import ENV_ID, TOTAL_TIMESTEPS, FRAME_STACK, MODEL_PATH, USE_CUSTOM_DISCRETE_ACTIONS, N_ENVS
from utils import make_vec_carracing_env, ensure_dirs
import torch

# Check before training
#print(torch.cuda.is_available())        # checks if cuda is available(nvidia gpu is available)
#print(torch.cuda.get_device_name(0))    # GPU name

def main():
    ensure_dirs()

    env = make_vec_carracing_env(
        env_id=ENV_ID,
        render_mode=None,
        use_custom_discrete_actions=USE_CUSTOM_DISCRETE_ACTIONS,
        frame_stack=FRAME_STACK,
        n_envs=N_ENVS
    )

    env.reset()  # Reset the environment before training
    env=VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0,clip_reward=10.0)
    env= VecMonitor(env)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./models/",
        name_prefix="ppo_carracing_checkpoint"
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=4096,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        vf_coef=0.5,         # add this, scales value function loss contribution
        clip_range=0.1,

        #sa nu exploreze exagerat ca se invartea continuu
        ent_coef=0.1,

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