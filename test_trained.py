from stable_baselines3 import PPO
from config import ENV_ID, MODEL_PATH, FRAME_STACK, USE_CUSTOM_DISCRETE_ACTIONS
from utils import make_vec_carracing_env


def main():
    env = make_vec_carracing_env(
        env_id=ENV_ID,
        render_mode="human",
        use_custom_discrete_actions=USE_CUSTOM_DISCRETE_ACTIONS,
        frame_stack=FRAME_STACK,
        n_envs=1
    )

    model = PPO.load(str(MODEL_PATH), env=env)

    obs = env.reset()
    total_reward = 0

    for step in range(9000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        total_reward += reward[0]

        if done[0]:
            print(f"Episode ended. Total reward: {total_reward:.2f}")
            total_reward = 0
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()