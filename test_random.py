from config import ENV_ID, USE_CUSTOM_DISCRETE_ACTIONS
from utils import make_carracing_env


def main():
    env = make_carracing_env(
        ENV_ID,
        render_mode="human",
        use_custom_discrete_actions=USE_CUSTOM_DISCRETE_ACTIONS
    )

    obs, info = env.reset()
    total_reward = 0

    for step in range(2000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 100 == 0:
            print(f"Step {step}, reward so far: {total_reward:.2f}")

        if terminated or truncated:
            print(f"Episode ended. Total reward: {total_reward:.2f}")
            total_reward = 0
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()