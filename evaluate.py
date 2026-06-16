import numpy as np
from stable_baselines3 import PPO
from config import ENV_ID, MODEL_PATH, FRAME_STACK, EVAL_EPISODES
from utils import make_carracing_env, make_vec_carracing_env, ensure_dirs


def evaluate_random(episodes=5):
    rewards = []

    for episode in range(episodes):
        env = make_carracing_env(ENV_ID, render_mode=None, continuous=True)
        obs, info = env.reset()
        total_reward = 0.0

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        env.close()

    return rewards


def evaluate_trained(episodes=5):
    env = make_vec_carracing_env(
        env_id=ENV_ID,
        render_mode=None,
        continuous=True,
        frame_stack=FRAME_STACK
    )

    model = PPO.load(str(MODEL_PATH), env=env)
    rewards = []

    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0.0
        done = [False]

        while not done[0]:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])

        rewards.append(total_reward)

    env.close()
    return rewards


def main():
    ensure_dirs()

    random_rewards = evaluate_random(EVAL_EPISODES)
    trained_rewards = evaluate_trained(EVAL_EPISODES)

    random_mean = np.mean(random_rewards)
    trained_mean = np.mean(trained_rewards)

    output = []
    output.append("RaceAI evaluation results")
    output.append("-------------------------")
    output.append(f"Random agent rewards: {random_rewards}")
    output.append(f"Trained PPO agent rewards: {trained_rewards}")
    output.append(f"Random agent average reward: {random_mean:.2f}")
    output.append(f"Trained PPO agent average reward: {trained_mean:.2f}")

    text = "\n".join(output)
    print(text)

    with open("results/evaluation.txt", "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
