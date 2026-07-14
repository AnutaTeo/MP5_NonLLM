import numpy as np
from stable_baselines3 import PPO
from config_minigrid import LEVELS, EVAL_EPISODES, RESULTS_DIR
from minigrid_env_utils import make_minigrid_env


def evaluate_random(env_id, episodes):
    rewards = []
    successes = 0

    for _ in range(episodes):
        env = make_minigrid_env(env_id)
        obs, info = env.reset()

        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        if total_reward > 0:
            successes += 1

        rewards.append(total_reward)
        env.close()

    return np.mean(rewards), successes


def evaluate_trained(env_id, model_path, episodes):
    env = make_minigrid_env(env_id)
    model = PPO.load(str(model_path), env=env)

    rewards = []
    successes = 0

    for _ in range(episodes):
        obs, info = env.reset()

        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward

        if total_reward > 0:
            successes += 1

        rewards.append(total_reward)

    env.close()
    return np.mean(rewards), successes


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    lines = []
    lines.append("MiniGrid Quest - Evaluation")
    lines.append("===========================")

    for level in LEVELS:
        env_id = level["env_id"]
        model_path = level["model_path"]

        random_reward, random_success = evaluate_random(env_id, EVAL_EPISODES)
        trained_reward, trained_success = evaluate_trained(env_id, model_path, EVAL_EPISODES)

        lines.append("")
        lines.append(level["name"])
        lines.append(f"Environment: {env_id}")
        lines.append(f"Random average reward: {random_reward:.3f}")
        lines.append(f"Trained average reward: {trained_reward:.3f}")
        lines.append(f"Random success rate: {random_success}/{EVAL_EPISODES}")
        lines.append(f"Trained success rate: {trained_success}/{EVAL_EPISODES}")

    output = "\n".join(lines)
    print(output)

    with open(RESULTS_DIR / "evaluation_levels.txt", "w") as f:
        f.write(output)


if __name__ == "__main__":
    main()