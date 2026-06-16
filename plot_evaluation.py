import matplotlib.pyplot as plt
import numpy as np
from evaluate import evaluate_random, evaluate_trained
from config import EVAL_EPISODES
from utils import ensure_dirs


def main():
    ensure_dirs()

    random_rewards = evaluate_random(EVAL_EPISODES)
    trained_rewards = evaluate_trained(EVAL_EPISODES)

    labels = ["Random Agent", "Trained PPO Agent"]
    means = [np.mean(random_rewards), np.mean(trained_rewards)]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, means)
    plt.ylabel("Average reward")
    plt.title("Random Agent vs Trained PPO Agent")
    plt.tight_layout()
    plt.savefig("results/random_vs_trained.png")
    plt.show()

    print("Saved plot to results/random_vs_trained.png")


if __name__ == "__main__":
    main()
