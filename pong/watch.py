"""
Pong AI viewer
  --mode single   : un singur model vs bot
  --mode double   : doua modele diferite fata in fata
  --left  <path>  : model stanga  (doar la double)
  --right <path>  : model dreapta (default: best_model.zip)
  ESC / Q         : quit
"""
import argparse
import os
import glob
import pygame
import numpy as np
from stable_baselines3 import PPO
from pong_env import PongEnv, SCREEN_W, SCREEN_H, PADDLE_SPEED, FPS

DEFAULT_MODEL = "./pong/models/best_model.zip"


def find_oldest_checkpoint():
    ckpts = sorted(glob.glob("./pong/models/pong_ppo_*_steps.zip"))
    return ckpts[0] if ckpts else None


def mirror_obs(obs):
    return np.array([
        1.0 - obs[0], obs[1], -obs[2], obs[3], obs[5], obs[4],
    ], dtype=np.float32)


class AiVsAiEnv(PongEnv):
    def __init__(self):
        super().__init__(render_mode="human")
        self.opponent_action = 0

    def set_opponent_action(self, action: int):
        self.opponent_action = action

    def _move_opponent(self):
        if self.opponent_action == 1:
            self.opponent_y -= PADDLE_SPEED
        elif self.opponent_action == 2:
            self.opponent_y += PADDLE_SPEED
        self.opponent_y = float(np.clip(self.opponent_y, 0, 480 - 60))


def run_single(model_right_path):
    """Un model vs bot-ul rule-based din env."""
    model = PPO.load(model_right_path,device="cpu")
    env = PongEnv(render_mode="human")
    obs, _ = env.reset()
    pygame.init()
    clock = pygame.time.Clock()
    score_bot, score_ai = 0, 0
    running = True
    print(f"Single mode — AI: {model_right_path}")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))

        if terminated or truncated:
            if reward > 0:
                score_ai += 1
            elif reward < -1:
                score_bot += 1
            print(f"Bot {score_bot} — AI {score_ai}")
            env.agent_score = score_ai
            env.opponent_score = score_bot
            obs, _ = env.reset()
            env.agent_score = score_ai
            env.opponent_score = score_bot

        clock.tick(FPS)
    env.close()


def run_double(model_left_path, model_right_path):
    """Doua modele diferite fata in fata."""
    model_left  = PPO.load(model_left_path,device="cpu")
    model_right = PPO.load(model_right_path,device="cpu")
    env = AiVsAiEnv()
    obs, _ = env.reset()
    pygame.init()
    clock = pygame.time.Clock()
    score_left, score_right = 0, 0
    running = True
    print(f"Double mode — Left: {model_left_path}  Right: {model_right_path}")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        action_right, _ = model_right.predict(obs, deterministic=True)
        action_left,  _ = model_left.predict(mirror_obs(obs), deterministic=True)
        env.set_opponent_action(int(action_left))
        obs, reward, terminated, truncated, _ = env.step(int(action_right))

        if terminated or truncated:
            if reward > 0:
                score_right += 1
            elif reward < -1:
                score_left += 1
            print(f"Left {score_left} — Right {score_right}")
            env.agent_score    = score_right
            env.opponent_score = score_left
            obs, _ = env.reset()
            env.agent_score    = score_right
            env.opponent_score = score_left

        clock.tick(FPS)
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  type=str, default="single",
                        choices=["single", "double"])
    parser.add_argument("--right", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--left",  type=str, default=None)
    args = parser.parse_args()

    if args.mode == "single":
        run_single(args.right)
    else:
        left = args.left or find_oldest_checkpoint()
        if not left:
            print("Nu am gasit un model pentru stanga. Seteaza --left <path>.")
            return
        run_double(left, args.right)


if __name__ == "__main__":
    main()
