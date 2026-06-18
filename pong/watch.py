import pygame
import torch
from stable_baselines3 import PPO
from pong_env import (
    PongEnv, SCREEN_W, SCREEN_H,
    PADDLE_W, PADDLE_H, BALL_SIZE, PADDLE_SPEED, FPS
)
import numpy as np

MODEL_PATH = "./pong/models/best_model.zip"


class AiVsAiEnv(PongEnv):
    """Ambii paddli controlati de acelasi model."""

    def __init__(self):
        super().__init__(render_mode="human")
        self.opponent_action = 0

    def set_opponent_action(self, action: int):
        self.opponent_action = action

    def _move_opponent(self):
        # suprascrie bot-ul rule-based cu actiunea AI
        if self.opponent_action == 1:
            self.opponent_y -= PADDLE_SPEED
        elif self.opponent_action == 2:
            self.opponent_y += PADDLE_SPEED
        self.opponent_y = float(np.clip(self.opponent_y, 0, SCREEN_H - PADDLE_H))


def main():
    model = PPO.load(MODEL_PATH, device="cpu")
    env = AiVsAiEnv()
    obs, _ = env.reset()

    pygame.init()
    clock = pygame.time.Clock()

    score_left = 0
    score_right = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        # obs pentru paddle stanga — obs e din perspectiva paddle-ului drept
        # inversam pentru a simula perspectiva opusa
        obs_left = np.array([
            1.0 - obs[0],   # ball_x inversat
            obs[1],          # ball_y la fel
            -obs[2],         # ball_vx inversat
            obs[3],          # ball_vy la fel
            obs[5],          # paddle stanga devine "agent"
            obs[4],          # paddle dreapta devine "opponent"
        ], dtype=np.float32)

        action_right, _ = model.predict(obs, deterministic=True)
        action_left, _ = model.predict(obs_left, deterministic=True)

        env.set_opponent_action(int(action_left))
        obs, reward, terminated, truncated, _ = env.step(int(action_right))

        if terminated or truncated:
            if reward > 0:
                score_right += 1
            elif reward < 0:
                score_left += 1
            print(f"AI_left {score_left} — AI_right {score_right}")
            env.agent_score = score_right
            env.opponent_score = score_left
            obs, _ = env.reset()
            env.agent_score = score_right
            env.opponent_score = score_left

        clock.tick(FPS)

    env.close()


if __name__ == "__main__":
    main()