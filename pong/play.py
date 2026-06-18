"""
Human vs AI Pong
  W / S  or  UP / DOWN  — move your paddle (left side)
  ESC / Q                — quit
"""
import pygame
import numpy as np
from stable_baselines3 import PPO
from pong_env import (
    PongEnv, SCREEN_W, SCREEN_H,
    PADDLE_W, PADDLE_H, BALL_SIZE, PADDLE_SPEED, FPS
)

MODEL_PATH = "./pong/models/pong_ppo_final"   # or pong_ppo_final.zip


class HumanVsAIEnv(PongEnv):
    """
    Overrides _move_opponent() so the LEFT paddle is controlled by
    keyboard input instead of the rule-based bot.
    The AI (PPO) still controls the RIGHT paddle via step().
    """

    def __init__(self):
        super().__init__(render_mode="human")
        self.human_action = 0  # 0=stay 1=up 2=down

    def set_human_action(self, action: int):
        self.human_action = action

    def _move_opponent(self):
        if self.human_action == 1:
            self.opponent_y -= PADDLE_SPEED
        elif self.human_action == 2:
            self.opponent_y += PADDLE_SPEED
        self.opponent_y = float(np.clip(self.opponent_y, 0, SCREEN_H - PADDLE_H))


def main():
    model = PPO.load(MODEL_PATH)
    env = HumanVsAIEnv()
    obs, _ = env.reset()
    pygame.init()

    human_score = 0
    ai_score = 0
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        keys = pygame.key.get_pressed()
        human_action = 0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            human_action = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            human_action = 2

        env.set_human_action(human_action)
        ai_action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(ai_action)

        if terminated or truncated:
            if reward > 0:
                ai_score += 1
            else:
                human_score += 1
            print(f"Score  Human {human_score} — AI {ai_score}")
            # injectează scorul în env înainte de reset
            env.agent_score = ai_score
            env.opponent_score = human_score
            obs, _ = env.reset()
            # restaurează după reset
            env.agent_score = ai_score
            env.opponent_score = human_score

        clock.tick(FPS)

    env.close()
    print(f"\nFinal: Human {human_score} — AI {ai_score}")


if __name__ == "__main__":
    main()
