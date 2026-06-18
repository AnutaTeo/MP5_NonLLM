"""
AI vs AI Pong — doua modele diferite
  Model stanga: checkpoint mai vechi (model_weak)
  Model dreapta: best_model (model_strong)
  ESC / Q — quit
"""
import pygame
import numpy as np
from stable_baselines3 import PPO
from pong_env import PongEnv, SCREEN_W, SCREEN_H, PADDLE_SPEED, FPS

MODEL_LEFT  = "./pong/models/pong_ppo_3125_steps.zip"   # checkpoint mai vechi
MODEL_RIGHT = "./pong/models/best_model.zip"             # cel mai bun


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
        import numpy as np
        self.opponent_y = float(np.clip(self.opponent_y, 0, 480 - 60))


def mirror_obs(obs):
    """Intoarce obs-ul pentru perspectiva paddle-ului stang."""
    return np.array([
        1.0 - obs[0],  # ball_x inversat
        obs[1],         # ball_y la fel
        -obs[2],        # ball_vx inversat
        obs[3],         # ball_vy la fel
        obs[5],         # paddle stanga devine agent
        obs[4],         # paddle dreapta devine opponent
    ], dtype=np.float32)


def main():
    import os, glob

    # gaseste automat cel mai vechi checkpoint daca MODEL_LEFT nu exista
    left_path = MODEL_LEFT
    if not os.path.exists(left_path) and not os.path.exists(left_path + ".zip"):
        checkpoints = sorted(glob.glob("./pong/models/pong_ppo_*_steps.zip"))
        if len(checkpoints) < 2:
            print("Nu am gasit doua modele diferite. Antreneaza mai mult sau seteaza manual MODEL_LEFT.")
            return
        left_path = checkpoints[0]   # cel mai vechi
        right_path = checkpoints[-1] # cel mai nou
        print(f"Stanga (slab):  {left_path}")
        print(f"Dreapta (bun):  {right_path}")
    else:
        right_path = MODEL_RIGHT
        print(f"Stanga:  {left_path}")
        print(f"Dreapta: {right_path}")

    model_left  = PPO.load(left_path)
    model_right = PPO.load(right_path)

    env = AiVsAiEnv()
    obs, _ = env.reset()

    pygame.init()
    clock = pygame.time.Clock()

    score_left  = 0
    score_right = 0
    running = True

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
            print(f"Slab(stanga) {score_left} — Bun(dreapta) {score_right}")
            env.agent_score    = score_right
            env.opponent_score = score_left
            obs, _ = env.reset()
            env.agent_score    = score_right
            env.opponent_score = score_left

        clock.tick(FPS)

    env.close()


if __name__ == "__main__":
    main()
