import gymnasium as gym
from gymnasium import spaces
import numpy as np


SCREEN_W = 640
SCREEN_H = 480
PADDLE_W = 12
PADDLE_H = 60
BALL_SIZE = 10
PADDLE_SPEED = 5
BALL_SPEED_INIT = 4
MAX_BALL_SPEED = 10
FPS = 60
MAX_STEPS = 2000


class PongEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super().__init__()

        self.action_space = spaces.Discrete(3)  # 0=stay 1=up 2=down
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1, -1, 0, 0], dtype=np.float32),
            high=np.array([1, 1,  1,  1, 1, 1], dtype=np.float32),
        )

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.current_step = 0
        self._reset_state()

    def _reset_state(self):
        self.ball_x = SCREEN_W / 2
        self.ball_y = SCREEN_H / 2
        angle = self.np_random.uniform(-np.pi / 4, np.pi / 4)
        direction = self.np_random.choice([-1, 1])
        self.ball_vx = direction * BALL_SPEED_INIT * np.cos(angle)
        self.ball_vy = BALL_SPEED_INIT * np.sin(angle)

        self.agent_y = SCREEN_H / 2 - PADDLE_H / 2
        self.opponent_y = SCREEN_H / 2 - PADDLE_H / 2

        self.agent_score = 0
        self.opponent_score = 0

    def _get_obs(self):
        return np.array([
            self.ball_x / SCREEN_W,
            self.ball_y / SCREEN_H,
            self.ball_vx / MAX_BALL_SPEED,
            self.ball_vy / MAX_BALL_SPEED,
            self.agent_y / (SCREEN_H - PADDLE_H),
            self.opponent_y / (SCREEN_H - PADDLE_H),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._reset_state()
        return self._get_obs(), {}

    def _move_opponent(self):
        center = self.opponent_y + PADDLE_H / 2
        difficulty = min(1.0, self.current_step / 1_000_000)
        speed = PADDLE_SPEED * (0.75 + 0.25 * difficulty)
        if center < self.ball_y - 5:
            self.opponent_y += speed
        elif center > self.ball_y + 5:
            self.opponent_y -= speed
        self.opponent_y = float(np.clip(self.opponent_y, 0, SCREEN_H - PADDLE_H))

    def step(self, action):
        self.current_step += 1

        if action == 1:
            self.agent_y -= PADDLE_SPEED
        elif action == 2:
            self.agent_y += PADDLE_SPEED
        self.agent_y = float(np.clip(self.agent_y, 0, SCREEN_H - PADDLE_H))

        self._move_opponent()

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        reward = 0.0
        terminated = False

        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_vy = abs(self.ball_vy)
        elif self.ball_y >= SCREEN_H - BALL_SIZE:
            self.ball_y = SCREEN_H - BALL_SIZE
            self.ball_vy = -abs(self.ball_vy)

        agent_paddle_x = SCREEN_W - PADDLE_W - 10
        if (
            agent_paddle_x <= self.ball_x + BALL_SIZE
            and self.ball_x + BALL_SIZE <= agent_paddle_x + PADDLE_W + 6
            and self.agent_y <= self.ball_y + BALL_SIZE
            and self.ball_y <= self.agent_y + PADDLE_H
            and self.ball_vx > 0
        ):
            self.ball_vx = -abs(self.ball_vx) * 1.05
            self.ball_vx = max(self.ball_vx, -MAX_BALL_SPEED)
            offset = (self.ball_y - (self.agent_y + PADDLE_H / 2)) / (PADDLE_H / 2)
            self.ball_vy += offset * 1.5
            # reward per hit scos — AI invata sa marcheze, nu sa tina mingea in joc

        opponent_paddle_x = 10
        if (
            opponent_paddle_x <= self.ball_x
            and self.ball_x <= opponent_paddle_x + PADDLE_W + 6
            and self.opponent_y <= self.ball_y + BALL_SIZE
            and self.ball_y <= self.opponent_y + PADDLE_H
            and self.ball_vx < 0
        ):
            self.ball_vx = abs(self.ball_vx) * 1.05
            self.ball_vx = min(self.ball_vx, MAX_BALL_SPEED)
            offset = (self.ball_y - (self.opponent_y + PADDLE_H / 2)) / (PADDLE_H / 2)
            self.ball_vy += offset * 1.5

        self.ball_vy = float(np.clip(self.ball_vy, -MAX_BALL_SPEED, MAX_BALL_SPEED))

        if self.ball_x > SCREEN_W:
            reward -= 5.0
            self.opponent_score += 1
            terminated = True
        elif self.ball_x < 0:
            reward += 5.0
            self.agent_score += 1
            terminated = True

        truncated = self.current_step >= MAX_STEPS
        #Penalizare pentru match fara gol scos — AI invata sa marcheze, anti infinite loop
        #if truncated and not terminated:
            #reward -= 1.0  # penalizare pentru match fara gol

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        try:
            import pygame
        except ImportError:
            raise ImportError("pip install pygame")

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
                pygame.display.set_caption("Pong — Human vs AI")
            else:
                self.screen = pygame.Surface((SCREEN_W, SCREEN_H))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("monospace", 36)

        self.screen.fill((20, 20, 20))

        for y in range(0, SCREEN_H, 20):
            pygame.draw.rect(self.screen, (60, 60, 60), (SCREEN_W // 2 - 1, y, 2, 10))

        pygame.draw.rect(self.screen, (255, 255, 255),
                         (10, int(self.opponent_y), PADDLE_W, PADDLE_H))
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (SCREEN_W - PADDLE_W - 10, int(self.agent_y), PADDLE_W, PADDLE_H))
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (int(self.ball_x), int(self.ball_y), BALL_SIZE, BALL_SIZE))

        opp_text = self.font.render(str(self.opponent_score), True, (200, 200, 200))
        agt_text = self.font.render(str(self.agent_score), True, (200, 200, 200))
        self.screen.blit(opp_text, (SCREEN_W // 2 - 60, 10))
        self.screen.blit(agt_text, (SCREEN_W // 2 + 36, 10))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(FPS)
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None
