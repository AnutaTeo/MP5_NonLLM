import gymnasium as gym
from gymnasium.envs.registration import register

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall, Lava
from minigrid.core.mission import MissionSpace


class LayoutMazeEnv(MiniGridEnv):
    """
    Custom MiniGrid maze environment built from a text layout.

    Symbols:
      # = wall
      . = empty cell
      A = agent start
      G = goal
      L = lava
    """

    LAYOUT = []

    def __init__(self, **kwargs):
        self.layout = self.LAYOUT
        self.size = len(self.layout)

        mission_space = MissionSpace(
            mission_func=lambda: "navigate through the maze and reach the green goal"
        )

        super().__init__(
            mission_space=mission_space,
            grid_size=self.size,
            max_steps=10 * self.size * self.size,
            see_through_walls=False,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        for y, row in enumerate(self.layout):
            for x, cell in enumerate(row):
                if cell == "#":
                    self.grid.set(x, y, Wall())

                elif cell == "G":
                    self.grid.set(x, y, Goal())

                elif cell == "L":
                    self.grid.set(x, y, Lava())

                elif cell == "A":
                    self.agent_pos = (x, y)
                    self.agent_dir = 0  # facing right

        self.mission = "navigate through the maze and reach the green goal"


class HardMaze15x15Env(LayoutMazeEnv):
    LAYOUT = [
        "###############",
        "#A..........#G#",
        "#.#########.#.#",
        "#.#.......#.#.#",
        "#.#.#####.#.#.#",
        "#.#.#...#.#.#.#",
        "#.#.#.#.#.#.#.#",
        "#.#.#.#...#.#.#",
        "#.#.#.#####.#.#",
        "#.#.#.......#.#",
        "#.#.#########.#",
        "#.#...........#",
        "#.#############",
        "#.............#",
        "###############",
    ]


class LavaMaze15x15Env(LayoutMazeEnv):
    LAYOUT = [
        "###############",
        "#A............#",
        "####L########.#",
        "#.............#",
        "#.######L######",
        "#.............#",
        "######L######.#",
        "#.............#",
        "#.########L####",
        "#.............#",
        "###L#########.#",
        "#.............#",
        "#.#####L#######",
        "#............G#",
        "###############",
    ]


def safe_register(env_id, entry_point):
    if env_id not in gym.envs.registry:
        register(
            id=env_id,
            entry_point=entry_point,
        )


safe_register(
    "MiniGrid-HardMaze-15x15-v0",
    "custom_minigrid_envs:HardMaze15x15Env",
)

safe_register(
    "MiniGrid-LavaMaze-15x15-v0",
    "custom_minigrid_envs:LavaMaze15x15Env",
)