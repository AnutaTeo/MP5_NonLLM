import gymnasium as gym
import numpy as np
from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper


class FixedSeedWrapper(gym.Wrapper):
    def __init__(self, env, seed=0):
        super().__init__(env)
        self.fixed_seed = seed

    def reset(self, **kwargs):
        kwargs["seed"] = self.fixed_seed
        return self.env.reset(**kwargs)


class SimplifiedActionWrapper(gym.ActionWrapper):
    """
    Reduces MiniGrid actions to useful actions only.

    Normal / Lava levels:
      0 = turn left
      1 = turn right
      2 = move forward

    DoorKey levels:
      0 = turn left
      1 = turn right
      2 = move forward
      3 = pick up
      4 = toggle/open
    """

    def __init__(self, env, target_type="goal"):
        super().__init__(env)
        self.target_type = target_type

        if target_type == "key_door_goal":
            self.original_actions = [0, 1, 2, 3, 5]
        else:
            self.original_actions = [0, 1, 2]

        self.action_space = spaces.Discrete(len(self.original_actions))

    def action(self, action):
        return self.original_actions[int(action)]


class FullObsVectorWrapper(gym.ObservationWrapper):
    """
    Converts full MiniGrid observation into a flat numeric vector.
    """

    def __init__(self, env):
        super().__init__(env)

        image_space = env.observation_space["image"]
        image_size = int(np.prod(image_space.shape))

        # image + direction + carrying flag + door open flag
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(image_size + 3,),
            dtype=np.float32,
        )

    def observation(self, obs):
        image = obs["image"].astype(np.float32).flatten()
        direction = np.array([obs["direction"]], dtype=np.float32)

        base = get_base_env(self.env)

        carrying = 0.0
        if hasattr(base, "carrying") and base.carrying is not None:
            carrying = 1.0

        door_open = 1.0 if is_any_door_open(base) else 0.0

        extra = np.array([direction[0], carrying, door_open], dtype=np.float32)

        return np.concatenate([image, extra])


class MiniGridRewardShapingWrapper(gym.Wrapper):
    """
    Reward shaping for easier and more stable training.
    """

    def __init__(self, env, target_type="goal"):
        super().__init__(env)
        self.target_type = target_type
        self.last_action = None
        self.same_action_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_action = None
        self.same_action_count = 0
        return obs, info

    def step(self, action):
        action = int(action)

        old_pos = self.get_agent_pos()
        old_distance = self.get_current_distance()
        old_carrying = self.is_carrying()
        old_door_open = self.is_door_open()

        obs, reward, terminated, truncated, info = self.env.step(action)

        new_pos = self.get_agent_pos()
        new_distance = self.get_current_distance()
        new_carrying = self.is_carrying()
        new_door_open = self.is_door_open()

        shaped_reward = float(reward)

        # Small step penalty: encourages shorter solutions.
        shaped_reward -= 0.002

        # Encourage moving closer to the current target.
        if old_distance is not None and new_distance is not None:
            if new_distance < old_distance:
                shaped_reward += 0.05
            elif new_distance > old_distance:
                shaped_reward -= 0.02

        # Penalize no position change.
        if old_pos is not None and new_pos is not None and old_pos == new_pos:
            shaped_reward -= 0.02

        # Penalize repeated same action.
        if action == self.last_action:
            self.same_action_count += 1
        else:
            self.same_action_count = 0
            self.last_action = action

        if self.same_action_count > 5:
            shaped_reward -= 0.04

        # DoorKey-specific shaping.
        if self.target_type == "key_door_goal":
            # In simplified action space:
            # 3 = pick up
            # 4 = toggle/open

            # Reward only actual key pickup.
            if action == 3:
                if not old_carrying and new_carrying:
                    shaped_reward += 0.6
                else:
                    shaped_reward -= 0.08

            # Reward only actual door opening.
            if action == 4:
                if not old_door_open and new_door_open:
                    shaped_reward += 0.6
                else:
                    shaped_reward -= 0.06

        # If episode ends without true success, discourage it.
        if terminated and reward <= 0:
            shaped_reward -= 1.0

        return obs, shaped_reward, terminated, truncated, info

    def get_agent_pos(self):
        base = get_base_env(self.env)
        if hasattr(base, "agent_pos"):
            return tuple(int(x) for x in base.agent_pos)
        return None

    def is_carrying(self):
        base = get_base_env(self.env)
        return hasattr(base, "carrying") and base.carrying is not None

    def is_door_open(self):
        base = get_base_env(self.env)
        return is_any_door_open(base)

    def get_current_distance(self):
        base = get_base_env(self.env)

        if not hasattr(base, "agent_pos"):
            return None

        agent_pos = tuple(int(x) for x in base.agent_pos)
        target_pos = find_current_target(base, self.target_type)

        if target_pos is None:
            return None

        return manhattan(agent_pos, target_pos)


def get_base_env(env):
    current = env
    while hasattr(current, "env"):
        current = current.env
    return current


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def is_any_door_open(base_env):
    if not hasattr(base_env, "grid"):
        return False

    for x in range(base_env.grid.width):
        for y in range(base_env.grid.height):
            obj = base_env.grid.get(x, y)
            if obj is not None and obj.type == "door":
                if getattr(obj, "is_open", False):
                    return True

    return False


def find_current_target(base_env, target_type):
    carrying = getattr(base_env, "carrying", None)

    key_pos = None
    door_pos = None
    goal_pos = None
    door_is_open = False

    if not hasattr(base_env, "grid"):
        return None

    for x in range(base_env.grid.width):
        for y in range(base_env.grid.height):
            obj = base_env.grid.get(x, y)

            if obj is None:
                continue

            if obj.type == "key":
                key_pos = (x, y)

            elif obj.type == "door":
                door_pos = (x, y)
                if getattr(obj, "is_open", False):
                    door_is_open = True

            elif obj.type == "goal":
                goal_pos = (x, y)

    if target_type == "key_door_goal":
        if carrying is None and key_pos is not None:
            return key_pos

        if door_pos is not None and not door_is_open:
            return door_pos

        if goal_pos is not None:
            return goal_pos

    return goal_pos


def make_minigrid_env(
    env_id,
    render_mode=None,
    seed=0,
    target_type="goal",
    shaped_reward=True,
):
    env = gym.make(env_id, render_mode=render_mode)

    env = FixedSeedWrapper(env, seed=seed)
    env = FullyObsWrapper(env)

    env = SimplifiedActionWrapper(env, target_type=target_type)

    if shaped_reward:
        env = MiniGridRewardShapingWrapper(env, target_type=target_type)

    env = FullObsVectorWrapper(env)

    return env