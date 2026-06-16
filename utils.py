import gymnasium as gym
import numpy as np
from pathlib import Path
from gymnasium import spaces
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack, VecTransposeImage


class CustomDiscreteCarRacingActions(gym.ActionWrapper):
    """
    CarRacing normally has continuous actions:
    [steering, gas, brake]

    This wrapper gives the agent a smaller set of useful combined actions.
    This is easier to learn than fully continuous control.
    """

    def __init__(self, env):
        super().__init__(env)

        self.actions = [
            np.array([0.0, 0.6, 0.0], dtype=np.float32),    # 0: straight + gas
            np.array([-0.4, 0.5, 0.0], dtype=np.float32),   # 1: soft left + gas
            np.array([0.4, 0.5, 0.0], dtype=np.float32),    # 2: soft right + gas
            np.array([-0.8, 0.3, 0.0], dtype=np.float32),   # 3: hard left + light gas
            np.array([0.8, 0.3, 0.0], dtype=np.float32),    # 4: hard right + light gas
            np.array([0.0, 0.0, 0.6], dtype=np.float32),    # 5: brake
            np.array([0.0, 0.0, 0.0], dtype=np.float32),    # 6: do nothing
        ]

        self.action_space = spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[int(action)]


def ensure_dirs():
    for folder in ["models", "results", "videos", "logs"]:
        Path(folder).mkdir(exist_ok=True)


def make_carracing_env(
    env_id="CarRacing-v3",
    render_mode=None,
    use_custom_discrete_actions=True
):
    """
    Creates CarRacing.
    We keep continuous=True internally, then optionally wrap it with our own
    discrete action set.
    """
    try:
        env = gym.make(env_id, render_mode=render_mode, continuous=True)
    except Exception as e:
        if env_id == "CarRacing-v3":
            print("Could not create CarRacing-v3. Falling back to CarRacing-v2.")
            print("Original error:", e)
            env = gym.make("CarRacing-v2", render_mode=render_mode, continuous=True)
        else:
            raise

    if use_custom_discrete_actions:
        env = CustomDiscreteCarRacingActions(env)

    return env


def make_vec_carracing_env(
    env_id="CarRacing-v3",
    render_mode=None,
    use_custom_discrete_actions=True,
    frame_stack=4,
    n_envs=1
):
    def make_one_env():
        def _init():
            return make_carracing_env(
                env_id=env_id,
                render_mode=render_mode,
                use_custom_discrete_actions=use_custom_discrete_actions
            )
        return _init

    env_fns = [make_one_env() for _ in range(n_envs)]

    if n_envs > 1 and render_mode is None:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    env = VecTransposeImage(env)

    if frame_stack and frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack, channels_order="first")

    return env