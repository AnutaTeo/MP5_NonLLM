import gymnasium as gym
import minigrid

# Important: this registers our custom MiniGrid environments, if the file exists.
try:
    import custom_minigrid_envs
except ImportError:
    pass


def main():
    envs = sorted([
        env_id for env_id in gym.envs.registry.keys()
        if "MiniGrid" in env_id
    ])

    keywords = [
        "Empty",
        "Lava",
        "DoorKey",
        "Crossing",
        "FourRooms",
        "Maze",
        "Unlock",
        "MultiRoom",
        "TwoDoors",
        "HardMaze",
        "LavaMaze",
    ]

    for env_id in envs:
        if any(keyword in env_id for keyword in keywords):
            print(env_id)


if __name__ == "__main__":
    main()