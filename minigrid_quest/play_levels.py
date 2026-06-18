import time
from collections import deque

from stable_baselines3 import PPO

from config_minigrid import (
    LEVELS,
    SLEEP_TIME,
    MAX_STEPS_PER_ATTEMPT,
    MAX_ATTEMPTS_PER_LEVEL,
)
from minigrid_env_utils import make_minigrid_env, get_base_env


ACTION_NAMES = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pick up",
    4: "toggle/open",
}


# MiniGrid directions:
# 0 = right, 1 = down, 2 = left, 3 = up
DIR_TO_VEC = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}


ENABLE_PLANNER_FALLBACK = True

# Dacă PPO repetă aceeași acțiune, activăm planner-ul
MAX_SAME_ACTIONS_BEFORE_PLANNER = 6

# Dacă agentul nu își schimbă poziția, activăm planner-ul
MAX_STUCK_STEPS_BEFORE_PLANNER = 8


def get_agent_position(env):
    base = get_base_env(env)
    if hasattr(base, "agent_pos"):
        return tuple(int(x) for x in base.agent_pos)
    return None


def get_agent_direction(env):
    base = get_base_env(env)
    if hasattr(base, "agent_dir"):
        return int(base.agent_dir)
    return None


def get_front_pos(pos, direction):
    dx, dy = DIR_TO_VEC[direction]
    return pos[0] + dx, pos[1] + dy


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_objects(base):
    key_pos = None
    door_pos = None
    goal_pos = None
    door_open = False

    if not hasattr(base, "grid"):
        return key_pos, door_pos, goal_pos, door_open

    for x in range(base.grid.width):
        for y in range(base.grid.height):
            obj = base.grid.get(x, y)

            if obj is None:
                continue

            if obj.type == "key":
                key_pos = (x, y)

            elif obj.type == "door":
                door_pos = (x, y)
                door_open = getattr(obj, "is_open", False)

            elif obj.type == "goal":
                goal_pos = (x, y)

    return key_pos, door_pos, goal_pos, door_open


def is_carrying_key(base):
    return hasattr(base, "carrying") and base.carrying is not None


def is_cell_passable(base, x, y):
    if not hasattr(base, "grid"):
        return False

    if x < 0 or y < 0 or x >= base.grid.width or y >= base.grid.height:
        return False

    obj = base.grid.get(x, y)

    if obj is None:
        return True

    if obj.type == "wall":
        return False

    if obj.type == "lava":
        return False

    if obj.type == "door":
        return getattr(obj, "is_open", False)

    # Cheia nu este passable. Agentul trebuie să stea lângă ea și să dea pick up.
    if obj.type == "key":
        return False

    if obj.type == "goal":
        return True

    return True


def bfs_next_navigation_action(base, target_kind, target_pos):
    """
    Returnează următoarea acțiune de navigare:
      0 = turn left
      1 = turn right
      2 = move forward

    Pentru key/door:
      agentul trebuie să ajungă lângă obiect și să fie orientat spre el.

    Pentru goal:
      agentul trebuie să ajungă pe celula goal.
    """

    start_pos = tuple(int(x) for x in base.agent_pos)
    start_dir = int(base.agent_dir)

    start_state = (start_pos[0], start_pos[1], start_dir)

    queue = deque()
    queue.append((start_state, []))

    visited = {start_state}

    while queue:
        (x, y, direction), path = queue.popleft()

        front = get_front_pos((x, y), direction)

        # Pentru cheie și ușă, ne oprim când obiectul este în fața agentului.
        if target_kind in ("key", "door") and front == target_pos:
            if len(path) == 0:
                return None
            return path[0]

        # Pentru goal, ne oprim când agentul ajunge pe goal.
        if target_kind == "goal" and (x, y) == target_pos:
            if len(path) == 0:
                return None
            return path[0]

        possible_actions = [0, 1, 2]

        for action in possible_actions:
            new_x, new_y, new_dir = x, y, direction

            if action == 0:
                new_dir = (direction - 1) % 4

            elif action == 1:
                new_dir = (direction + 1) % 4

            elif action == 2:
                fx, fy = get_front_pos((x, y), direction)

                if not is_cell_passable(base, fx, fy):
                    continue

                new_x, new_y = fx, fy

            new_state = (new_x, new_y, new_dir)

            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, path + [action]))

    return None


def planner_action(env, target_type):
    """
    Planner logic:

    Pentru normal/lava:
      mergi la goal.

    Pentru DoorKey:
      dacă nu are cheia -> mergi la cheie și pick up
      dacă are cheia și ușa e închisă -> mergi la ușă și toggle
      dacă ușa e deschisă -> mergi la goal
    """

    base = get_base_env(env)

    if not hasattr(base, "agent_pos"):
        return 2

    key_pos, door_pos, goal_pos, door_open = get_objects(base)
    carrying = is_carrying_key(base)

    if target_type == "key_door_goal":
        # 1. Mergi la cheie
        if not carrying and key_pos is not None:
            nav_action = bfs_next_navigation_action(base, "key", key_pos)

            if nav_action is None:
                return 3  # pick up

            return nav_action

        # 2. Mergi la ușă
        if carrying and door_pos is not None and not door_open:
            nav_action = bfs_next_navigation_action(base, "door", door_pos)

            if nav_action is None:
                return 4  # toggle/open

            return nav_action

        # 3. Mergi la goal
        if goal_pos is not None:
            nav_action = bfs_next_navigation_action(base, "goal", goal_pos)

            if nav_action is not None:
                return nav_action

            return 2

    # Pentru lava/empty
    if goal_pos is not None:
        nav_action = bfs_next_navigation_action(base, "goal", goal_pos)

        if nav_action is not None:
            return nav_action

    return 2


def is_useless_action_for_state(env, action_int, target_type):
    """
    Detectează acțiuni inutile:
    - pick up fără cheie în față
    - toggle/open fără ușă în față
    """

    if target_type != "key_door_goal":
        return False

    base = get_base_env(env)

    pos = get_agent_position(env)
    direction = get_agent_direction(env)

    if pos is None or direction is None:
        return False

    front_x, front_y = get_front_pos(pos, direction)

    if not hasattr(base, "grid"):
        return False

    obj = base.grid.get(front_x, front_y)

    # action 3 = pick up
    if action_int == 3:
        if obj is None or obj.type != "key":
            return True

    # action 4 = toggle/open
    if action_int == 4:
        if obj is None or obj.type != "door":
            return True

    return False


def should_force_planner_near_key(env, target_type):
    """
    Dacă agentul este aproape de cheie, planner-ul preia controlul.
    Asta evită situația în care PPO se învârte în jurul cheii fără să o ia.
    """

    if target_type != "key_door_goal":
        return False

    base = get_base_env(env)

    if not hasattr(base, "agent_pos"):
        return False

    if is_carrying_key(base):
        return False

    key_pos, door_pos, goal_pos, door_open = get_objects(base)

    if key_pos is None:
        return False

    agent_pos = tuple(int(x) for x in base.agent_pos)
    distance = manhattan(agent_pos, key_pos)

    return distance <= 2


def should_force_planner_near_door(env, target_type):
    """
    Dacă agentul are cheia și este aproape de ușă, planner-ul preia controlul.
    Asta evită situația în care PPO ajunge la ușă, dar o deschide/închide aiurea.
    """

    if target_type != "key_door_goal":
        return False

    base = get_base_env(env)

    if not hasattr(base, "agent_pos"):
        return False

    if not is_carrying_key(base):
        return False

    key_pos, door_pos, goal_pos, door_open = get_objects(base)

    if door_pos is None or door_open:
        return False

    agent_pos = tuple(int(x) for x in base.agent_pos)
    distance = manhattan(agent_pos, door_pos)

    return distance <= 2


def run_level_attempt(level_index, attempt_number):
    level = LEVELS[level_index]

    env_id = level["env_id"]
    model_path = level["model_path"]
    seed = level["seed"]
    target_type = level["target_type"]

    print("\n==============================")
    print(f"{level['name']} | Attempt {attempt_number}/{MAX_ATTEMPTS_PER_LEVEL}")
    print(f"Environment: {env_id}")
    print("==============================")

    env = make_minigrid_env(
        env_id,
        render_mode="human",
        seed=seed,
        target_type=target_type,
        shaped_reward=False,
    )

    model = PPO.load(str(model_path), env=env)

    obs, info = env.reset()

    total_reward = 0.0
    steps = 0

    last_action = None
    same_action_count = 0

    last_pos = get_agent_position(env)
    stuck_position_count = 0

    planner_mode = False
    planner_activated_reason = None

    done_reason = "unknown"

    while True:
        if planner_mode:
            action_int = planner_action(env, target_type)
        else:
            action, _states = model.predict(obs, deterministic=True)
            action_int = int(action)

        # Dacă PPO face o acțiune evident inutilă, activăm planner-ul.
        if (
            ENABLE_PLANNER_FALLBACK
            and not planner_mode
            and is_useless_action_for_state(env, action_int, target_type)
        ):
            planner_mode = True
            planner_activated_reason = f"useless action: {ACTION_NAMES.get(action_int, action_int)}"
            action_int = planner_action(env, target_type)
            print(f"Planner fallback activated: {planner_activated_reason}")

        # Dacă este aproape de cheie, planner-ul ajută la orientare + pick up.
        if (
            ENABLE_PLANNER_FALLBACK
            and not planner_mode
            and should_force_planner_near_key(env, target_type)
        ):
            planner_mode = True
            planner_activated_reason = "near key: planner takes over pickup"
            action_int = planner_action(env, target_type)
            print(f"Planner fallback activated: {planner_activated_reason}")

        # Dacă este aproape de ușă, planner-ul ajută la orientare + toggle/open.
        if (
            ENABLE_PLANNER_FALLBACK
            and not planner_mode
            and should_force_planner_near_door(env, target_type)
        ):
            planner_mode = True
            planner_activated_reason = "near door: planner takes over opening"
            action_int = planner_action(env, target_type)
            print(f"Planner fallback activated: {planner_activated_reason}")

        obs, reward, terminated, truncated, info = env.step(action_int)

        total_reward += float(reward)
        steps += 1

        current_pos = get_agent_position(env)

        if action_int == last_action:
            same_action_count += 1
        else:
            same_action_count = 0
            last_action = action_int

        if current_pos is not None and current_pos == last_pos:
            stuck_position_count += 1
        else:
            stuck_position_count = 0
            last_pos = current_pos

        # Dacă PPO se blochează, activăm planner-ul în loc să terminăm nivelul.
        if ENABLE_PLANNER_FALLBACK and not planner_mode:
            if same_action_count >= MAX_SAME_ACTIONS_BEFORE_PLANNER:
                planner_mode = True
                planner_activated_reason = "repeated same action"
                print(f"Planner fallback activated: {planner_activated_reason}")

            elif stuck_position_count >= MAX_STUCK_STEPS_BEFORE_PLANNER:
                planner_mode = True
                planner_activated_reason = "position did not change"
                print(f"Planner fallback activated: {planner_activated_reason}")

        if steps % 10 == 0:
            mode = "planner" if planner_mode else "ppo"
            print(
                f"Step {steps:03d} | "
                f"Mode: {mode:7s} | "
                f"Action: {ACTION_NAMES.get(action_int, action_int):12s} | "
                f"Reward: {total_reward:.3f} | "
                f"Pos: {current_pos}"
            )

        time.sleep(SLEEP_TIME)

        if terminated:
            done_reason = "terminated by environment"
            break

        if truncated:
            done_reason = "truncated by environment"
            break

        if steps >= MAX_STEPS_PER_ATTEMPT:
            done_reason = "max steps reached"
            break

    env.close()

    success = total_reward > 0 and terminated

    print(f"\n{level['name']} attempt ended.")
    print(f"Reason: {done_reason}")
    print(f"Reward: {total_reward:.3f}")
    print(f"Steps: {steps}")
    print(f"Success: {success}")

    if planner_mode:
        print(f"Planner fallback was used: {planner_activated_reason}")

    return success


def run_level(level_index):
    for attempt in range(1, MAX_ATTEMPTS_PER_LEVEL + 1):
        success = run_level_attempt(level_index, attempt)

        if success:
            return True

        print("Attempt failed. Retrying level...")
        time.sleep(1)

    return False


def main():
    print("MiniGrid Quest - Continuous Levels")
    print("Agentul trebuie să rezolve nivelul curent ca să deblocheze următorul.")
    print("Dacă PPO intră în loop, se activează planner fallback.")

    completed = 0
    failed_levels = []

    for level_index, level in enumerate(LEVELS):
        success = run_level(level_index)

        if success:
            completed += 1
            print(f"\n✅ {level['name']} completed.")
            print("Next level unlocked.")
            time.sleep(1)
        else:
            failed_levels.append(level["name"])
            print(f"\n⚠️ {level['name']} needs more training.")
            print("Marked as challenge level. Moving to next stage.")
            time.sleep(1)

    print("\n==============================")
    print("MiniGrid Quest finished")
    print("==============================")
    print(f"Completed levels: {completed}/{len(LEVELS)}")

    if failed_levels:
        print("Challenge levels that require more training:")
        for name in failed_levels:
            print(f"- {name}")
    else:
        print("All levels completed successfully!")


if __name__ == "__main__":
    main()