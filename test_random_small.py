import cv2
import gymnasium as gym
from config import ENV_ID
from utils import make_carracing_env


def main():
    env = make_carracing_env(ENV_ID, render_mode="rgb_array", continuous=True)

    obs, info = env.reset()
    total_reward = 0

    for step in range(3000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        frame = env.render()

        # resize window
        frame = cv2.resize(frame, (600, 600))

        # convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.putText(
            frame,
            f"Random Agent | Reward: {total_reward:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("RaceAI - Random Agent", frame)

        # 25 ms = viteza mai ok pentru vizualizare
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

        if terminated or truncated:
            print(f"Episode ended. Total reward: {total_reward:.2f}")
            total_reward = 0
            obs, info = env.reset()

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()