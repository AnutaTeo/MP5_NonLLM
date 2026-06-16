import imageio
from stable_baselines3 import PPO
from config import ENV_ID, MODEL_PATH, FRAME_STACK
from utils import make_vec_carracing_env, ensure_dirs


def main():
    ensure_dirs()

    env = make_vec_carracing_env(
        env_id=ENV_ID,
        render_mode="rgb_array",
        continuous=True,
        frame_stack=FRAME_STACK
    )

    model = PPO.load(str(MODEL_PATH), env=env)

    obs = env.reset()
    frames = []

    for step in range(1500):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done[0]:
            break

    env.close()

    video_path = "videos/trained_agent.mp4"
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Saved video to {video_path}")


if __name__ == "__main__":
    main()
