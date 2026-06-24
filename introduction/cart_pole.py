import gymnasium as gym
from stable_baselines3 import PPO
from windy_cartpole import WindyCartPole

TOTAL_STEPS     = 100_000
CHUNKS          = 10
STEPS_PER_CHUNK = TOTAL_STEPS // CHUNKS

train_env = WindyCartPole(wind_mean=2.0, wind_std=1.5)
eval_env  = WindyCartPole(render_mode="human", wind_mean=2.0, wind_std=1.5)
model     = PPO("MlpPolicy", train_env, verbose=0)

for chunk in range(1, CHUNKS + 1):
    model.learn(total_timesteps=STEPS_PER_CHUNK, reset_num_timesteps=False)

    obs, _ = eval_env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"[{chunk * STEPS_PER_CHUNK:>7,} / {TOTAL_STEPS:,} steps]  score: {total_reward:.1f}")

train_env.close()
eval_env.close()
