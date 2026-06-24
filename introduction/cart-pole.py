import gymnasium as gym
from gymnasium import RewardWrapper
from stable_baselines3 import PPO

TOTAL_STEPS     = 50_000
CHUNKS          = 10
STEPS_PER_CHUNK = TOTAL_STEPS // CHUNKS
CENTER_PENALTY  = 0.5


class StayCentered(RewardWrapper):
    def reward(self, reward):
        cart_position = self.unwrapped.state[0]
        return reward - CENTER_PENALTY * abs(cart_position)


train_env = StayCentered(gym.make("CartPole-v1"))
eval_env  = gym.make("CartPole-v1", render_mode="human")
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

    print(f"[{chunk * STEPS_PER_CHUNK:>6,} / {TOTAL_STEPS:,} steps]  score: {total_reward:.0f}")

train_env.close()
eval_env.close()
