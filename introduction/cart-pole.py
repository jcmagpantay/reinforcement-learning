import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

# --- Hyperparameters ---
TOTAL_TIMESTEPS = 50_000
N_STEPS    = 2048
N_EPOCHS   = 10
BATCH_SIZE = 64
GAMMA      = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS   = 0.2
LR         = 3e-4
ENT_COEF   = 0.01
VF_COEF    = 0.5

CHUNKS     = 10
STEPS_PER_CHUNK = TOTAL_TIMESTEPS // CHUNKS

CENTER_PENALTY = 0.5  # weight of position penalty in reward shaping


# --- Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),     nn.Tanh(),
        )
        self.actor  = nn.Linear(64, act_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x).squeeze(-1)

    def get_action(self, obs):
        logits, value = self(obs)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


def compute_gae(rewards, values, dones, last_value):
    advantages = torch.zeros_like(rewards)
    last_gae   = 0.0
    for t in reversed(range(len(rewards))):
        next_value       = last_value if t == len(rewards) - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta            = rewards[t] + GAMMA * next_value * next_nonterminal - values[t]
        last_gae         = delta + GAMMA * GAE_LAMBDA * next_nonterminal * last_gae
        advantages[t]    = last_gae
    return advantages, advantages + values


def ppo_update(model, optimizer, obs_buf, act_buf, logp_buf, advantages, returns):
    indices = np.arange(len(obs_buf))
    for _ in range(N_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, len(obs_buf), BATCH_SIZE):
            batch = indices[start : start + BATCH_SIZE]

            logits, values = model(obs_buf[batch])
            dist     = Categorical(logits=logits)
            new_logp = dist.log_prob(act_buf[batch])
            entropy  = dist.entropy().mean()

            ratio      = (new_logp - logp_buf[batch]).exp()
            b_adv      = advantages[batch]
            policy_loss = -torch.min(
                ratio * b_adv,
                torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * b_adv
            ).mean()

            value_loss = (values - returns[batch]).pow(2).mean()
            loss       = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


def collect_rollout(env, model, n_steps):
    obs_dim = env.observation_space.shape[0]
    obs_buf  = torch.zeros(n_steps, obs_dim)
    act_buf  = torch.zeros(n_steps, dtype=torch.long)
    logp_buf = torch.zeros(n_steps)
    rew_buf  = torch.zeros(n_steps)
    done_buf = torch.zeros(n_steps)
    val_buf  = torch.zeros(n_steps)

    obs, _ = env.reset()
    for step in range(n_steps):
        obs_t = torch.FloatTensor(obs)
        with torch.no_grad():
            action, logp, _, value = model.get_action(obs_t)

        obs_buf[step]  = obs_t
        act_buf[step]  = action
        logp_buf[step] = logp
        val_buf[step]  = value

        obs, reward, terminated, truncated, _ = env.step(action.item())

        # Reward shaping: penalize distance from center (cart position is obs[0])
        cart_position = obs[0]
        reward -= CENTER_PENALTY * abs(cart_position)

        done_buf[step] = float(terminated or truncated)
        rew_buf[step]  = reward

        if terminated or truncated:
            obs, _ = env.reset()

    with torch.no_grad():
        _, last_value = model(torch.FloatTensor(obs))

    return obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, last_value


def evaluate(model, eval_env):
    obs, _ = eval_env.reset()
    total_reward = 0.0
    done = False
    while not done:
        with torch.no_grad():
            action, _, _, _ = model.get_action(torch.FloatTensor(obs))
        obs, reward, terminated, truncated, _ = eval_env.step(action.item())
        total_reward += reward
        done = terminated or truncated
    return total_reward


# --- Main ---
train_env = gym.make("CartPole-v1")
eval_env  = gym.make("CartPole-v1", render_mode="human")

model     = ActorCritic(train_env.observation_space.shape[0], train_env.action_space.n)
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"Training PPO on CartPole-v1 (stay-centered) for {TOTAL_TIMESTEPS:,} timesteps\n")

for chunk in range(1, CHUNKS + 1):
    obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, last_value = \
        collect_rollout(train_env, model, STEPS_PER_CHUNK)

    advantages, returns = compute_gae(rew_buf, val_buf, done_buf, last_value)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ppo_update(model, optimizer, obs_buf, act_buf, logp_buf, advantages, returns)

    score = evaluate(model, eval_env)
    print(f"[{chunk * STEPS_PER_CHUNK:>6,} / {TOTAL_TIMESTEPS:,} steps]  eval score: {score:.0f}")

train_env.close()
eval_env.close()
print("\nDone.")
