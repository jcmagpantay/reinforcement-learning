from stable_baselines3 import PPO
from windy_cartpole import WindyCartPole

# Curriculum: easy → hard. Each phase ramps the wind; the agent carries its
# learned policy forward, so each phase starts from a competent baseline.
# Smaller rungs near the top (1.0→1.5→2.0) — the 1.0→2.0 jump alone is too big.
# Each phase has its own advance bar: stochastic strong gusts cap the achievable
# score below 500, so the bar relaxes as the wind gets harder.
#
# NOTE: `std` is the AR(1) per-step kick, NOT the effective volatility. The gust
# process amplifies variance by 1/√(1−gust²) ≈ 1.9x. These kicks are calibrated
# (= intended_effective_std × √(1−0.85²)) so the *effective* std matches the
# intended values below — i.e. `strong` now equals true original difficulty
# (effective std 0.8) instead of the ~1.5 it was secretly running at.
#                 label        mean  kick   advance_score   # intended eff. std
CURRICULUM = [
    ("calm",       0.0,  0.0,   450),                        # 0.0
    ("breeze",     0.5,  0.16,  450),                        # 0.3
    ("moderate",   1.0,  0.26,  440),                        # 0.5
    ("brisk",      1.5,  0.34,  400),                        # 0.65
    ("gale",       1.75, 0.38,  375),                        # 0.72
    ("strong",     2.0,  0.42,  350),                        # 0.8 (= original)
]

CHUNK_STEPS     = 25_000    # steps trained between evals
MAX_PHASE_STEPS = 400_000   # fallback cap so a phase can't loop forever
N_EVAL_EPISODES = 10        # average several episodes — wind is stochastic
ADVANCE_STREAK  = 3         # require this many consecutive evals over the bar
                            # before advancing — kills lucky-spike advancement


# LR is driven by a mutable box we update each chunk (SB3 re-reads the schedule
# every train iteration). We decay it *within* each phase and reset it at the
# start of every phase — so each phase starts with a high LR to learn the new
# wind, then settles to avoid the climb-then-overshoot wobble we saw.
LR_MAX = 3e-4
LR_MIN_FRAC = 0.1
lr_box = {"lr": LR_MAX}

train_env = WindyCartPole(wind_mean=0.0, wind_std=0.0)
eval_env  = WindyCartPole(wind_mean=0.0, wind_std=0.0)   # headless
model     = PPO("MlpPolicy", train_env, learning_rate=lambda _: lr_box["lr"], verbose=0)


def set_wind(mean, std):
    # train_env/eval_env are our own objects — SB3 wraps but references them,
    # so mutating these attributes changes the live environments.
    for env in (train_env, eval_env):
        env.wind_mean = mean
        env.wind_std  = std


def evaluate(n_episodes):
    scores = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total += reward
            done = terminated or truncated
        scores.append(total)
    return sum(scores) / len(scores)


for label, mean, std, advance_score in CURRICULUM:
    set_wind(mean, std)
    print(f"\n=== Phase: {label}  (wind_mean={mean}, wind_std={std}, target={advance_score}) ===")

    phase_steps = 0
    streak = 0   # consecutive evals clearing the bar — guards against lucky spikes
    while True:
        # Linear LR decay within the phase: LR_MAX → LR_MAX*LR_MIN_FRAC,
        # floored so late-phase learning doesn't fully stall. Resets each phase.
        frac = max(LR_MIN_FRAC, 1.0 - phase_steps / MAX_PHASE_STEPS)
        lr_box["lr"] = LR_MAX * frac

        model.learn(total_timesteps=CHUNK_STEPS, reset_num_timesteps=False)
        phase_steps += CHUNK_STEPS

        score = evaluate(N_EVAL_EPISODES)
        streak = streak + 1 if score >= advance_score else 0
        print(f"  [{phase_steps:>7,} steps in phase]  avg score: {score:.1f}"
              f"   (streak {streak}/{ADVANCE_STREAK})")

        # Advance only on sustained performance, not a single lucky eval.
        if streak >= ADVANCE_STREAK:
            print(f"  ✓ cleared '{label}' ({ADVANCE_STREAK} consecutive evals ≥ {advance_score})")
            break
        if phase_steps >= MAX_PHASE_STEPS:
            print(f"  ⚠ hit step cap on '{label}' (best this run {score:.1f}) — advancing anyway")
            break

train_env.close()
eval_env.close()

# --- Final visual demo at full difficulty ---
input("\nCurriculum complete. Press Enter to watch the agent at full difficulty...")
demo_env = WindyCartPole(render_mode="human", wind_mean=2.0, wind_std=0.8)
obs, _ = demo_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = demo_env.step(action)
    done = terminated or truncated
demo_env.close()
