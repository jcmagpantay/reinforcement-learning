import gymnasium as gym

def run():
    env = gym.make('FrozenLake-v1', map_name = '8x8', is_slippery = True, render_mode = 'human')

    state = env.reset()[0]
    terminated = False
    truncated = False

    while (not terminated and not truncated):
        action = env.action_space.sample()

        new_state, reward, terminated, truncated, _ = env.step(action)

        state = new_state

    env.close()

if __name__ == 'main':
    run()