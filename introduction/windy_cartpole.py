import numpy as np
import gymnasium as gym
from gymnasium import spaces


class WindyCartPole(gym.Env):
    """
    CartPole with wind that applies a horizontal force to both the cart
    and a torque on the pole. Wind is stochastic — sampled each step from
    N(wind_mean, wind_std). The agent observes the current wind force so
    it can learn to compensate.

    Observation: [cart_pos, cart_vel, pole_angle, pole_angular_vel, wind_force]
    Action:      0 (push left) | 1 (push right)
    Reward:      1.0 - 0.5*|cart_pos| - 0.3*|pole_angle|
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # Physics constants (identical to CartPole-v1)
    gravity          = 9.8
    masscart         = 1.0
    masspole         = 0.1
    length           = 0.5   # half-pole length
    force_mag        = 10.0
    tau              = 0.02  # seconds per step

    x_threshold      = 2.4
    theta_threshold  = 12 * np.pi / 180  # radians

    def __init__(self, render_mode=None, wind_mean=2.0, wind_std=1.5):
        super().__init__()

        self.wind_mean   = wind_mean
        self.wind_std    = wind_std
        self.wind_force  = 0.0
        self.render_mode = render_mode
        self.screen      = None
        self.clock       = None
        self.state       = None

        self.total_mass      = self.masscart + self.masspole
        self.polemass_length = self.masspole * self.length

        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold * 2,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,   # wind
        ], dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space      = spaces.Discrete(2)

    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state      = self.np_random.uniform(-0.05, 0.05, size=(4,))
        self.wind_force = self._sample_wind()
        if self.render_mode == "human":
            self._render_frame()
        return self._obs(), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        self.wind_force = self._sample_wind()

        force      = self.force_mag if action == 1 else -self.force_mag
        cart_force = force + self.wind_force  # wind pushes cart laterally

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # Standard CartPole equations of motion
        temp     = (cart_force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))

        # Wind also torques the pole: horizontal force at pole midpoint
        # τ = F_wind * length * cos(θ),  α_wind = τ / (m * l² * 4/3)
        thetaacc += (self.wind_force * costheta) / \
                    (self.masspole * self.length * (4.0/3.0))

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler integration
        x         += self.tau * x_dot
        x_dot     += self.tau * xacc
        theta     += self.tau * theta_dot
        theta_dot += self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot])

        terminated = bool(
            abs(x)     > self.x_threshold or
            abs(theta) > self.theta_threshold
        )

        reward = 1.0 - 0.5 * abs(x) - 0.3 * abs(theta)

        if self.render_mode == "human":
            self._render_frame()

        return self._obs(), reward, terminated, False, {"wind": self.wind_force}

    # ------------------------------------------------------------------

    def _sample_wind(self):
        return float(self.np_random.normal(self.wind_mean, self.wind_std))

    def _obs(self):
        return np.append(self.state, self.wind_force).astype(np.float32)

    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        import pygame

        W, H       = 600, 400
        scale      = W / (self.x_threshold * 2)
        cartw      = 50
        carth      = 30
        polewidth  = 10
        polelen    = scale * (2 * self.length)
        carty      = 300

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("Windy CartPole")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((245, 245, 245))

        x, _, theta, _ = self.state
        cartx = int(x * scale + W / 2)

        # Track
        pygame.draw.line(self.screen, (180, 180, 180), (0, carty + carth // 2), (W, carty + carth // 2), 2)

        # Center marker
        pygame.draw.line(self.screen, (200, 200, 200), (W // 2, carty - 25), (W // 2, carty + 25), 2)

        # Cart
        pygame.draw.rect(self.screen, (60, 60, 60),
                         (cartx - cartw // 2, carty - carth // 2, cartw, carth))

        # Pole
        px = cartx + polelen * np.sin(theta)
        py = carty - carth // 2 - polelen * np.cos(theta)
        pygame.draw.line(self.screen, (180, 90, 40),
                         (cartx, carty - carth // 2), (int(px), int(py)), polewidth)

        # Wind arrow
        font      = pygame.font.SysFont(None, 28)
        arrow_dir = "→" if self.wind_force > 0 else "←"
        strength  = abs(self.wind_force)
        color     = (50, 100, 200) if strength < 2 else (200, 60, 60)
        wind_surf = font.render(f"Wind {arrow_dir} {strength:.1f} N", True, color)
        self.screen.blit(wind_surf, (10, 10))

        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None
