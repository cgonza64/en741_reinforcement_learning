import numpy as np
import pygame
import math

# --- Constants ---
FPS = 60
DT = 1.0 / FPS

# PHYSICS CONSTANTS (MOON)
GRAVITY = -1.625  # Moon gravity (approx 1/6 Earth)
SCALE = 10.0  # Pixels per meter (Zoomed out to show approach)

# WORLD
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
GROUND_Y = 50.0  # Meters from bottom of screen (visual offset)

# LANDER PROPERTIES (Apollo-esque ratios)
LANDER_DRY_MASS = 600.0  # kg (Structure + Descent Engine)
FUEL_MASS_START = 400.0  # kg (Propellant)
MAX_FUEL = 400.0

LANDER_WIDTH = 6.0  # Wider stance like the LEM
LANDER_HEIGHT = 4.0

# THRUST PARAMETERS
MAIN_THRUST = 4500.0  # T/W Ratio ~ 2.7 (at start) -> ~ 4.6 (when empty)
SIDE_THRUST = 1000.0  # RCS Thrusters
SIDE_ENGINE_OFFSET = 3.0  # Torque leverage

FUEL_CONSUMPTION_MAIN = 2.0  # kg per second (approx)
FUEL_CONSUMPTION_SIDE = 0.5

# LANDING ZONE
PAD_WIDTH = 20.0  # Meters
PAD_X_TARGET = 0.0  # Center of the world (0,0)

# COLORS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
GOLD = (212, 175, 55)  # Apollo Foil Gold
GREY = (100, 100, 100)
DARK_GREY = (50, 50, 50)
BLUE = (50, 150, 255)
ORANGE = (255, 165, 0)

# ----------------------------
# DISCRETIZATION CONFIGURATION
# ----------------------------
Y_MIN, Y_MAX, Y_BINS = 0.0, 60.0, 10
VY_MIN, VY_MAX, VY_BINS = -20.0, 20.0, 10

# STARTING POSITION (Fixed for simplified learning)
START_X = 0.0  # Always start at center
START_Y = 45.0  # Fixed starting altitude


def _digitize_clamped(val, vmin, vmax, bins):
    """Returns an integer bin index in [0, bins-1], clamped to range"""
    if val <= vmin:
        return 0
    if val >= vmax:
        return bins - 1
    ratio = (val - vmin) / (vmax - vmin)
    idx = int(ratio * bins)
    if idx >= bins:
        idx = bins - 1
    return idx


class LunarLanderEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # SIMPLIFIED INITIALIZATION - Fixed starting position for easier learning
        # Always start at center (x=0) with fixed altitude
        self.x = START_X
        self.y = START_Y

        # No horizontal velocity, slight downward drift
        self.vx = 0.0
        self.vy = np.random.uniform(-2.0, 0.0)  # Slight downward drift

        # Start level (no tilt)
        self.theta = 0.0
        self.omega = 0.0

        self.fuel = FUEL_MASS_START
        self.mass = LANDER_DRY_MASS + self.fuel

        self.landed = False
        self.crashed = False
        self.trace = []

        self.prev_shaping = None

        # New: ground contact flag
        self.made_contact = False

        # Approximate Moment of Inertia (Rectangle)
        self.moment_of_inertia = LANDER_DRY_MASS * (LANDER_WIDTH ** 2 + LANDER_HEIGHT ** 2) / 12.0

        self._calculate_shaping()
        return self._get_state()

    def _calculate_shaping(self):
        # Potential-based reward shaping
        dist_penalty = np.sqrt(self.x ** 2 + self.y ** 2)
        vel_penalty = np.sqrt(self.vx ** 2 + self.vy ** 2)
        tilt_penalty = abs(self.theta)
        return -10.0 * dist_penalty - 100.0 * vel_penalty - 100.0 * tilt_penalty

    def _get_state(self):
        """
        Returns discretized observation: [y_bin, vy_bin, made_contact]
        - y_bin: discretized vertical position (0 to Y_BINS-1)
        - vy_bin: discretized vertical velocity (0 to VY_BINS-1)
        - made_contact: binary (0 or 1) indicating ground contact
        """
        y_bin = _digitize_clamped(self.y, Y_MIN, Y_MAX, Y_BINS)
        vy_bin = _digitize_clamped(self.vy, VY_MIN, VY_MAX, VY_BINS)
        contact = 1 if self.made_contact else 0
        return np.array([y_bin, vy_bin, contact], dtype=np.int32)

    def step(self, action):
        """
        REDUCED Action space:
        0: Do nothing
        2: Main Engine (vertical thrust)
        
        Note: Actions 1 and 3 (side thrusters) are no longer available
        """
        if self.landed or self.crashed:
            return self._get_state(), 0, True, {}

        # Update Mass based on fuel burn
        self.mass = LANDER_DRY_MASS + self.fuel

        force_x = 0.0
        force_y = self.mass * GRAVITY  # Weight
        torque = 0.0

        # Apply Thrust - ONLY MAIN ENGINE (action 2)
        if self.fuel > 0 and action == 2:
            sin_theta = np.sin(self.theta)
            cos_theta = np.cos(self.theta)
            
            f_thrust = MAIN_THRUST
            force_x += -sin_theta * f_thrust
            force_y += cos_theta * f_thrust
            self.fuel -= FUEL_CONSUMPTION_MAIN * DT

        # Integrate (Newton's Second Law: F = ma)
        accel_x = force_x / self.mass
        accel_y = force_y / self.mass

        self.vx += accel_x * DT
        self.vy += accel_y * DT
        self.x += self.vx * DT
        self.y += self.vy * DT

        alpha = torque / self.moment_of_inertia
        self.omega += alpha * DT
        self.theta += self.omega * DT

        self.trace.append((self.x, self.y))
        if len(self.trace) > 200:
            self.trace.pop(0)

        # Reward Calculation
        shaping = self._calculate_shaping()
        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Constant fuel penalty (encourages time-optimality)
        reward -= 0.05

        # Collision Check
        done = False

        # Legs: Define the LEM feet relative to center
        half_w = LANDER_WIDTH / 2.0 + 1.0  # Gear extends past body
        half_h = LANDER_HEIGHT / 2.0 + 1.0  # Gear extends below body

        def get_world_point(lx, ly):
            c = np.cos(self.theta)
            s = np.sin(self.theta)
            return self.x + lx * c - ly * s, self.y + lx * s + ly * c

        # Check Left and Right foot
        left_foot = get_world_point(-half_w, -half_h)
        right_foot = get_world_point(half_w, -half_h)

        # Ground Plane (y=0)
        if left_foot[1] <= 0 or right_foot[1] <= 0:
            # Physics adjustment to sit on ground
            min_y = min(left_foot[1], right_foot[1])
            self.y -= min_y

            # New: mark terrain contact
            self.made_contact = True

            # Landing Criteria (Apollo style: very gentle)
            vel_safe = abs(self.vy) < 2.5 and abs(self.vx) < 2.0
            angle_safe = abs(self.theta) < 0.3
            on_pad = abs(self.x) < (PAD_WIDTH / 2.0)

            if vel_safe and angle_safe and on_pad:
                self.landed = True
                reward += 100.0
                print(f"EAGLE HAS LANDED. Fuel Left: {self.fuel:.1f}kg")
            else:
                self.crashed = True
                reward -= 100.0
                reason = []
                if not vel_safe:
                    reason.append("Too Fast")
                if not angle_safe:
                    reason.append("Tilted")
                if not on_pad:
                    reason.append("Missed LZ")
                print(f"ABORT/CRASH: {', '.join(reason)}")

            # End the episode on contact (either landing or crash)
            done = True
            self.vy = 0
            self.omega = 0

        # Out of bounds
        if abs(self.x) > (SCREEN_WIDTH / SCALE) / 2 + 20 or self.y > (SCREEN_HEIGHT / SCALE) + 20:
            done = True
            self.crashed = True
            reward -= 100.0

        return self._get_state(), reward, done, {}


class Renderer:
    def __init__(self, env):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Lunar Lander - Vertical Descent (Discretized)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)
        self.env = env

    def world_to_screen(self, x, y):
        # Center x on screen (Screen Center = World 0)
        screen_x = int(SCREEN_WIDTH / 2 + x * SCALE)
        # Flip Y (Screen 0 is top)
        screen_y = int(SCREEN_HEIGHT - (GROUND_Y + y * SCALE))
        return screen_x, screen_y

    def render(self, action=0):
        self.screen.fill(BLACK)

        # 1. Stars (Static background decoration)
        for i in range(50):
            sx = (i * 137) % SCREEN_WIDTH
            sy = (i * 93) % SCREEN_HEIGHT
            self.screen.set_at((sx, sy), WHITE)

        # 2. Moon Surface
        ground_px = int(SCREEN_HEIGHT - GROUND_Y)
        pygame.draw.rect(self.screen, GREY, (0, ground_px, SCREEN_WIDTH, SCREEN_HEIGHT - ground_px))

        # 3. Landing Pad (Target)
        pad_w_px = PAD_WIDTH * SCALE
        pad_left, _ = self.world_to_screen(-PAD_WIDTH / 2, 0)
        pygame.draw.rect(self.screen, DARK_GREY, (pad_left, ground_px, pad_w_px, 10))
        # Landing marker
        pygame.draw.circle(self.screen, WHITE, (int(SCREEN_WIDTH / 2), ground_px + 5), 5)

        # 4. Trajectory Trace
        if len(self.env.trace) > 1:
            pts = [self.world_to_screen(px, py) for px, py in self.env.trace]
            pygame.draw.lines(self.screen, BLUE, False, pts, 1)

        # 5. Draw Lander (LEM Style)
        w = LANDER_WIDTH * SCALE
        h = LANDER_HEIGHT * SCALE

        # Bigger surface to accommodate rotation and legs
        surf_size = int(max(w, h) * 3)
        lander_surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
        cx, cy = surf_size // 2, surf_size // 2

        # -- DESCENT STAGE (Gold Octagon-ish) --
        pygame.draw.rect(lander_surf, GOLD, (cx - w / 2, cy, w, h / 1.5))
        # Legs (4 legs, simplified to 2 striding)
        pygame.draw.line(lander_surf, GOLD, (cx - w / 2, cy + h / 2), (cx - w / 2 - 10, cy + h), 3)
        pygame.draw.line(lander_surf, GOLD, (cx + w / 2, cy + h / 2), (cx + w / 2 + 10, cy + h), 3)
        # Footpads
        pygame.draw.circle(lander_surf, GREY, (int(cx - w / 2 - 10), int(cy + h)), 4)
        pygame.draw.circle(lander_surf, GREY, (int(cx + w / 2 + 10), int(cy + h)), 4)

        # -- ASCENT STAGE (Grey/Black Top) --
        pygame.draw.rect(lander_surf, DARK_GREY, (cx - w / 2.2, cy - h / 2, w / 1.1, h / 2))
        pygame.draw.polygon(lander_surf, BLACK, [(cx - w / 2.2, cy - h / 2), (cx, cy - h), (cx + w / 2.2, cy - h / 2)])

        # -- FLAMES (ONLY MAIN ENGINE) --
        if self.env.fuel > 0 and action == 2:
            pygame.draw.polygon(lander_surf, ORANGE,
                                [(cx - w / 3, cy + h / 1.5), (cx + w / 3, cy + h / 1.5), (cx, cy + h * 1.5)])

        # Rotate and Blit
        rot_surf = pygame.transform.rotate(lander_surf, math.degrees(self.env.theta))
        rect = rot_surf.get_rect()
        rect.center = self.world_to_screen(self.env.x, self.env.y)
        self.screen.blit(rot_surf, rect)

        # 6. HUD / Telemetry
        # Get current discretized state
        state = self.env._get_state()
        telemetry = [
            f"ALTITUDE: {self.env.y:.1f} m (bin={state[0]})",
            f"H-SPEED:  {self.env.vx:.1f} m/s",
            f"V-SPEED:  {self.env.vy:.1f} m/s (bin={state[1]})",
            f"FUEL:     {self.env.fuel:.1f} kg",
            f"MASS:     {self.env.mass:.0f} kg",
            f"CONTACT:  {state[2]}",
            f"ACTION:   {'THRUST' if action == 2 else 'NONE'}"
        ]

        for i, line in enumerate(telemetry):
            color = WHITE
            if "FUEL" in line and self.env.fuel < 50:
                color = RED
            txt = self.font.render(line, True, color)
            self.screen.blit(txt, (10, 10 + i * 20))

        # Landing/Crash overlays
        if self.env.landed:
            s = pygame.Surface((SCREEN_WIDTH, 60), pygame.SRCALPHA)
            s.fill((0, 255, 0, 100))
            self.screen.blit(s, (0, SCREEN_HEIGHT / 2 - 30))
            msg = self.font.render("TOUCHDOWN CONFIRMED - R to Reset", True, WHITE)
            self.screen.blit(msg, (SCREEN_WIDTH / 2 - 150, SCREEN_HEIGHT / 2 - 10))

        if self.env.crashed:
            s = pygame.Surface((SCREEN_WIDTH, 60), pygame.SRCALPHA)
            s.fill((255, 0, 0, 100))
            self.screen.blit(s, (0, SCREEN_HEIGHT / 2 - 30))
            msg = self.font.render("VEHICLE LOST - R to Reset", True, WHITE)
            self.screen.blit(msg, (SCREEN_WIDTH / 2 - 120, SCREEN_HEIGHT / 2 - 10))

        pygame.display.flip()


def main():
    env = LunarLanderEnv()
    renderer = Renderer(env)
    running = True

    print("=" * 70)
    print("LUNAR LANDER - DISCRETIZED STATE & SIMPLIFIED VERTICAL DESCENT")
    print("=" * 70)
    print("Controls:")
    print("  UP Arrow  = Fire Main Engine (Action 2)")
    print("  (no key)  = Do Nothing (Action 0)")
    print("  R         = Reset")
    print("=" * 70)
    print(f"Starting Position: x={START_X:.1f}m, y={START_Y:.1f}m (FIXED)")
    print("=" * 70)
    print(f"State Space:")
    print(f"  - y_bin: 0 to {Y_BINS-1} (altitude bins from {Y_MIN} to {Y_MAX}m)")
    print(f"  - vy_bin: 0 to {VY_BINS-1} (velocity bins from {VY_MIN} to {VY_MAX}m/s)")
    print(f"  - made_contact: 0 or 1 (ground contact)")
    print(f"  Total states: {Y_BINS * VY_BINS * 2} = {Y_BINS}×{VY_BINS}×2")
    print("=" * 70)
    print(f"Action Space: 2 actions")
    print(f"  - 0: Do nothing")
    print(f"  - 2: Fire main engine")
    print("=" * 70)

    while running:
        renderer.clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()

        keys = pygame.key.get_pressed()
        action = 0
        if keys[pygame.K_UP]:
            action = 2

        env.step(action)
        renderer.render(action)

    pygame.quit()


if __name__ == "__main__":
    main()