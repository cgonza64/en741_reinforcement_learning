GRAVITY = -10.0
DT = 0.02
MAIN_THRUST = 13.0
SIDE_THRUST = 1.0
TORQUE = 0.05

LANDER_W = 0.2
LANDER_H = 0.3
GROUND_Y = 0.0

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import transforms

class SimpleLunarLander:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = np.random.uniform(-0.5, 0.5)
        self.y = 1.2
        self.vx = 0.0
        self.vy = 0.0
        self.theta = 0.0
        self.omega = 0.0
        self.l_contact = 0
        self.r_contact = 0
        return self._state()

    def step(self, action):
        # Forces
        fx, fy = 0.0, GRAVITY
        torque = 0.0

        if action == 2:  # main engine
            fy += MAIN_THRUST
        elif action == 1:  # left engine
            fx -= SIDE_THRUST
            torque += TORQUE
        elif action == 3:  # right engine
            fx += SIDE_THRUST
            torque -= TORQUE

        # Integrate
        self.vx += fx * DT
        self.vy += fy * DT
        self.x += self.vx * DT
        self.y += self.vy * DT
        self.omega += torque
        self.theta += self.omega * DT

        # Collision with ground
        done = False
        reward = -1.0

        if self.y <= GROUND_Y:
            self.y = GROUND_Y
            self.vy = 0
            done = True

            # Landing reward
            if abs(self.vy) < 0.5 and abs(self.vx) < 0.5 and abs(self.theta) < 0.2:
                reward = 100.0
            else:
                reward = -100.0

        return self._state(), reward, done, {}

    def _state(self):
        return np.array([
            self.x, self.y, self.vx, self.vy,
            self.theta, self.omega,
            self.l_contact, self.r_contact
        ], dtype=np.float32)
    
class LanderRenderer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-0.5, 2)
        self.ax.set_aspect("equal")

        self.ground, = self.ax.plot([-2, 2], [0, 0], lw=3)

        self.body = Rectangle((-0.1, 0), 0.2, 0.3, fc="red")
        self.ax.add_patch(self.body)

        plt.ion()
        plt.show()

    def render(self, env):
        # Base transform
        t = transforms.Affine2D().rotate(env.theta).translate(env.x, env.y)

        self.body.set_transform(t + self.ax.transData)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

if __name__ == "__main__":
    env = SimpleLunarLander()
    viewer = LanderRenderer()

    s = env.reset()
    for _ in range(300):
        a = np.random.randint(4)
        s, r, done, _ = env.step(a)
        viewer.render(env)
        if done:
            break