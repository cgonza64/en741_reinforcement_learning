import numpy as np


class RlAgent:
    """RL agent for the Fred the Robot game"""

    def __init__(self):
        self.q = np.zeros((12, 4), dtype="float64")
        self.state = 0
        self.next_state = 0
        self.reward = 0
        self.action = 0
        self.turn = 0
        self.epsilon = 1.0
        self.alpha = 0.1
        self.gamma = 0.9
        self.number_of_states = 11
        self.number_of_actions = 4

    def get_number_of_states(self):
        return self.number_of_states

    def get_number_of_actions(self):
        return self.number_of_actions

    def reset(self, eps=0.2):
        for i in range(12):
            for j in range(4):
                self.q[i, j] = 0.0
        self.state = 0
        self.next_state = 0
        self.reward = 0
        self.action = 0
        self.turn = 0
        self.epsilon = eps

    def e_greedy(self, actions):
        a_star_idx = np.argmax(actions)
        rng = np.random.default_rng()
        if self.epsilon <= rng.random():
            return a_star_idx
        else:
            b = actions.size
            idx = rng.integers(low=0, high=b)
            return idx

    def select_action(self, state):
        self.turn += 1
        # print("Turn = ", self.turn)
        self.state = state
        # print("State = ", self.state)
        actions = self.q[state, ]
        action = self.e_greedy(actions)
        self.action = action
        return action

    def update_q(self, new_state, reward):
        self.next_state = new_state
        self.q[self.state, self.action] = reward + (self.gamma * max(self.q[new_state, ]))
        f"Turn = {self.turn} \nQ = {self.q}"

