import numpy as np

class RLAgent:
    """
    TODO
    """
    def __init__(self,
                 agent_type='SARSA',
                 num_of_states=96,
                 num_of_actions=4,
                 alpha=0.4,
                 gamma=1.0,
                 epsilon=0.1,
                 eps_decay=False):
        """
        TODO
        """
        self.agent_type = agent_type
        self.number_of_states = num_of_states
        self.number_of_actions = num_of_actions
        self.q = np.zeros((self.number_of_states, self.number_of_actions), dtype="float64")
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_enabled = True

        # epsilon decay parameters
        self.eps_decay = eps_decay
        self.decay_step = 1
        self.decay_episodes = 2000
        self.eps_max = 0.7
        self.eps_min = 0.02
        
    def learning_control(self, enabled):
        """
        TODO
        """
        self.learning_enabled = enabled
        self.epsilon = 0.0 if not enabled else self.epsilon

    def get_number_of_states(self):
        """ Returns the total number of states in the environment."""
        return self.number_of_states

    def get_number_of_actions(self):
        """ Returns the total number of actions available to the agent."""
        return self.number_of_actions

    def reset(self, eps=0.1, learning_enabled=True):
        """ 
        TODO
        """
        for i in range(self.number_of_states):
            for j in range(self.number_of_actions):
                self.q[i, j] = 0.0
        self.epsilon = eps
        self.learning_control(learning_enabled)

    def epsilon_decay(self):
        """
        Decay function for epsilon to reduce exploration over time.
        The decay is linear over a number of episodes specified by the decay_episodes
        attribute, starting from eps_max and decaying down to eps_min.
        """
        r = max(0, (self.decay_episodes - self.decay_step) / self.decay_episodes)
        self.epsilon = self.eps_min + (self.eps_max - self.eps_min) * r
        self.decay_step += 1
    
    def e_greedy(self, actions):
        """
        Uses epsilon-greedy action selection to choose an action.

        :param actions (List[int]): Array of integer-encoded actions for the current state.
        :return: Selected action index based on epsilon-greedy strategy.
        """
        if self.eps_decay:
            # Decay epsilon over time to reduce exploration as 
            # the agent learns more about the environment
            self.epsilon_decay()  
        a_star_idx = np.argmax(actions)
        rng = np.random.default_rng()
        if self.epsilon <= rng.random():
            return a_star_idx
        else:
            b = actions.size
            idx = rng.integers(low=0, high=b)
            return idx
            
    def select_action(self, state):
        """
        Selects an action for the given state using either a rule-based policy or epsilon-greedy strategy.

        :param state (int): The current state of the game.
        :return: Action (0 for stick, 1 for hit) based on the selected policy.
        """
        A = self.q[state, ]
        action = self.e_greedy(A)
        return action

    def update_q(self, state, action, reward, next_state, next_action=None):
        """ 
        Updates the Q-function using the first-visit Monte Carlo method based on the stored state-action pairs 
        and rewards for the episode. This method should be called at the end of each episode after all rewards 
        have been stored.
        """
        if self.learning_enabled:
            if self.agent_type == 'SARSA' and next_action is None:
                next_action = self.select_action(next_state)
            elif self.agent_type == "Q-learning" and next_action is None:
                next_action = np.argmax(self.q[next_state, ])
            q_current = self.q[state, action]
            q_next = self.q[next_state, next_action] if next_action is not None else 0.0  # check if terminal state has been reached
            self.q[state, action] = q_current + self.alpha*(reward + self.gamma*q_next - q_current)

        return next_action

    def return_policy(self):
        """ 
        Returns the learned policy as a 2D array where each entry corresponds to the action with the highest Q-value for that state.
        This can be used to visualize the learned policy after training.
        """
        policy = np.zeros(self.number_of_states, dtype=int)
        for i in range(self.number_of_states):
            policy[i] = np.argmax(self.q[i, ])
        return policy

    def print_q(self):
        """ 
        Helper function to print the Q-function values for all state-action pairs.
        This can be useful for debugging and understanding the learned values after training.
        """
        print("Q-function:")
        for i in range(self.number_of_states):
            for j in range(self.number_of_actions):
                print(f"Q[{i}, {j}] = {self.q[i, j]:.3f}")
