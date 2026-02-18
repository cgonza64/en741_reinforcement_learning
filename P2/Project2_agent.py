import numpy as np

class RLAgent:
    """
    On-Policy First-Visit Monte Carlo RL agent for the Blackjack casino card game.
    By default, no discounting is applied (gamma = 1.0) since the game is episodic and has a finite horizon.
    For learning agents, e-greedy action selection is used with a default epsilon of 0.2.
    For random agents, epsilon should be set to 1.0 to ensure completely random action selection.
    The agent can also be configured to use a simple rule-based policy instead of learning, which is based on the player's
    hand and the dealer's visible card.
    """
    def __init__(self, rule_based=False, epsilon=0.2, gamma=1.0, num_of_states=204, num_of_actions=2):
        """
        Initializes the RL agent.

        :param rule_based (bool): If True, the agent uses a rule-based policy instead of learning.
        :param epsilon (float): Exploration rate for epsilon-greedy action selection.
        :param gamma (float): Discount factor (1.0 for episodic games).
        :param num_of_states (int): Total number of states in the environment (204).
        :param num_of_actions (int): Total number of actions available (2).
        """
        self.number_of_states = num_of_states
        self.number_of_actions = num_of_actions
        self.q = np.zeros((self.number_of_states, self.number_of_actions), dtype="float64")
        self.n = np.zeros((self.number_of_states, self.number_of_actions), dtype="int64")
        self.epsilon = epsilon
        self.gamma = gamma
        self.rule_based = rule_based
        self.learning_enabled = True
        self.episode_sa = []
        self.episode_r = []
        
    def learning_control(self, enabled):
        """
        Controls whether learning is enabled or disabled for the agent. 
        When learning is disabled, epsilon is set to 0 to ensure optimal action selection.
        
        :param enabled (bool): If True, learning is enabled; otherwise, it is disabled.
        """
        self.learning_enabled = enabled
        self.epsilon = 0.0 if not enabled else self.epsilon

    def get_number_of_states(self):
        """ Returns the total number of states in the environment."""
        return self.number_of_states

    def get_number_of_actions(self):
        """ Returns the total number of actions available to the agent."""
        return self.number_of_actions

    def reset(self, eps=0.2, learning_enabled=True):
        """ 
        Resets the agent's Q-function, visit counts, and episode data.
        Also resets epsilon and learning control settings.

        :param eps (float): Exploration rate for epsilon-greedy action selection.
        :param learning_enabled (bool): If True, learning is enabled; otherwise, it is disabled.
        """
        for i in range(self.number_of_states):
            for j in range(self.number_of_actions):
                self.q[i, j] = 0.0
                self.n[i, j] = 0
        self.epsilon = eps
        self.learning_control(learning_enabled)

    def reset_episode(self):
        """ 
        Resets the episode-specific data (state-action pairs and rewards) for the agent.
        This should be called at the end of each episode to prepare for the next episode.
        """
        self.episode_sa.clear()
        self.episode_r.clear()

    def e_greedy(self, actions):
        """
        Uses epsilon-greedy action selection to choose an action.

        :param actions (List[int]): Array of integer-encoded actions for the current state.
        :return: Selected action index based on epsilon-greedy strategy.
        """
        a_star_idx = np.argmax(actions)
        rng = np.random.default_rng()
        if self.epsilon <= rng.random():
            return a_star_idx
        else:
            b = actions.size
            idx = rng.integers(low=0, high=b)
            return idx

    def rule_based_policy(self, state):
        """
        Implements a simple rule-based policy for the Blackjack game based on the player's 
        hand and the dealer's visible card.

        :param state (int): The current state of the game, encoded as an integer where:
                      - The hundreds digit indicates whether the player has a usable ace (1) or not (0).
                        - The tens digit indicates the dealer's visible card (1-10).
                        - The units digit indicates the player's current hand total (12-21).
        :return: Action (0 for stick, 1 for hit) based on the rules of the policy.
        """
        # Extract the components of the state
        ace = int(np.floor(state / 100))
        dealer_card = int(np.round((state % 100) / 10))
        player_total = int(state % 10 + 12)
        if ace == 1:
            if dealer_card in [1, 9, 10]:
                return 1 if player_total <= 18 else 0
            else:
                return 1 if player_total <= 17 else 0
        else:
            if dealer_card in [1, 7, 8, 9, 10]:
                return 1 if player_total <= 16 else 0
            elif dealer_card in [2, 3]:
                return 1 if player_total <= 12 else 0
            else:
                return 1 if player_total <= 11 else 0
            
    def select_action(self, state):
        """
        Selects an action for the given state using either a rule-based policy or epsilon-greedy strategy.

        :param state (int): The current state of the game.
        :return: Action (0 for stick, 1 for hit) based on the selected policy.
        """
        assert state < 200, "No action to perform since the game is already over."
        if self.rule_based:
            return self.rule_based_policy(state)
        A = self.q[state, ]
        action = self.e_greedy(A)
        self.episode_sa.append((state, action))
        return action

    def store_reward(self, reward):
        """
        Stores the reward for the current episode. This should be called after each action is taken and the 
        reward is received from the environment.

        :param reward (float): The reward received for the current episode.
        """
        self.episode_r.append(reward)

    def update_q(self):
        """ 
        Updates the Q-function using the first-visit Monte Carlo method based on the stored state-action pairs 
        and rewards for the episode. This method should be called at the end of each episode after all rewards 
        have been stored.
        """
        if not self.rule_based and self.learning_enabled:
            assert len(self.episode_sa) == len(self.episode_r), f"Number of state-action pairs ({len(self.episode_sa)}) and rewards ({len(self.episode_r)}) do not match for this episode."
            G = 0.0
            for i in range(len(self.episode_sa)-1, -1, -1):
                sa_pair = self.episode_sa[i]
                r = self.episode_r[i]
                G = self.gamma * G + r
                if self.episode_sa.index(sa_pair) == i:  # only update for first visit to this state-action pair
                    self.n[sa_pair] += 1
                    alpha = 1 / self.n[sa_pair]
                    q_old = self.q[sa_pair]
                    self.q[sa_pair] +=  alpha * (G - q_old)

    def print_q(self):
        """ 
        Helper function to print the Q-function values for all state-action pairs.
        This can be useful for debugging and understanding the learned values after training.
        """
        print("Q-function:")
        for i in range(self.number_of_states):
            for j in range(self.number_of_actions):
                print(f"Q[{i}, {j}] = {self.q[i, j]:.2f}")

    def print_n(self):
        """ 
        Helper function to print the visit counts for all state-action pairs.
        This can be useful for debugging and understanding how many times each state-action pair was visited during training.
        """
        print("Visit counts:")
        for i in range(self.number_of_states):
            for j in range(self.number_of_actions):
                print(f"N[{i}, {j}] = {self.n[i, j]}")
