class RobotGame:
    """
        Implements a simple environment for the Fred the Robot game. 
        The environment consists of 11 states (1 to 11) where states 4 and 7 are terminal.
        The transition function and rewards are deterministic. 
        The available actions are: 0 = up, 1 = down, 2 = left, and 3 = right
    """

    def __init__(self, start_state=8):
        """ Initialize the RobotGame environment. The initial state defaults to 8). """
        self.current_state = start_state
        self.action_space = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
        
        # Determinisitic transition function
        self.tf = {
            1: {'u': 1, 'd': 5, 'l': 1, 'r': 2},
            2: {'u': 2, 'd': 6, 'l': 1, 'r': 3},
            3: {'u': 3, 'd': 3, 'l': 2, 'r': 4},
            4: {'u': 4, 'd': 4, 'l': 4, 'r': 4},
            5: {'u': 1, 'd': 8, 'l': 5, 'r': 6},
            6: {'u': 2, 'd': 9, 'l': 5, 'r': 6},
            7: {'u': 7, 'd': 7, 'l': 7, 'r': 7},
            8: {'u': 5, 'd': 8, 'l': 8, 'r': 9},
            9: {'u': 6, 'd': 9, 'l': 8, 'r': 10},
            10: {'u': 10, 'd': 10, 'l': 9, 'r': 11},
            11: {'u': 7, 'd': 11, 'l': 10, 'r': 11}
        }

        # Deterministic rewards
        self.rewards = {
            1: -1.0,
            2: -1.0,
            3: -1.0,
            4: 25.0,
            5: -1.0,
            6: -1.0,
            7: -25.0,
            8: -1.0,
            9: -1.0,
            10: -1.0,
            11: -1.0
        }

        self.num_states = len(self.tf)
        self.num_actions = len(self.action_space)

    def get_number_of_states(self):
        """ Returns the (constant) integer number of states. """
        return self.num_states

    def get_number_of_actions(self):
        """ Returns the (constant) integer number of actions. """
        return self.num_actions

    def reset(self, start_state=8, es_flag=False):
        """
            Resets the environment to the beginning of the episode. The es_flag 
            is currently not used, but can be implemented to choose a random 
            starting state.
        """
        self.current_state = start_state
        return start_state

    def execute_action(self, action):
        """
            Executes the action given by action. Causes the next state to be 
            determined, the state of the environment to be updated, and the 
            applicable reward to be calculated. Returns the new state, the reward, & the 
            terminal flag.
        """
        a = self.action_space[action]
        next_state = self.tf[self.current_state][a]
        reward = self.get_reward(next_state)
        self.set_state(next_state)
        game_end = self.get_terminal_flag()
        return next_state, reward, game_end
    
    def get_state(self):
        """ Return the current state of the environment. """
        return self.current_state
    
    # Helper methods
    def get_reward(self, state):
        """ Return the reward for the given state. """
        if self.get_terminal_flag():
            return 0.0
        else:
            return self.rewards[state]

    def get_terminal_flag(self):
        """ Return True if the current state is terminal, False otherwise. """
        return self.current_state in [4, 7]
    
    def set_state(self, state):
        """ Set the current state of the environment. """
        self.current_state = state
