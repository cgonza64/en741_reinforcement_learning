# Globals
ANTIQUITY_STATE = 65
BETA_STATE = 83
MUSEUM_STATE = 95
SWAMP_STATES = list(range(84, 94)) + list(range(52, 62))
TEMPLE_BOUNDARY_LEFT = [0, 16, 32, 48, 64, 80]  # Left boundary when inside the temple
TEMPLE_BOUNDARY_RIGHT = [2, 18, 34, 50, 66, 82]  # Right boundary when inside the temple
OUTSIDE_BOUNDARY_LEFT = [3, 19, 35, 51, 67, 83]  # Left boundary when outside the temple
OUTSIDE_BOUNDARY_RIGHT = [15, 31, 47, 63, 79, 95]  # Right boundary when outside the temple

class IndianaJonesAdventure:
    def __init__(self):
        """ Initializes the Indiana Jones Adventure environment. """
        self.num_states = 96
        self.action_space = {0: 'N', 1: 'S', 2: 'W', 3: 'E'}
        self.num_actions = len(self.action_space)
        self.current_state = 0

    def get_number_of_states(self):
        """ Returns the total number of states in the environment. """
        return self.num_states

    def get_number_of_actions(self):
        """ Returns the total number of actions in the environment. """
        return self.num_actions

    def get_state(self):
        """ Returns the current state of the environment. """
        return self.current_state
    
    def reset(self, start_state=0, es_flag=False):
        """ Resets the environment to the specified start state (default is 0) 
            and returns the initial state.
        """
        self.current_state = start_state
        return start_state

    def execute_action(self, action):
        """ Executes the given action in the environment, updates the current state, 
            and returns the new state, reward, and game end status.
            :param action: An integer representing the action to take (0: North, 1: South, 2: West, 3: East).
            :return: A tuple (new_state, reward, game_end) where:
                - new_state: The updated state after executing the action.
                - reward: The reward received for taking the action.
                - game_end: A boolean indicating whether the game has ended.
        """
        game_end = False

        # Determine next state
        if action == 0:  # Go North
            if self.current_state < 80:
                self.current_state += 16
        elif action == 1:  # Go South
            if self.current_state > 15:
                self.current_state -= 16
        elif action == 2:  # Go West
            if self.current_state not in TEMPLE_BOUNDARY_LEFT and self.current_state not in OUTSIDE_BOUNDARY_LEFT:
                self.current_state -= 1
        elif action == 3:  # Go East
            if self.current_state not in TEMPLE_BOUNDARY_RIGHT and self.current_state not in OUTSIDE_BOUNDARY_RIGHT:
                self.current_state += 1

        # Special state handling and rewards
        reward = 0.0
        if self.current_state == ANTIQUITY_STATE:  # Found the antiquity?
            reward += 4.0
            self.current_state = BETA_STATE
        elif self.current_state in SWAMP_STATES:  # Fell into the swamp?
            reward -= 100.0
            self.current_state = BETA_STATE
        elif self.current_state == MUSEUM_STATE:  # Reached the museum? (Yay!)
            game_end = True
            reward -= 1.0
        else:
            reward -= 1.0

        return self.current_state, reward, game_end
    
    def set_state(self, state):
        """ Set the current state of the environment. """
        self.current_state = state

if __name__=='__main__':
    env = IndianaJonesAdventure()
    current_st = env.reset()
    assert current_st == 0

    # Test all actions
    current_st, _, _ = env.execute_action(0)  # North
    assert current_st == 16
    current_st, _, _ = env.execute_action(3)  # East
    assert current_st == 17
    current_st, _, _ = env.execute_action(2)  # West
    assert current_st == 16
    current_st, reward, _ = env.execute_action(1)  # South
    assert current_st == 0
    assert reward == -1.0

    # Test antiquity state
    current_st = env.reset()
    for a in [0, 0, 0, 0, 3]:  # North, North, North, North, East
        current_st, reward, _ = env.execute_action(a)
    assert current_st == BETA_STATE
    assert reward == 4.0

    # Test Swamp states
    current_st, reward, _ = env.execute_action(3)  # Fall into the swamp
    assert current_st == BETA_STATE
    assert reward == -100.0
    env.set_state(70)  # On the bridge
    current_st, reward, _ = env.execute_action(1)  # Fall into the swamp
    assert current_st == BETA_STATE
    assert reward == -100.0

    # Test world boundaries
    env.set_state(32)
    current_st, _, _ = env.execute_action(2)
    assert current_st == 32
    env.set_state(35)
    current_st, _, _ = env.execute_action(2)
    assert current_st == 35
    env.set_state(79)
    current_st, _, _ = env.execute_action(3)
    assert current_st == 79
    env.set_state(66)
    current_st, _, _ = env.execute_action(3)
    assert current_st == 66
    env.set_state(94)
    current_st, _, _ = env.execute_action(0)
    assert current_st == 94
    env.set_state(8)
    current_st, _, _ = env.execute_action(1)
    assert current_st == 8

    print("The environment for the Indiana Jones' Adventure puzzle works!")

