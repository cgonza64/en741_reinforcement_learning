import Project1_env as env1

def unit_test():
    environment = env1.RobotGame()

    # Test environment initialization
    assert environment.get_number_of_states() == 11, "Number of states should be 11."
    assert environment.get_number_of_actions() == 4, "Number of actions should be 4."

    # Test reset method
    initial_state = environment.reset()
    assert initial_state == 8, "Initial state should be 8."

    # Test execute_action method
    new_state, reward, game_end = environment.execute_action(0)  # Action 'up'
    assert new_state == 5, "New state after action 'up' from state 8 should be 5."
    assert reward == -1.0, "Reward for moving to state 5 should be -1.0."
    assert not game_end, "Game should not end after moving to state 5."

    # Test get_state method
    current_state = environment.get_state()
    assert current_state == 5, "Current state should be 5."

    # Test helpers
    environment.set_state(4)
    assert environment.get_terminal_flag(), "State 4 should be terminal."

    reward = environment.get_reward(4)
    assert reward == 0.0, "Reward for terminal state should be 0.0."

    environment.set_state(1)
    assert not environment.get_terminal_flag(), "State 1 should not be terminal."

    reward = environment.get_reward(1)
    assert reward == -1.0, "Reward for state 1 should be -1.0."

    print("All unit tests passed.")

if __name__ == "__main__":
    unit_test()