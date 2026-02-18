import math
import Project2_agent
import Project2_env

def test_q_function_update():
    """ Validates the q-values for 3 episodes from the From Fred the Robot example """
    A = ['up', 'left', 'down', 'right']
    agent = Project2_agent.RLAgent(gamma=0.9, num_of_states=12, num_of_actions=4)

    # Episode 1
    state_action_pairs = [
        (8, A.index('up')),  
        (5, A.index('right')),  
        (6, A.index('up')),  
        (2, A.index('left')),
        (1, A.index('right')),
        (2, A.index('right')),
        (3, A.index('right'))
    ]
    rewards = [-1, -1, -1, -1, -1, -1, 25]
    for (state, action), reward in zip(state_action_pairs, rewards):
        agent.episode_sa.append((state, action))
        agent.store_reward(reward)
    agent.update_q()

    assert math.isclose(agent.q[8, A.index('up')], 8.6, abs_tol=1e-1), f"Expected Q-value for (8, 'up') to be 8.6, got {agent.q[8, A.index('up')]}"
    assert math.isclose(agent.q[5, A.index('right')], 10.7, abs_tol=1e-1), f"Expected Q-value for (5, 'right') to be 10.7, got {agent.q[5, A.index('right')]}"
    assert math.isclose(agent.q[6, A.index('up')], 13.0, abs_tol=1e-1), f"Expected Q-value for (6, 'up') to be 13.0, got {agent.q[6, A.index('up')]}"
    assert math.isclose(agent.q[2, A.index('left')], 15.5, abs_tol=1e-1), f"Expected Q-value for (2, 'left') to be 15.5, got {agent.q[2, A.index('left')]}"
    assert math.isclose(agent.q[1, A.index('right')], 18.4, abs_tol=1e-1), f"Expected Q-value for (1, 'right') to be 18.4, got {agent.q[1, A.index('right')]}"
    assert math.isclose(agent.q[2, A.index('right')], 21.5, abs_tol=1e-1), f"Expected Q-value for (2, 'right') to be 21.5, got {agent.q[2, A.index('right')]}"
    assert math.isclose(agent.q[3, A.index('right')], 25.0, abs_tol=1e-1), f"Expected Q-value for (3, 'right') to be 25.0, got {agent.q[3, A.index('right')]}"

    # Episode 2
    agent.reset_episode()
    state_action_pairs = [
        (8, A.index('right')),  
        (9, A.index('left')),  
        (8, A.index('right')),  
        (9, A.index('left')),
        (10, A.index('right')),
        (11, A.index('up'))
    ]
    rewards = [-1, -1, -1, -1, -1, -25]
    for (state, action), reward in zip(state_action_pairs, rewards):
        agent.episode_sa.append((state, action))
        agent.store_reward(reward)
    agent.update_q()
    assert math.isclose(agent.q[8, A.index('right')], -18.9, abs_tol=1e-1), f"Expected Q-value for (8, 'right') to be -18.9, got {agent.q[8, A.index('right')]}"
    assert math.isclose(agent.q[9, A.index('left')], -19.8, abs_tol=1e-1), f"Expected Q-value for (9, 'left') to be 19.8, got {agent.q[9, A.index('left')]}"
    assert math.isclose(agent.q[10, A.index('right')], -23.5, abs_tol=1e-1), f"Expected Q-value for (10, 'right') to be -23.5, got {agent.q[10, A.index('right')]}"
    assert math.isclose(agent.q[11, A.index('up')], -25.0, abs_tol=1e-1), f"Expected Q-value for (11, 'up') to be -25.0, got {agent.q[11, A.index('up')]}"

    # Episode 3
    agent.reset_episode()
    state_action_pairs = [
        (8, A.index('up')),  
        (5, A.index('right')),  
        (6, A.index('up')),  
        (2, A.index('right')),
        (3, A.index('right'))
    ]
    rewards = [-1, -1, -1, -1, 25]
    for (state, action), reward in zip(state_action_pairs, rewards):
        agent.episode_sa.append((state, action))
        agent.store_reward(reward)
    agent.update_q()

    print("\nUpdated Q-values and n-values after Episode 3:")
    assert math.isclose(agent.q[8, A.index('up')], 10.8, abs_tol=1e-1), f"Expected Q-value for (8, 'up') to be 10.8, got {agent.q[8, A.index('up')]}"
    assert math.isclose(agent.q[5, A.index('right')], 13.1, abs_tol=1e-1), f"Expected Q-value for (5, 'right') to be 13.1, got {agent.q[5, A.index('right')]}"
    assert math.isclose(agent.q[6, A.index('up')], 15.7, abs_tol=1e-1), f"Expected Q-value for (6, 'up') to be 15.7, got {agent.q[6, A.index('up')]}"
    assert math.isclose(agent.q[2, A.index('right')], 21.5, abs_tol=1e-1), f"Expected Q-value for (2, 'right') to be 21.5, got {agent.q[2, A.index('right')]}"
    assert math.isclose(agent.q[3, A.index('right')], 25.0, abs_tol=1e-1), f"Expected Q-value for (3, 'right') to be 25.0, got {agent.q[3, A.index('right')]}"

    print("\nq-function update test completed successfully.")
    # agent.print_q()
    # agent.print_n()

def test_environment_agent_interaction():
    env = Project2_env.Blackjack()
    agent = Project2_agent.RLAgent()

    # Check that the environment parameters match
    if (env.get_number_of_states() == agent.get_number_of_states()) and \
            (env.get_number_of_actions() == agent.get_number_of_actions()):
        print("Environment and Agent parameters match. Starting test.")
        env.reset()
        env.get_state()

        # Play 100 games
        for i in range(100):
            current_state = env.reset()
            if current_state < 200:  # only play if the game isn't over yet
                game_end = False
                total_reward = 0.0
                while not game_end:
                    action = agent.select_action(current_state)
                    new_state, reward, game_end = env.execute_action(action)
                    agent.store_reward(reward)
                    current_state = new_state
                total_reward = sum(agent.episode_r)
                agent.update_q()
                print(f"Episode {i+1} completed. Total reward: {total_reward}")
                agent.reset_episode()

        # Disable learning for testing
        agent.learning_control(False)
        for i in range(10):
            current_state = env.reset()
            game_end = False
            total_reward = 0.0
            while not game_end:
                action = agent.select_action(current_state)
                new_state, reward, game_end = env.execute_action(action)
                current_state = new_state
                total_reward += reward
            print(f"Test Episode {i+1} completed. Total reward: {total_reward}")

        print("\nEnvironment-agent interaction test completed successfully.")
        # agent.print_q()
        # agent.print_n()

if __name__ == "__main__":
    test_q_function_update()
    test_environment_agent_interaction()
