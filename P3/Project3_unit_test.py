import math
import Project3_agent
import Project3_env
import display_policy

def test_q_function_update():
    """ Validates the q-values for 2 episodes from the From Fred the Robot example """
    A = ['up', 'left', 'down', 'right']
    agent = Project3_agent.RLAgent(alpha=0.1,gamma=0.9, num_of_states=11, num_of_actions=4)

    # Episode 1
    trajectory = [
        (8, A.index('up')),  
        (5, A.index('right')),  
        (6, A.index('up')),  
        (2, A.index('left')),
        (1, A.index('right')),
        (2, A.index('right')),
        (3, A.index('right'))
    ]
    rewards = [-1, -1, -1, -1, -1, -1, 25]

    for i in range(len(trajectory)):
        s, a = trajectory[i]
        r = rewards[i]
        s_prime, a_prime = trajectory[i+1] if (i+1) < len(trajectory) else (4, None)
        agent.update_q(s, a, r, s_prime, a_prime)

    assert math.isclose(agent.q[8, A.index('up')], -0.1, abs_tol=1e-1), f"Expected Q-value for (8, 'up') to be -0.1, got {agent.q[8, A.index('up')]}"
    assert math.isclose(agent.q[5, A.index('right')], -0.1, abs_tol=1e-1), f"Expected Q-value for (5, 'right') to be -0.1, got {agent.q[5, A.index('right')]}"
    assert math.isclose(agent.q[6, A.index('up')], -0.1, abs_tol=1e-1), f"Expected Q-value for (6, 'up') to be -0.1, got {agent.q[6, A.index('up')]}"
    assert math.isclose(agent.q[2, A.index('left')], -0.1, abs_tol=1e-1), f"Expected Q-value for (2, 'left') to be -0.1, got {agent.q[2, A.index('left')]}"
    assert math.isclose(agent.q[1, A.index('right')], -0.1, abs_tol=1e-1), f"Expected Q-value for (1, 'right') to be -0.1, got {agent.q[1, A.index('right')]}"
    assert math.isclose(agent.q[2, A.index('right')], -0.1, abs_tol=1e-1), f"Expected Q-value for (2, 'right') to be -0.1, got {agent.q[2, A.index('right')]}"
    assert math.isclose(agent.q[3, A.index('right')], 2.5, abs_tol=1e-1), f"Expected Q-value for (3, 'right') to be 2.5, got {agent.q[3, A.index('right')]}"

    # Episode 2
    for i in range(len(trajectory)):
        s, a = trajectory[i]
        r = rewards[i]
        s_prime, a_prime = trajectory[i+1] if (i+1) < len(trajectory) else (4, None)
        agent.update_q(s, a, r, s_prime, a_prime)

    assert math.isclose(agent.q[8, A.index('up')], -0.199, abs_tol=1e-3), f"Expected Q-value for (8, 'up') to be -0.199, got {agent.q[8, A.index('up')]}"
    assert math.isclose(agent.q[5, A.index('right')], -0.199, abs_tol=1e-3), f"Expected Q-value for (5, 'right') to be -0.199, got {agent.q[5, A.index('right')]}"
    assert math.isclose(agent.q[6, A.index('up')], -0.199, abs_tol=1e-3), f"Expected Q-value for (6, 'up') to be -0.199, got {agent.q[6, A.index('up')]}"
    assert math.isclose(agent.q[2, A.index('left')], -0.199, abs_tol=1e-3), f"Expected Q-value for (2, 'left') to be -0.199, got {agent.q[2, A.index('left')]}"
    assert math.isclose(agent.q[1, A.index('right')], -0.199, abs_tol=1e-3), f"Expected Q-value for (1, 'right') to be -0.199, got {agent.q[1, A.index('right')]}"
    assert math.isclose(agent.q[2, A.index('right')], 0.035, abs_tol=1e-3), f"Expected Q-value for (2, 'right') to be 0.035, got {agent.q[2, A.index('right')]}"
    assert math.isclose(agent.q[3, A.index('right')], 4.750, abs_tol=1e-3), f"Expected Q-value for (3, 'right') to be 4.750, got {agent.q[3, A.index('right')]}"

    print("\nq-function update test completed successfully.")
    # agent.print_q()

def test_environment_agent_interaction():
    # SARSA agent
    agent_type = 'SARSA'
    agent = Project3_agent.RLAgent(agent_type=agent_type)
    env = Project3_env.IndianaJonesAdventure()
    for ep in range(500):
        current_state = env.reset()
        a = agent.select_action(current_state)
        done = False
        while not done:
            next_state, reward, done = env.execute_action(a)
            a_next = agent.update_q(current_state, a, reward, next_state)
            current_state = next_state
            a = a_next

    # display learned policy
    pi = agent.return_policy()
    display_policy.draw_policy_on_game_grid(pi, agent_type=agent_type)

    # Q-learning agent
    agent_type = 'Q-learning'
    agent = Project3_agent.RLAgent(agent_type=agent_type)
    env = Project3_env.IndianaJonesAdventure()
    for ep in range(500):
        current_state = env.reset()
        a = agent.select_action(current_state)
        done = False
        while not done:
            next_state, reward, done = env.execute_action(a)
            agent.update_q(current_state, a, reward, next_state)
            current_state = next_state
            a = agent.select_action(current_state)

    # display learned policy
    pi = agent.return_policy()
    display_policy.draw_policy_on_game_grid(pi, agent_type=agent_type)

    print("\nEnvironment-agent interaction test completed successfully.")
    print("Learned Q-values after 100 episodes:")
    # agent.print_q()
    

if __name__ == "__main__":
    test_q_function_update()
    test_environment_agent_interaction()
