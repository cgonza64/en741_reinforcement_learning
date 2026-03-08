import Project3_env
import Project3_agent
import numpy as np
from tqdm import tqdm


def play_games(env, agent, num_episodes: int = 500):
    """
    Plays a specified number of episodes in the environment using the given agent.
    Returns the total rewards for each episode.
    
    :param env (any): The environment in which the agent will interact (e.g., IndianaJonesAdventure).
    :param agent (any): The RL agent that will be playing the games that uses either SARSA or Q-learning.
    :param num_episodes (int): The number of episodes to play.
    :return: A numpy array containing the total rewards for each episode.
    """
    returns = np.zeros(num_episodes, dtype="float64")
    for i in range(num_episodes):
        cumulative_rewards = 0.0
        current_state = env.reset()
        a = agent.select_action(current_state)
        game_end = False
        while not game_end:
            next_state, reward, game_end = env.execute_action(a)
            cumulative_rewards += reward
            a_next = agent.update_q(current_state, a, reward, next_state)
            current_state = next_state
            if agent.agent_type == 'SARSA':
                a = a_next
            elif agent.agent_type == 'Q-learning':
                a = agent.select_action(current_state)
        returns[i] = cumulative_rewards

    return returns

def train_and_evaluate(agent_type:str = "SARSA", 
                       num_agents:int = 20, 
                       num_tr_episodes:int = 500, 
                       use_eps_decay:bool = False,
                       debug:bool = False):
    """
    Trains and evaluates a specified type of agent (SARSA, Q-learning) in the IndianaJonesAdventure environment.
    The function trains the agent for a specified number of episodes, then evaluates its performance in testing episodes.
    The returns from both training and testing are saved to a CSV file for later analysis and plotting.
    
    :param agent_type (str): The type of agent to train and evaluate (e.g., "SARSA", "Q-learning").
    :param num_agents (int): The number of agents to train and evaluate.
    :param num_tr_episodes (int): The number of training episodes for each agent.
    :param use_eps_decay (bool): If True, enables epsilon decay for learning agents.
    :param debug (bool): If True, print average returns and standard deviations.
    """
    env = Project3_env.IndianaJonesAdventure()
    all_returns = np.zeros((num_agents, num_tr_episodes), dtype="float64")
    for i in tqdm(range(num_agents)):
        agent = Project3_agent.RLAgent(agent_type=agent_type, eps_decay=use_eps_decay)
        G_tr = play_games(env, agent, num_tr_episodes)
        all_returns[i] = G_tr
    if debug:
        # Print average returns and standard deviations
        avg_G_i_tr = np.mean(all_returns, axis=0)
        print(f"Average return during training: {np.mean(avg_G_i_tr)}, standard deviation: {np.std(avg_G_i_tr)}")

    # Save the returns to a CSV file
    using_eps_decay_str = "eps_decay_" if use_eps_decay else ""
    filename = f"{using_eps_decay_str}{agent_type}_agent_returns.csv"
    np.savetxt(filename, all_returns, delimiter=",")

    # Save the last trained agent's policy for later analysis
    np.save(f"{using_eps_decay_str}{agent_type}_agent_policy.npy", agent.return_policy())

def main(use_eps_decay=False):
    num_agents = 20
    for use_eps_decay in [False, True]:
        using_eps_decay_str = "with Epsilon Decay" if use_eps_decay else ""
        for agent_type in ["SARSA", "Q-learning"]:
            print(f"\nTraining and evaluating {agent_type} agents {using_eps_decay_str}...")
            train_and_evaluate(agent_type=agent_type, 
                            num_agents=num_agents, 
                            num_tr_episodes=500, 
                            use_eps_decay=use_eps_decay, 
                            debug=True)
            print("\n" + "="*50 + "\n")
    

if __name__ == "__main__":
    main(use_eps_decay=False)
