import Project3_env
import Project3_agent
import numpy as np
from tqdm import tqdm


def play_games(env, agent, num_episodes: int = 500):
    """
    Plays a specified number of episodes in the environment using the given agent.
    Returns the total rewards for each episode.
    
    :param env (any): The environment in which the agent will interact (e.g., Blackjack).
    :param agent (any): The RL agent that will be playing the games, which should have methods for action selection, reward storage, and Q-function updates.
    :param num_episodes (int): The number of episodes to play.
    :return: A numpy array containing the total rewards for each episode.
    """
    returns = np.zeros(num_episodes, dtype="float64")
    for i in range(num_episodes):
        current_state = env.reset()
        if current_state < 200:  # only learn if the game isn't over yet
            game_end = False
            total_reward = 0.0
            while not game_end:
                action = agent.select_action(current_state)
                new_state, reward, game_end = env.execute_action(action)
                agent.store_reward(reward)
                current_state = new_state
            total_reward = sum(agent.episode_r)
            if agent.learning_enabled:
                agent.update_q()
            returns[i] = total_reward
            agent.reset_episode()
        else:
            if current_state == 201:
                returns[i] = -1.0
            elif current_state == 202:
                returns[i] = 0.0
            else:
                returns[i] = 1.0
    return returns

def train_and_evaluate(agent_type:str = "learning", 
                       num_agents:int = 100, 
                       num_tr_episodes:int = 5000, 
                       num_ts_episodes:int = 100,
                       use_eps_decay:bool = False,
                       debug:bool = False):
    """
    Trains and evaluates a specified type of agent (learning, random, or rule-based) in the Blackjack environment.
    The function trains the agent for a specified number of episodes, then evaluates its performance in testing episodes.
    The returns from both training and testing are saved to a CSV file for later analysis and plotting.
    
    :param agent_type (str): The type of agent to train and evaluate (e.g., "learning", "random", "rule_based").
    :param num_agents (int): The number of agents to train and evaluate.
    :param num_tr_episodes (int): The number of training episodes for each agent.
    :param num_ts_episodes (int): The number of testing episodes for each agent.
    :param use_eps_decay (bool): If True, enables epsilon decay for learning agents.
    :param debug (bool): If True, print average returns and standard deviations.
    """
    env = Project3_env.IndianaJonesAdventure()
    eps = 1.0 if agent_type == "random" else 0.2
    is_rule_based = True if agent_type == "rule_based" else False
    all_returns = np.zeros((num_agents, num_tr_episodes + num_ts_episodes), dtype="float64")
    for i in tqdm(range(num_agents)):
        agent = Project3_agent.RLAgent(rule_based=is_rule_based, epsilon=eps, eps_decay=use_eps_decay)
        G_tr = play_games(env, agent, num_tr_episodes)
        agent.learning_control(False)  # Disable learning for testing
        G_ts = play_games(env, agent, num_ts_episodes)
        all_returns[i, :num_tr_episodes] = G_tr
        all_returns[i, num_tr_episodes:] = G_ts

    if debug:
        print_avg_returns(agent_type, all_returns, num_tr_episodes)

    # Save the returns to a CSV file
    filename = f"{agent_type}_agent_returns.csv"
    np.savetxt(filename, all_returns, delimiter=",")

def print_avg_returns(agent_type, all_returns, num_tr_episodes):
    """
    Prints the average returns and standard deviations for both training and testing phases of the specified agent type.
    
    :param agent_type (str): The type of agent being evaluated.
    :param all_returns (np.ndarray): The returns for all agents and episodes.
    :param num_tr_episodes (int): The number of training episodes.
    """
    # Print average returns and standard deviations
    if agent_type == "rule_based":
        # No training for rule-based agent
        avg_return_ts = np.mean(all_returns[:, num_tr_episodes:])
        std_return_ts = np.std(all_returns[:, num_tr_episodes:])
        print(f"Average return during testing: {avg_return_ts}, standard deviation: {std_return_ts}")
    else:
        avg_G_i_tr = np.mean(all_returns[:, :num_tr_episodes], axis=0)
        avg_G_i_ts = np.mean(all_returns[:, num_tr_episodes:], axis=0)
        print(f"Average return during training: {np.mean(avg_G_i_tr)}, standard deviation: {np.std(avg_G_i_tr)}")
        print(f"Average return during testing: {np.mean(avg_G_i_ts)}, standard deviation: {np.std(avg_G_i_ts)}")

def main(eps_decay_learning_agents=False):
    num_agents = 100
    for agent_type in ["learning", "random"]:
        print(f"\nTraining and evaluating {agent_type} agents...")
        use_eps_decay = eps_decay_learning_agents if agent_type == "learning" else False
        train_and_evaluate(agent_type=agent_type, 
                           num_agents=num_agents, 
                           num_tr_episodes=5000, 
                           num_ts_episodes=100, 
                           use_eps_decay=use_eps_decay, 
                           debug=True)
        print("\n" + "="*50 + "\n")
    
    # Rule-based agent evaluation
    print("Evaluating rule-based agent...")
    train_and_evaluate(agent_type="rule_based", 
                       num_agents=1, 
                       num_tr_episodes=0, 
                       num_ts_episodes=100, 
                       use_eps_decay=False, 
                       debug=True)

if __name__ == "__main__":
    main(eps_decay_learning_agents=False)
