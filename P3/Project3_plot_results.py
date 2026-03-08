import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from display_policy import draw_policy_on_game_grid

def plot_learning_curves(x, 
                        y, 
                        title="Fred the Robot Learning Curve (100 episodes)", 
                        xlabel="Episode", 
                        ylabel="Return (Total Reward)"):
    """
    Plots a learning curve.
    
    Args:
        x: List or array of x-coordinates
        y: Dictionary of y-coordinates, where keys are RL agent types and values are lists or arrays of returns
        title: Title of the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        color: Color of the line
    """
    plt.figure(figsize=(14, 5))
    keys = y.keys()
    for key in keys:
        plt.plot(x, y[key], label=key, linewidth=2)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.ylabel(ylabel, fontsize=16)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)
    plt.show()

def smoothing_function(data, window_size=21):
    """
    Applies a moving average smoothing function to the data.
    
    Args:
        data: List or array of data points to smooth
        window_size: Size of the moving average window
    
    Returns:
        Smoothed data as a numpy array
    """
    if len(data) < window_size:
        return np.array(data)  # Not enough data to smooth, return original
    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    smoothed_data = np.concatenate((smoothed_data, smoothed_data[-window_size+1:]))  # Pad the boundaries to maintain original length
    return smoothed_data

def main():
    for eps_decay in [False, True]:
        eps_decay_str = "eps_decay_" if eps_decay else ""
        smoothed_returns = {'SARSA': None, 'Q-learning': None}
        num_tr_episodes = 500
        for agent_type in ["SARSA", "Q-learning"]:
            print(f"\n{agent_type} agents:")
            filename = f"{eps_decay_str}{agent_type}_agent_returns.csv"
            data = pd.read_csv(filename, header=None)
            episodes_tr = list(range(1, num_tr_episodes + 1))
            avg_G_i_tr = np.mean(data.values[:, :num_tr_episodes], axis=0)
            smoothed_returns[agent_type] = smoothing_function(avg_G_i_tr, window_size=11)
            print(f"Average return during training: {np.mean(avg_G_i_tr):.4f}, standard deviation: {np.std(avg_G_i_tr):.6f}")

            # Draw the learned policy for the agent        
            draw_policy_on_game_grid(np.load(f"{eps_decay_str}{agent_type}_agent_policy.npy"),
                                     agent_type=agent_type,
                                     eps_decay=eps_decay)

            print("\n" + "="*50 + "\n")

        # Plot learning curves for both agents
        plot_title = f"Learning Curves over {num_tr_episodes} Training Episodes"
        if eps_decay:
            plot_title += f" (with Epsilon Decay)"
        plot_learning_curves(episodes_tr, 
                            smoothed_returns, 
                            title=plot_title, 
                            xlabel="Training Episode",
                            ylabel="Avg. Return G_i (Smoothed)")

if __name__ == "__main__":
    main()