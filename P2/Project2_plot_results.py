import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, 
                        y, 
                        title="Fred the Robot Learning Curve (100 episodes)", 
                        xlabel="Episode", 
                        ylabel="Return (Total Reward)", 
                        color="blue"):
    """
    Plots a learning curve.
    
    Args:
        x: List or array of x-coordinates
        y: List or array of y-coordinates
        title: Title of the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        color: Color of the line
    """
    plt.figure(figsize=(14, 5))
    plt.plot(x, y, color=color)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.ylabel(ylabel, fontsize=16)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
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
    smoothed_data = np.concatenate((smoothed_data[:window_size//2 - 1], smoothed_data, smoothed_data[-window_size//2:]))  # Pad the boundaries to maintain original length
    return smoothed_data

def main(eps_decay=False):
    for agent_type in ["learning", "random"]:
        print(f"\n{agent_type} agents:")
        filename = f"{agent_type}_agent_returns.csv"
        num_tr_episodes = 5000
        data = pd.read_csv(filename, header=None)
        episodes_tr = list(range(1, num_tr_episodes + 1))
        avg_G_i_tr = np.mean(data.values[:, :num_tr_episodes], axis=0)
        smoothed_G_i_tr = smoothing_function(avg_G_i_tr, window_size=201)
        avg_G_i_ts = np.mean(data.values[:, num_tr_episodes:], axis=0)
        print(f"Average return during training: {np.mean(smoothed_G_i_tr):.4f}, standard deviation: {np.std(smoothed_G_i_tr):.6f}")
        print(f"Average return during testing: {np.mean(avg_G_i_ts):.4f}, standard deviation: {np.std(avg_G_i_ts):.6f}")
        algo_type = "MC-Based RL" if agent_type == "learning" else "Random"
        plot_title = f"Learning Curve for {algo_type} Agents"
        if eps_decay:
            plot_title += f" (with Epsilon Decay)"
        plot_learning_curve(episodes_tr, 
                            smoothed_G_i_tr, 
                            title=plot_title, 
                            xlabel="Training Episode",
                            ylabel="Avg. Return G_i (Smoothed)", 
                            color="blue")
        print("\n" + "="*50 + "\n")
    
    # Rule-based agent's testing returns
    print("Rule-Based agent:")
    data = pd.read_csv("rule_based_agent_returns.csv", header=None)
    avg_return_ts = np.mean(data)
    std_return_ts = np.std(data)
    print(f"Average return during testing: {avg_return_ts:.4f}, standard deviation: {std_return_ts:.6f}")

if __name__ == "__main__":
    main(eps_decay=False)