import pandas as pd
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
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv("project1_data.csv", header=None)
    episodes = data[0].tolist()
    returns = data[1].tolist()
    plot_learning_curve(episodes, returns)