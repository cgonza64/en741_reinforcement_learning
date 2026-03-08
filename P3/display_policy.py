import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrow
import numpy as np
from PIL import Image

def draw_policy_on_game_grid(pi, agent_type='SARSA'):
    """
    Draws the learned policy on the game grid with directional arrows.
    """
    # Grid dimensions
    rows = 6
    cols = 17  # +1 for temple/outside separation
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(17, 6))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis so (0,0) is top-left

    # 'S' = Start (green), 'E' = End (red), 'B' = Blocked (gray), ' ' = Empty (white with arrow)
    grid_layout = [
        [' ', ' ', ' ', 'E', 'B', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', ' ', 'M'],
        [' ', 'A', ' ', 'E', ' ', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', ' ', ' '],
        [' ', ' ', ' ', 'E', ' ', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', ' ', ' '],
        [' ', ' ', ' ', 'E', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', 'E', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        ['I', ' ', ' ', 'E', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    ]
    
    # Direction arrows for empty cells (cycle through N, S, E, W)
    directions = {0: 'N', 1: 'S', 2: 'W', 3: 'E'}
    direction_idx = 0

    # map the learned policy to a 2D grid format
    policy = map_policy(pi)
    
    # Draw grid cells
    for row in range(rows):
        for col in range(cols):
            cell_type = grid_layout[row][col]
            
            # Determine cell color
            if cell_type == 'I':  # Start location
                color =  "#24B147"
                edge_color = "#082908"
                linewidth = 3
            elif cell_type == 'M':  # Museum location
                color = '#FFB6C6'
                edge_color = '#DC143C'
                linewidth = 3
            elif cell_type == 'A':  # Antiquity location
                color = "#B48624"
                edge_color = '#000000'
                linewidth = 1
            elif cell_type == 'B':  # Beta location
                color = "#103EA1"
                edge_color = '#000000'
                linewidth = 1
            elif cell_type == 'S':  # Swamp space
                color = "#6A8825"
                edge_color = '#000000'
                linewidth = 1
            elif cell_type == 'R':  # Rope and plank bridge
                color = "#66430DC3"
                edge_color = '#000000'
                linewidth = 1
            elif cell_type == 'E': # temple boundary
                color = '#FFFFFF'
                edge_color = '#FFFFFF'
                linewidth = 1
            else:  # Empty (traversable) space
                color = "#DCEBF0"
                edge_color = '#000000'
                linewidth = 1
            
            # Draw rectangle for cell
            facecolor = 'none' if cell_type in ['I', 'M', 'A', 'B', 'S'] else color
            w = 0.5 if cell_type == 'E' else 1.0
            rect = patches.Rectangle((col, row), w, 1, 
                                     linewidth=linewidth, 
                                     edgecolor=edge_color, 
                                     facecolor=facecolor)
            ax.add_patch(rect)
            
            # Add arrow or image (if applicable)
            image = None
            if cell_type == 'I':
                image = Image.open('D:\\Grad School\\Reinforcment_Learning_S26\\Project3\\game_tokens\\indiana_jones.png')
            elif cell_type == 'M':  # Museum token
                image = Image.open('D:\\Grad School\\Reinforcment_Learning_S26\\Project3\\game_tokens\\museum.png')
            elif cell_type == 'A':
                image = Image.open('D:\\Grad School\\Reinforcment_Learning_S26\\Project3\\game_tokens\\antiquity.png')
            elif cell_type == 'B':
                image = Image.open('D:\\Grad School\\Reinforcment_Learning_S26\\Project3\\game_tokens\\beta_space.png')
            elif cell_type == 'S':
                image = Image.open('D:\\Grad School\\Reinforcment_Learning_S26\\Project3\\game_tokens\\swamp_space.png')
            # elif cell_type == 'R':
            #     image = Image.open('D:\\Grad School\\Reinforcment_Learning_S26\\Project3\\game_tokens\\rope_and_plank_bridge_space.png')
            elif cell_type == ' ' or cell_type == 'R':
                # Draw directional arrow
                y = col if col < 3 else col - 1  # Adjust row index for temple boundary
                direction_idx = policy[row, y]
                direction = directions[direction_idx]
                draw_arrow(ax, col, row, direction)

            # Render image
            if image is not None:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                img = np.array(image)
                im = ax.imshow(img, extent=[col, col + 1, row, row + 1], origin='upper')
                im.set_clip_path(rect)
    
    # Remove axis ticks and labels
    ax.set_xticks([i + 0.5 for i in range(cols)])
    ax.set_xticklabels(list(range(0, 3)) + [' '] + list(range(3, cols - 1)))
    ax.set_yticks([i + 0.5 for i in range(rows)])
    ax.set_yticklabels([0, 16, 32, 48, 64, 80][::-1])
    ax.grid(False)
    
    plt.title('Learned Policy For ' + agent_type + " Agent", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def draw_arrow(ax, col, row, direction):
    """
    Draw a directional arrow in the specified cell
    
    Args:
        ax: matplotlib axis
        col: column index
        row: row index
        direction: 'N', 'S', 'E', or 'W'
    """
    center_x = col + 0.5
    center_y = row + 0.5
    arrow_length = 0.3
    
    # Define arrow direction vectors
    if direction == 'N':
        dx, dy = 0, -arrow_length
    elif direction == 'S':
        dx, dy = 0, arrow_length
    elif direction == 'E':
        dx, dy = arrow_length, 0
    elif direction == 'W':
        dx, dy = -arrow_length, 0
    
    # Draw arrow
    arrow = FancyArrow(center_x - dx/2, center_y - dy/2, dx, dy,
                      width=0.08, head_width=0.2, head_length=0.15,
                      fc='blue', ec='darkblue', linewidth=1.5)
    ax.add_patch(arrow)

def map_policy(policy):
    """
    Maps the learned policy (1D array of action indices) to a 2D grid format for visualization.
    
    Args:
        policy: 1D array of action indices for each state (length should be 96)
    
    Returns:
        2D array (6x16) where each entry corresponds to the action index for that cell
    """
    grid_policy = np.zeros((6, 16), dtype=int)
    for state in range(len(policy)):
        row = state // 16
        col = state % 16
        grid_policy[row, col] = policy[state]
    return grid_policy[::-1, :]  # reverse rows to match top-down grid layout

if __name__ == "__main__":
    pi = np.random.randint(0, 4, size=96)  # Example random policy for testing
    draw_policy_on_game_grid(pi)
