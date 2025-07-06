"""
-------------------------------------------------------
Grid Rendering and Visualization Utilities
-------------------------------------------------------
Author:       Ahmed Nafees
ID:           169053598
Email:        nafe3598@mylaurier.ca
__updated__ = "2025-07-06"
-------------------------------------------------------
This file contains functions for visualizing the grid,
paths, and other elements for the pathfinding project.
-------------------------------------------------------
"""

def visualize_path(grid, path, start_pos, goal_pos):
    """
    Overlays the found path onto the grid for visualization.
    
    :param grid: The original 2D grid.
    :param path: A list of (row, col) tuples representing the path.
    :param start_pos: The robot's starting position.
    :param goal_pos: The goal position.
    """
    if not path:
        print("No path to visualize.")
        return

    # Create a mutable copy of the grid for visualization
    vis_grid = [list(row) for row in grid]
    
    # Mark the path with '•'
    for step in path:
        if step != start_pos and step != goal_pos:
            vis_grid[step[0]][step[1]] = '•'
    
    # Ensure start and goal are marked correctly
    vis_grid[start_pos[0]][start_pos[1]] = 'R'
    vis_grid[goal_pos[0]][goal_pos[1]] = 'G'
        
    print("\n--- Path Visualization ---")
    print("Legend: R=Robot, G=Goal, •=Path, 1=Obstacle, 0=Free")
    for row in vis_grid:
        # Use more padding for better alignment
        print(' '.join(f'{cell:^2}' for cell in row))
    print("------------------------")

def print_grid_initial(grid):
    """Prints the initial grid setup."""
    print("--- Initial Grid ---")
    for row in grid:
        print(' '.join(f'{cell:^2}' for cell in row))
    print("--------------------")
