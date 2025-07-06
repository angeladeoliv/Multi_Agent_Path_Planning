"""
-------------------------------------------------------
Test Harness for Search Algorithms
-------------------------------------------------------
Author:       Ahmed Nafees
ID:           169053598
Email:        nafe3598@mylaurier.ca
__updated__ = "2025-07-06"
-------------------------------------------------------
This script loads a robot environment, runs various search 
algorithms, and visualizes the results.
-------------------------------------------------------
"""
from interpret_environment import read_robot_file
from search_algorithms import a_star_search, greedy_bfs_search, weighted_a_star_search, manhattan_distance, euclidean_distance
from grid_renderer import visualize_path, print_grid_initial

def run_and_display(algo_name, search_func, grid, start, goal, heuristic, **kwargs):
    """Helper to run a search and display its results."""
    print(f"\n===== Running {algo_name} with {heuristic.__name__} =====")
    
    path = search_func(grid, start, goal, heuristic_func=heuristic, **kwargs)

    if path:
        print(f"Path Found! Length: {len(path)}, Cost: {len(path) - 1}")
        visualize_path(grid, path, start, goal)
    else:
        print("Search could not find a path.")

def main():
    """
    Main function to run the search algorithm test harness.
    """
    filepath = "robot_room.txt"
    try:
        grid_size, robot_starts, goal_pos, grid = read_robot_file(filepath)
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found. Run 'create_environment.py' first.")
        return

    if not robot_starts:
        print("Error: No robots found in the environment file.")
        return
    start_pos = robot_starts[0]

    print("--- Environment Loaded ---")
    print(f"Grid Size: {grid_size}")
    print(f"Robot Start: {start_pos}")
    print(f"Goal: {goal_pos}")
    print_grid_initial(grid)
    print("--------------------------")

    # --- Run All Implemented Algorithms ---
    heuristic_to_use = manhattan_distance # or euclidean_distance

    run_and_display("A* Search", a_star_search, grid, start_pos, goal_pos, heuristic_to_use)
    
    run_and_display("Greedy Best-First Search", greedy_bfs_search, grid, start_pos, goal_pos, heuristic_to_use)

    run_and_display("Weighted A* Search (w=1.5)", weighted_a_star_search, grid, start_pos, goal_pos, heuristic_to_use, weight=1.5)
    
    run_and_display("Weighted A* Search (w=3.0)", weighted_a_star_search, grid, start_pos, goal_pos, heuristic_to_use, weight=3.0)


if __name__ == "__main__":
    main()