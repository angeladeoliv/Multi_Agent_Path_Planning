"""
-------------------------------------------------------
Test Harness for Search Algorithms with Gemini API Integration
-------------------------------------------------------
Author:       Ahmed Nafees
ID:           169053598
Email:        nafe3598@mylaurier.ca
__updated__ = "2025-07-20"
-------------------------------------------------------
This script loads a robot environment, runs various search 
algorithms including Gemini API, and visualizes the results.
Supports comparison mode for performance analysis.
-------------------------------------------------------
"""
import argparse
import sys
from interpret_environment import read_robot_file
from search_algorithms import a_star_search, greedy_bfs_search, weighted_a_star_search, manhattan_distance, euclidean_distance
from grid_renderer import visualize_path, print_grid_initial
from gemini_comparison import SimpleGeminiClient

def run_and_display(algo_name, search_func, grid, start, goal, heuristic, **kwargs):
    """Helper to run a search and display its results."""
    print(f"\n===== Running {algo_name} with {heuristic.__name__} =====")
    
    path = search_func(grid, start, goal, heuristic_func=heuristic, **kwargs)

    if path:
        print(f"Path Found! Length: {len(path)}, Cost: {len(path) - 1}")
        visualize_path(grid, path, start, goal)
    else:
        print("Search could not find a path.")

def run_gemini_comparison(grid, start, goal):
    """Run Gemini API and display results."""
    print(f"\n===== Running Gemini API =====")
    
    try:
        client = SimpleGeminiClient()
        path = client.generate_path(grid, start, goal)
        
        if path:
            print(f"Path Found! Length: {len(path)}, Cost: {len(path) - 1}")
            visualize_path(grid, path, start, goal)
        else:
            print("Gemini API could not find a path.")
    except Exception as e:
        print(f"Error running Gemini API: {str(e)}")

def run_simple_comparison(filepath):
    """Run simple comparison using the basic comparison tool."""
    print("--- Starting Simple Comparison Mode ---")
    print("Use 'python gemini_comparison.py' for detailed comparison data.")
    print("This mode is for basic algorithm testing only.")
    print("-" * 50)
    
    # Just run the basic algorithms for comparison
    try:
        grid_size, robot_starts, goal_pos, grid = read_robot_file(filepath)
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found. Run 'create_environment.py' first.")
        return

    if not robot_starts:
        print("Error: No robots found in the environment file.")
        return
    
    start_pos = robot_starts[0]
    print(f"Testing algorithms from {start_pos} to {goal_pos}")
    
    # Run basic algorithms
    heuristic = manhattan_distance
    algorithms = [
        ("A* Search", a_star_search),
        ("Greedy BFS", greedy_bfs_search),
        ("Weighted A*", lambda g, s, e, h: weighted_a_star_search(g, s, e, h, weight=1.5))
    ]
    
    for name, algo in algorithms:
        try:
            path = algo(grid, start_pos, goal_pos, heuristic)
            if path:
                print(f"{name}: SUCCESS - Length {len(path)-1}")
            else:
                print(f"{name}: FAILED")
        except Exception as e:
            print(f"{name}: ERROR - {e}")
    
    # Test Gemini API
    try:
        client = SimpleGeminiClient()
        path = client.generate_path(grid, start_pos, goal_pos)
        if path:
            print(f"Gemini API: SUCCESS - Length {len(path)-1}")
        else:
            print(f"Gemini API: FAILED")
    except Exception as e:
        print(f"Gemini API: ERROR - {e}")
    
    print("\nFor detailed comparison data, run:")
    print("python gemini_comparison.py")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run pathfinding algorithms with optional Gemini API comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_search.py                           # Run traditional algorithms only
  python run_search.py --include-gemini          # Include Gemini API in traditional mode
  python run_search.py --compare                 # Run comparison mode with all robots
  python run_search.py --compare --single-robot  # Run comparison with first robot only
  python run_search.py --compare --algorithms "A* Manhattan,Greedy BFS Manhattan"
        """
    )
    
    parser.add_argument(
        '--compare', 
        action='store_true',
        help='Run in comparison mode to analyze performance differences'
    )
    
    parser.add_argument(
        '--include-gemini',
        action='store_true', 
        help='Include Gemini API in traditional algorithm display mode'
    )
    
    parser.add_argument(
        '--single-robot',
        action='store_true',
        help='Use only the first robot for comparison (faster testing)'
    )
    
    parser.add_argument(
        '--algorithms',
        type=str,
        help='Comma-separated list of algorithms to include in comparison'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        default='robot_room.txt',
        help='Path to the robot environment file (default: robot_room.txt)'
    )
    
    return parser.parse_args()

def main():
    """
    Main function to run the search algorithm test harness.
    """
    args = parse_arguments()
    
    # Handle comparison mode
    if args.compare:
        run_simple_comparison(args.file)
        return
    
    # Traditional mode with optional Gemini integration
    try:
        grid_size, robot_starts, goal_pos, grid = read_robot_file(args.file)
    except FileNotFoundError:
        print(f"Error: '{args.file}' not found. Run 'create_environment.py' first.")
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

    # Include Gemini API if requested
    if args.include_gemini:
        run_gemini_comparison(grid, start_pos, goal_pos)


if __name__ == "__main__":
    main()