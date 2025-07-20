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
from comparison_runner import ComparisonRunner, ComparisonConfig
from gemini_api_client import GeminiAPIClient

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
        client = GeminiAPIClient()
        path = client.generate_path(grid, start, goal)
        
        if path:
            print(f"Path Found! Length: {len(path)}, Cost: {len(path) - 1}")
            visualize_path(grid, path, start, goal)
        else:
            print("Gemini API could not find a path.")
    except Exception as e:
        print(f"Error running Gemini API: {str(e)}")

def run_comparison_mode(filepath, algorithms=None, single_robot=False):
    """Run comprehensive comparison between algorithms and Gemini API."""
    try:
        grid_size, robot_starts, goal_pos, grid = read_robot_file(filepath)
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found. Run 'create_environment.py' first.")
        return

    if not robot_starts:
        print("Error: No robots found in the environment file.")
        return

    print("--- Starting Comparison Mode ---")
    print(f"Grid Size: {grid_size}")
    print(f"Goal: {goal_pos}")
    print(f"Robots: {len(robot_starts)}")
    print("-" * 40)

    # Configure comparison
    config = ComparisonConfig()
    if algorithms:
        config.include_algorithms = algorithms
    
    runner = ComparisonRunner(config)
    
    if single_robot:
        # Compare using only the first robot
        start_pos = robot_starts[0]
        print(f"Running single robot comparison from {start_pos}")
        result = runner.run_single_comparison(grid, start_pos, goal_pos)
        
        # Display results
        print("\n--- Comparison Results ---")
        print(f"Scenario: {result.scenario_id}")
        print(f"Grid Size: {result.grid_size}")
        print(f"Start: {result.start_pos}, Goal: {result.goal_pos}")
        
        print("\nManual Algorithms:")
        for name, path_result in result.manual_results.items():
            status = "SUCCESS" if path_result.success else "FAILED"
            time_ms = path_result.execution_time
            length = path_result.path_length if path_result.success else "N/A"
            optimal = "OPTIMAL" if path_result.is_optimal else ""
            print(f"  {name}: {status} | {time_ms:.2f}ms | Length: {length} {optimal}")
        
        print(f"\nGemini API:")
        gemini = result.gemini_result
        status = "SUCCESS" if gemini.success else "FAILED"
        time_ms = gemini.execution_time
        length = gemini.path_length if gemini.success else "N/A"
        optimal = "OPTIMAL" if gemini.is_optimal else ""
        print(f"  Gemini API: {status} | {time_ms:.2f}ms | Length: {length} {optimal}")
        
    else:
        # Compare using all robots
        print(f"Running multi-robot comparison with {len(robot_starts)} robots")
        results = runner.run_multiple_robots(robot_starts, goal_pos, grid)
        
        # Display summary
        print(f"\n--- Multi-Robot Comparison Summary ---")
        successful_comparisons = sum(1 for r in results if r.gemini_result.success or any(mr.success for mr in r.manual_results.values()))
        print(f"Successful comparisons: {successful_comparisons}/{len(results)}")
        
        # Show per-robot summary
        for i, result in enumerate(results):
            print(f"\nRobot {i+1} (from {result.start_pos}):")
            manual_successes = sum(1 for r in result.manual_results.values() if r.success)
            gemini_success = "SUCCESS" if result.gemini_result.success else "FAILED"
            print(f"  Manual algorithms: {manual_successes}/{len(result.manual_results)} successful")
            print(f"  Gemini API: {gemini_success}")

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
        algorithms = None
        if args.algorithms:
            algorithms = [alg.strip() for alg in args.algorithms.split(',')]
        run_comparison_mode(args.file, algorithms, args.single_robot)
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