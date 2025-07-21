"""
-------------------------------------------------------
Gemini API Comparison
-------------------------------------------------------
Author: Bilal Rashid, Rahnuma Haque
ID: 210648510, 169024593
Email: rash8510@mylaurier.ca, haqu4593@mylaurier.ca
__updated__ = "2025-07-20"
-------------------------------------------------------
"""

import os
import time
import json
from typing import List, Tuple, Optional, Dict
from dotenv import load_dotenv
import google.generativeai as genai

from search_algorithms import a_star_search, greedy_bfs_search, weighted_a_star_search, manhattan_distance
from interpret_environment import read_robot_file

load_dotenv()

class GeminiClient:
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def generate_path(self, grid: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        try:
            grid_str = "\n".join([" ".join(row) for row in grid])
            

            prompt = f"""
You are navigating a robot on a 2D grid from a START position to a GOAL position.

- The grid contains '0' (free space) and '1' (obstacles).
- You can only move UP, DOWN, LEFT, or RIGHT (no diagonals).
- You CANNOT pass through or land on obstacles.
- Stay within the bounds of the grid.
- Return a valid path from {start} to {goal}, as a list of coordinate tuples.
- Return ONLY the path as Python code, e.g., [(x1,y1), (x2,y2), ..., (xn,yn)].

Grid:
{grid_str}
"""

            response = self.model.generate_content(prompt)
            
            if response and response.text:
                return self._parse_path(response.text)
            return None
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None
    
    def _parse_path(self, response_text: str) -> Optional[List[Tuple[int, int]]]:
        try:
            # look for coordinate pairs
            import re
            coords = re.findall(r'\((\d+),\s*(\d+)\)', response_text)
            if coords:
                return [(int(r), int(c)) for r, c in coords]
            return None
        except:
            return None


class Comparing:
    
    def __init__(self):
        self.gemini_client = GeminiClient()
        self.algorithms = {
            "A* Manhattan": lambda grid, start, goal: a_star_search(grid, start, goal, manhattan_distance),
            "Greedy BFS": lambda grid, start, goal: greedy_bfs_search(grid, start, goal, manhattan_distance),
            "Weighted A*": lambda grid, start, goal: weighted_a_star_search(grid, start, goal, manhattan_distance, weight=1.5)
        }
    
    def run_single_comparison(self, grid: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int]) -> Dict:
        results = {
            'scenario': f"Start: {start}, Goal: {goal}",
            'grid_size': f"{len(grid)}x{len(grid[0])}",
            'manual_algorithms': {},
            'gemini_api': {},
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        for name, algorithm in self.algorithms.items():
            start_time = time.perf_counter()
            try:
                path = algorithm(grid, start, goal)
                end_time = time.perf_counter()
                
                execution_time = (end_time - start_time) * 1000  # ms conversion
                
                results['manual_algorithms'][name] = {
                    'success': path is not None,
                    'execution_time_ms': round(execution_time, 3),
                    'path_length': len(path) - 1 if path else 0,
                    'path': path
                }
            except Exception as e:
                results['manual_algorithms'][name] = {
                    'success': False,
                    'error': str(e),
                    'execution_time_ms': 0,
                    'path_length': 0
                }
        
        # test Gemini API
        start_time = time.perf_counter()
        try:
            path = self.gemini_client.generate_path(grid, start, goal)
            end_time = time.perf_counter()
            
            execution_time = (end_time - start_time) * 1000  # ms conversion
            
            results['gemini_api'] = {
                'success': path is not None,
                'execution_time_ms': round(execution_time, 3),
                'path_length': len(path) - 1 if path else 0,
                'path': path
            }
        except Exception as e:
            results['gemini_api'] = {
                'success': False,
                'error': str(e),
                'execution_time_ms': 0,
                'path_length': 0
            }
        
        return results
    
    def validate_path(self, path: List[Tuple[int, int]], grid: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        if not path or len(path) < 2:
            return False
        
        # check start and end
        if path[0] != start or path[-1] != goal:
            return False
        
        # check bounds and obstacles
        for row, col in path:
            if not (0 <= row < len(grid) and 0 <= col < len(grid[0])):
                return False
            if grid[row][col] == '1':
                return False
        
        # check adjacency
        for i in range(1, len(path)):
            prev_row, prev_col = path[i-1]
            curr_row, curr_col = path[i]
            
            row_diff = abs(curr_row - prev_row)
            col_diff = abs(curr_col - prev_col)
            
            if not ((row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)):
                return False
        
        return True
    
    def save_results(self, results: Dict, filename: str = None):
        # save comparison results as .JSON
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filename}")
    
    def print_summary(self, results: Dict):
        print("\n" + "="*60)
        print("PATHFINDING ALGORITHM COMPARISON SUMMARY")
        print("="*60)
        print(f"Scenario: {results['scenario']}")
        print(f"Grid Size: {results['grid_size']}")
        print(f"Timestamp: {results['timestamp']}")
        
        print("\nManual Algorithms:")
        print("-" * 30)
        for name, result in results['manual_algorithms'].items():
            status = "SUCCESS" if result['success'] else "FAILED"
            time_ms = result.get('execution_time_ms', 0)
            length = result.get('path_length', 0)
            print(f"  {name:15}: {status:7} | {time_ms:6.2f}ms | Length: {length}")
        
        print("\nGemini API:")
        print("-" * 30)
        gemini = results['gemini_api']
        status = "SUCCESS" if gemini['success'] else "FAILED"
        time_ms = gemini.get('execution_time_ms', 0)
        length = gemini.get('path_length', 0)
        print(f"  {'Gemini API':15}: {status:7} | {time_ms:6.2f}ms | Length: {length}")
        
        successful_lengths = []
        for result in results['manual_algorithms'].values():
            if result['success'] and result['path_length'] > 0:
                successful_lengths.append(result['path_length'])
        if gemini['success'] and gemini['path_length'] > 0:
            successful_lengths.append(gemini['path_length'])
        
        if successful_lengths:
            optimal_length = min(successful_lengths)
            print(f"\nOptimal Path Length: {optimal_length}")
            
            print("Optimal Solutions:")
            for name, result in results['manual_algorithms'].items():
                if result['success'] and result['path_length'] == optimal_length:
                    print(f"  ✓ {name}")
            if gemini['success'] and gemini['path_length'] == optimal_length:
                print(f"  ✓ Gemini API")


def main():
    print("Gemini API Pathfinding Comparison")
    print("="*50)
    
    try:
        grid_size, robot_starts, goal_pos, grid = read_robot_file("robot_room.txt")
        print(f"Environment loaded: {grid_size[0]}x{grid_size[1]} grid")
        print(f"Robots: {len(robot_starts)}, Goal: {goal_pos}")
    except FileNotFoundError:
        print("Error: robot_room.txt not found. Run 'python create_environment.py' first.")
        return
    except Exception as e:
        print(f"Error loading environment: {e}")
        return
    
    if not robot_starts:
        print("Error: No robots found in environment.")
        return
    
    comparison = Comparing()
    
    start_pos = robot_starts[0]
    print(f"\nRunning comparison from {start_pos} to {goal_pos}")
    
    results = comparison.run_single_comparison(grid, start_pos, goal_pos)
    
    comparison.print_summary(results)
    
    comparison.save_results(results)
    
    print(f"\n{'='*60}")
    print("Comparison complete! Use this data for your PDF analysis.")
    print("Key metrics collected:")
    print("- Execution times for performance comparison")
    print("- Path lengths for optimality analysis") 
    print("- Success rates for reliability assessment")
    print("- Raw path data for detailed analysis")


if __name__ == "__main__":
    main()
