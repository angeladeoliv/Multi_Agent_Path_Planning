"""
-------------------------------------------------------
Path Validation and Safety Checking System
-------------------------------------------------------
Author: Kiro AI Assistant
__updated__ = "2025-07-20"
-------------------------------------------------------
This module provides comprehensive path validation and safety
checking for generated paths in grid-based pathfinding.
-------------------------------------------------------
"""

import logging
from typing import List, Tuple, Optional


class PathValidator:
    """
    Validates paths for safety, connectivity, and compliance with grid constraints.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the PathValidator.
        
        Args:
            logger: Optional logger instance. If not provided, creates a new one.
        """
        self.logger = logger or self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for path validation operations."""
        logger = logging.getLogger(f"{__name__}.PathValidator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def is_valid_path(self, path: List[Tuple[int, int]], grid: List[List[str]], 
                     start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """
        Comprehensive path validation combining all safety checks.
        
        Args:
            path: List of coordinate tuples representing the path
            grid: 2D grid representation of the environment
            start: Starting position as (row, col)
            goal: Goal position as (row, col)
            
        Returns:
            True if path is completely valid and safe, False otherwise
        """
        if not path:
            self.logger.warning("Path validation failed: Empty path provided")
            return False
            
        try:
            # Perform all validation checks
            checks = [
                ("bounds", self.check_bounds(path, grid)),
                ("obstacles", self.check_obstacles(path, grid)),
                ("connectivity", self.check_connectivity(path)),
                ("endpoints", self._check_start_and_goal(path, start, goal))
            ]
            
            failed_checks = [name for name, result in checks if not result]
            
            if failed_checks:
                self.logger.warning(f"Path validation failed checks: {', '.join(failed_checks)}")
                return False
                
            self.logger.info(f"Path validation successful - {len(path)} coordinates validated")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during path validation: {str(e)}")
            return False
            
    def check_bounds(self, path: List[Tuple[int, int]], grid: List[List[str]]) -> bool:
        """
        Validate that all coordinates in the path are within grid boundaries.
        
        Args:
            path: List of coordinate tuples to validate
            grid: 2D grid to check bounds against
            
        Returns:
            True if all coordinates are within bounds, False otherwise
        """
        if not grid or not grid[0]:
            self.logger.error("Invalid grid provided for bounds checking")
            return False
            
        grid_height = len(grid)
        grid_width = len(grid[0])
        
        for i, (row, col) in enumerate(path):
            if not (0 <= row < grid_height and 0 <= col < grid_width):
                self.logger.warning(
                    f"Coordinate ({row}, {col}) at path index {i} is out of bounds. "
                    f"Grid size: {grid_height}x{grid_width}"
                )
                return False
                
        self.logger.debug(f"Bounds check passed for {len(path)} coordinates")
        return True
        
    def check_obstacles(self, path: List[Tuple[int, int]], grid: List[List[str]]) -> bool:
        """
        Ensure the path doesn't collide with any obstacles in the grid.
        
        Args:
            path: List of coordinate tuples representing the path
            grid: 2D grid representation of the environment
            
        Returns:
            True if path avoids all obstacles, False if collision detected
        """
        if not grid:
            self.logger.error("Invalid grid provided for obstacle checking")
            return False
            
        obstacle_positions = []
        
        for i, (row, col) in enumerate(path):
            try:
                cell_value = grid[row][col]
                
                # Check for obstacle collision
                if cell_value == '1':
                    obstacle_positions.append((row, col, i))
                    
            except IndexError:
                self.logger.error(f"Grid access error at coordinate ({row}, {col})")
                return False
                
        if obstacle_positions:
            self.logger.warning(
                f"Path collides with obstacles at positions: "
                f"{[(pos[0], pos[1]) for pos in obstacle_positions]} "
                f"(path indices: {[pos[2] for pos in obstacle_positions]})"
            )
            return False
            
        self.logger.debug(f"Obstacle check passed - no collisions detected")
        return True
        
    def check_connectivity(self, path: List[Tuple[int, int]]) -> bool:
        """
        Verify that the path consists of connected adjacent moves (no gaps or diagonals).
        
        Args:
            path: List of coordinate tuples representing the path
            
        Returns:
            True if path is properly connected, False if gaps or invalid moves detected
        """
        if len(path) < 2:
            self.logger.debug("Path connectivity check passed - single coordinate or empty path")
            return True
            
        invalid_moves = []
        
        for i in range(1, len(path)):
            prev_row, prev_col = path[i-1]
            curr_row, curr_col = path[i]
            
            row_diff = abs(curr_row - prev_row)
            col_diff = abs(curr_col - prev_col)
            
            # Valid moves: exactly one step in one direction (no diagonals, no gaps)
            is_valid_move = (
                (row_diff == 1 and col_diff == 0) or  # Vertical move
                (row_diff == 0 and col_diff == 1)     # Horizontal move
            )
            
            if not is_valid_move:
                invalid_moves.append({
                    'from': path[i-1],
                    'to': path[i],
                    'index': i,
                    'row_diff': row_diff,
                    'col_diff': col_diff
                })
                
        if invalid_moves:
            move_descriptions = [f"{move['from']} -> {move['to']}" for move in invalid_moves]
            self.logger.warning(
                f"Path connectivity check failed. Invalid moves detected: {move_descriptions}"
            )
            return False
            
        self.logger.debug(f"Connectivity check passed - {len(path)-1} moves validated")
        return True
        
    def _check_start_and_goal(self, path: List[Tuple[int, int]], 
                             start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """
        Verify that the path starts at the correct start position and ends at the goal.
        
        Args:
            path: List of coordinate tuples representing the path
            start: Expected starting position
            goal: Expected goal position
            
        Returns:
            True if path has correct start and end points, False otherwise
        """
        if not path:
            return False
            
        start_correct = path[0] == start
        goal_correct = path[-1] == goal
        
        if not start_correct:
            self.logger.warning(f"Path start mismatch: expected {start}, got {path[0]}")
            
        if not goal_correct:
            self.logger.warning(f"Path goal mismatch: expected {goal}, got {path[-1]}")
            
        return start_correct and goal_correct
        
    def get_path_safety_report(self, path: List[Tuple[int, int]], grid: List[List[str]], 
                              start: Tuple[int, int], goal: Tuple[int, int]) -> dict:
        """
        Generate a comprehensive safety report for a path.
        
        Args:
            path: List of coordinate tuples representing the path
            grid: 2D grid representation of the environment
            start: Starting position
            goal: Goal position
            
        Returns:
            Dictionary containing detailed validation results and metrics
        """
        report = {
            'path_length': len(path) if path else 0,
            'is_valid': False,
            'checks': {},
            'issues': [],
            'metrics': {}
        }
        
        if not path:
            report['issues'].append("Empty path provided")
            return report
            
        try:
            # Perform individual checks
            report['checks']['bounds'] = self.check_bounds(path, grid)
            report['checks']['obstacles'] = self.check_obstacles(path, grid)
            report['checks']['connectivity'] = self.check_connectivity(path)
            report['checks']['endpoints'] = self._check_start_and_goal(path, start, goal)
            
            # Overall validity
            report['is_valid'] = all(report['checks'].values())
            
            # Collect issues
            if not report['checks']['bounds']:
                report['issues'].append("Path contains out-of-bounds coordinates")
            if not report['checks']['obstacles']:
                report['issues'].append("Path collides with obstacles")
            if not report['checks']['connectivity']:
                report['issues'].append("Path contains invalid moves or gaps")
            if not report['checks']['endpoints']:
                report['issues'].append("Path has incorrect start or goal positions")
                
            # Calculate metrics
            report['metrics']['total_moves'] = len(path) - 1 if len(path) > 1 else 0
            report['metrics']['manhattan_distance'] = self._calculate_manhattan_distance(start, goal)
            
            if report['metrics']['total_moves'] > 0:
                report['metrics']['efficiency'] = (
                    report['metrics']['manhattan_distance'] / report['metrics']['total_moves']
                )
            else:
                report['metrics']['efficiency'] = 0.0
                
        except Exception as e:
            report['issues'].append(f"Error during validation: {str(e)}")
            self.logger.error(f"Error generating safety report: {str(e)}")
            
        return report
        
    def _calculate_manhattan_distance(self, start: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])


# Example usage and testing
if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Create validator
    validator = PathValidator()
    
    # Test grid
    test_grid = [
        ['R', '0', '0', '0'],
        ['0', '1', '0', '0'],
        ['0', '1', '0', '0'],
        ['0', '0', '0', 'G']
    ]
    
    start = (0, 0)
    goal = (3, 3)
    
    # Test cases
    test_cases = [
        {
            'name': 'Valid path',
            'path': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3)]
        },
        {
            'name': 'Path with obstacle collision',
            'path': [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (3, 2), (3, 3)]
        },
        {
            'name': 'Path with diagonal move',
            'path': [(0, 0), (1, 1), (2, 2), (3, 3)]
        },
        {
            'name': 'Path out of bounds',
            'path': [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4), (3, 3)]
        },
        {
            'name': 'Empty path',
            'path': []
        }
    ]
    
    print("Testing PathValidator...")
    print("=" * 50)
    
    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print(f"Path: {test_case['path']}")
        
        is_valid = validator.is_valid_path(test_case['path'], test_grid, start, goal)
        print(f"Valid: {is_valid}")
        
        # Generate detailed report
        report = validator.get_path_safety_report(test_case['path'], test_grid, start, goal)
        print(f"Report: {report}")
        print("-" * 30)