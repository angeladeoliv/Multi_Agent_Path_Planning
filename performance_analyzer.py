"""
-------------------------------------------------------
Performance Measurement and Comparison Framework
-------------------------------------------------------
Author: Kiro AI Assistant
__updated__ = "2025-07-20"
-------------------------------------------------------
This module provides comprehensive performance analysis and comparison
capabilities for pathfinding algorithms including Gemini API integration.
-------------------------------------------------------
"""

import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any, Callable
from datetime import datetime
import statistics


@dataclass
class PathResult:
    """Data structure to store path generation results and metrics."""
    path: List[Tuple[int, int]]
    execution_time: float
    algorithm_name: str
    success: bool
    error_message: Optional[str] = None
    path_length: int = 0
    is_optimal: bool = False
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.path and self.success:
            self.path_length = len(self.path) - 1 if len(self.path) > 1 else 0


@dataclass
class ComparisonResult:
    """Data structure to store comprehensive comparison results."""
    scenario_id: str
    manual_results: Dict[str, PathResult]
    gemini_result: PathResult
    grid_size: Tuple[int, int]
    start_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
    timestamp: str
    optimal_path_length: Optional[int] = None
    
    def __post_init__(self):
        """Calculate optimal path length and mark optimal results."""
        if self.optimal_path_length is None:
            # Find the shortest successful path length
            successful_lengths = []
            
            for result in self.manual_results.values():
                if result.success and result.path_length > 0:
                    successful_lengths.append(result.path_length)
                    
            if self.gemini_result.success and self.gemini_result.path_length > 0:
                successful_lengths.append(self.gemini_result.path_length)
                
            if successful_lengths:
                self.optimal_path_length = min(successful_lengths)
                
                # Mark optimal results
                for result in self.manual_results.values():
                    if result.success and result.path_length == self.optimal_path_length:
                        result.is_optimal = True
                        
                if self.gemini_result.success and self.gemini_result.path_length == self.optimal_path_length:
                    self.gemini_result.is_optimal = True


@dataclass
class PerformanceMetrics:
    """Aggregate performance metrics for analysis."""
    avg_execution_time: float
    success_rate: float
    avg_path_length: float
    optimality_rate: float
    total_comparisons: int
    min_execution_time: float = 0.0
    max_execution_time: float = 0.0
    std_execution_time: float = 0.0


class PerformanceAnalyzer:
    """
    Comprehensive performance measurement and comparison system for pathfinding algorithms.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the PerformanceAnalyzer.
        
        Args:
            logger: Optional logger instance. If not provided, creates a new one.
        """
        self.logger = logger or self._setup_logger()
        self.results_history: List[ComparisonResult] = []
        
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for performance analysis operations."""
        logger = logging.getLogger(f"{__name__}.PerformanceAnalyzer")
        if not logger.handlers:
            # File handler for detailed logs
            file_handler = logging.FileHandler('performance_analysis.log')
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)
            
            # Console handler for important messages
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.INFO)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.DEBUG)
            
        return logger
        
    def measure_algorithm_performance(self, algorithm_func: Callable, *args, **kwargs) -> PathResult:
        """
        Measure the performance of a single algorithm execution.
        
        Args:
            algorithm_func: The pathfinding algorithm function to measure
            *args: Arguments to pass to the algorithm function
            **kwargs: Keyword arguments to pass to the algorithm function
            
        Returns:
            PathResult containing timing and path data
        """
        algorithm_name = kwargs.pop('algorithm_name', algorithm_func.__name__)
        
        self.logger.debug(f"Measuring performance for algorithm: {algorithm_name}")
        
        start_time = time.perf_counter()
        path = None
        success = False
        error_message = None
        
        try:
            path = algorithm_func(*args, **kwargs)
            success = path is not None and len(path) > 0
            
            if success:
                self.logger.debug(f"{algorithm_name} succeeded with path length: {len(path)}")
            else:
                self.logger.warning(f"{algorithm_name} returned empty or None path")
                error_message = "Algorithm returned empty or None path"
                
        except Exception as e:
            error_message = str(e)
            self.logger.error(f"{algorithm_name} failed with error: {error_message}")
            
        finally:
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
        result = PathResult(
            path=path or [],
            execution_time=execution_time,
            algorithm_name=algorithm_name,
            success=success,
            error_message=error_message
        )
        
        self.logger.info(f"{algorithm_name}: {execution_time:.2f}ms, Success: {success}")
        return result
        
    def compare_paths(self, manual_results: Dict[str, PathResult], 
                     gemini_result: PathResult) -> Dict[str, Any]:
        """
        Analyze and compare paths from different algorithms.
        
        Args:
            manual_results: Dictionary of algorithm name to PathResult for manual algorithms
            gemini_result: PathResult from Gemini API
            
        Returns:
            Dictionary containing detailed comparison analysis
        """
        self.logger.debug("Starting path comparison analysis")
        
        comparison = {
            'execution_times': {},
            'path_lengths': {},
            'success_rates': {},
            'optimality_analysis': {},
            'performance_ranking': [],
            'summary': {}
        }
        
        # Collect all results for analysis
        all_results = dict(manual_results)
        all_results['Gemini API'] = gemini_result
        
        # Analyze execution times
        successful_times = []
        for name, result in all_results.items():
            comparison['execution_times'][name] = result.execution_time
            if result.success:
                successful_times.append((name, result.execution_time))
                
        # Analyze path lengths
        successful_paths = []
        for name, result in all_results.items():
            if result.success and result.path_length > 0:
                comparison['path_lengths'][name] = result.path_length
                successful_paths.append((name, result.path_length))
            else:
                comparison['path_lengths'][name] = None
                
        # Calculate success rates (for this single comparison)
        total_algorithms = len(all_results)
        successful_algorithms = sum(1 for result in all_results.values() if result.success)
        
        for name, result in all_results.items():
            comparison['success_rates'][name] = 1.0 if result.success else 0.0
            
        # Optimality analysis
        if successful_paths:
            optimal_length = min(length for _, length in successful_paths)
            
            for name, result in all_results.items():
                if result.success and result.path_length > 0:
                    is_optimal = result.path_length == optimal_length
                    comparison['optimality_analysis'][name] = {
                        'is_optimal': is_optimal,
                        'optimality_ratio': optimal_length / result.path_length,
                        'extra_steps': result.path_length - optimal_length
                    }
                else:
                    comparison['optimality_analysis'][name] = {
                        'is_optimal': False,
                        'optimality_ratio': 0.0,
                        'extra_steps': float('inf')
                    }
                    
        # Performance ranking (successful algorithms only)
        if successful_times:
            # Rank by execution time (faster is better)
            time_ranking = sorted(successful_times, key=lambda x: x[1])
            comparison['performance_ranking'] = [
                {
                    'algorithm': name,
                    'execution_time': time_ms,
                    'rank': rank + 1
                }
                for rank, (name, time_ms) in enumerate(time_ranking)
            ]
            
        # Summary statistics
        if successful_times:
            times = [time_ms for _, time_ms in successful_times]
            comparison['summary']['avg_execution_time'] = statistics.mean(times)
            comparison['summary']['min_execution_time'] = min(times)
            comparison['summary']['max_execution_time'] = max(times)
            comparison['summary']['std_execution_time'] = statistics.stdev(times) if len(times) > 1 else 0.0
            
        if successful_paths:
            lengths = [length for _, length in successful_paths]
            comparison['summary']['avg_path_length'] = statistics.mean(lengths)
            comparison['summary']['optimal_path_length'] = min(lengths)
            
        comparison['summary']['total_algorithms'] = total_algorithms
        comparison['summary']['successful_algorithms'] = successful_algorithms
        comparison['summary']['overall_success_rate'] = successful_algorithms / total_algorithms
        
        self.logger.info(f"Comparison complete: {successful_algorithms}/{total_algorithms} algorithms succeeded")
        return comparison
        
    def calculate_path_quality_metrics(self, path: List[Tuple[int, int]], 
                                     start: Tuple[int, int], goal: Tuple[int, int]) -> Dict[str, float]:
        """
        Calculate quality metrics for a given path.
        
        Args:
            path: List of coordinate tuples representing the path
            start: Starting position
            goal: Goal position
            
        Returns:
            Dictionary containing path quality metrics
        """
        if not path or len(path) < 2:
            return {
                'path_length': 0,
                'manhattan_distance': self._manhattan_distance(start, goal),
                'optimality_ratio': 0.0,
                'efficiency': 0.0,
                'straightness': 0.0
            }
            
        path_length = len(path) - 1
        manhattan_dist = self._manhattan_distance(start, goal)
        
        metrics = {
            'path_length': path_length,
            'manhattan_distance': manhattan_dist,
            'optimality_ratio': manhattan_dist / path_length if path_length > 0 else 0.0,
            'efficiency': manhattan_dist / path_length if path_length > 0 else 0.0,
            'straightness': self._calculate_straightness(path)
        }
        
        return metrics
        
    def _manhattan_distance(self, start: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        
    def _calculate_straightness(self, path: List[Tuple[int, int]]) -> float:
        """
        Calculate how straight a path is (0.0 = very winding, 1.0 = perfectly straight).
        
        Args:
            path: List of coordinate tuples representing the path
            
        Returns:
            Straightness ratio between 0.0 and 1.0
        """
        if len(path) < 3:
            return 1.0  # A path with 2 or fewer points is perfectly straight
            
        # Count direction changes
        direction_changes = 0
        prev_direction = None
        
        for i in range(1, len(path)):
            curr_pos = path[i]
            prev_pos = path[i-1]
            
            # Determine direction
            if curr_pos[0] > prev_pos[0]:
                direction = 'down'
            elif curr_pos[0] < prev_pos[0]:
                direction = 'up'
            elif curr_pos[1] > prev_pos[1]:
                direction = 'right'
            else:
                direction = 'left'
                
            if prev_direction is not None and direction != prev_direction:
                direction_changes += 1
                
            prev_direction = direction
            
        # Calculate straightness (fewer direction changes = straighter)
        max_possible_changes = len(path) - 2
        if max_possible_changes == 0:
            return 1.0
            
        straightness = 1.0 - (direction_changes / max_possible_changes)
        return max(0.0, straightness)
        
    def generate_performance_report(self, results: List[ComparisonResult], 
                                  output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report from multiple comparison results.
        
        Args:
            results: List of ComparisonResult objects to analyze
            output_file: Optional file path to save the report
            
        Returns:
            Dictionary containing the complete performance report
        """
        if not results:
            self.logger.warning("No results provided for performance report")
            return {}
            
        self.logger.info(f"Generating performance report from {len(results)} comparison results")
        
        report = {
            'metadata': {
                'total_comparisons': len(results),
                'report_generated': datetime.now().isoformat(),
                'algorithms_analyzed': set()
            },
            'aggregate_metrics': {},
            'algorithm_performance': {},
            'detailed_analysis': {
                'execution_time_analysis': {},
                'path_quality_analysis': {},
                'success_rate_analysis': {},
                'optimality_analysis': {}
            },
            'recommendations': []
        }
        
        # Collect all algorithm names
        all_algorithms = set()
        for result in results:
            all_algorithms.update(result.manual_results.keys())
            all_algorithms.add('Gemini API')
            
        report['metadata']['algorithms_analyzed'] = list(all_algorithms)
        
        # Analyze each algorithm's performance
        for algorithm in all_algorithms:
            algorithm_data = {
                'execution_times': [],
                'path_lengths': [],
                'successes': [],
                'optimality_flags': []
            }
            
            # Collect data for this algorithm across all comparisons
            for comparison in results:
                if algorithm == 'Gemini API':
                    result = comparison.gemini_result
                else:
                    result = comparison.manual_results.get(algorithm)
                    
                if result:
                    algorithm_data['execution_times'].append(result.execution_time)
                    algorithm_data['successes'].append(result.success)
                    
                    if result.success and result.path_length > 0:
                        algorithm_data['path_lengths'].append(result.path_length)
                        algorithm_data['optimality_flags'].append(result.is_optimal)
                        
            # Calculate metrics for this algorithm
            if algorithm_data['execution_times']:
                times = algorithm_data['execution_times']
                successes = algorithm_data['successes']
                
                metrics = PerformanceMetrics(
                    avg_execution_time=statistics.mean(times),
                    success_rate=sum(successes) / len(successes),
                    avg_path_length=statistics.mean(algorithm_data['path_lengths']) if algorithm_data['path_lengths'] else 0.0,
                    optimality_rate=sum(algorithm_data['optimality_flags']) / len(algorithm_data['optimality_flags']) if algorithm_data['optimality_flags'] else 0.0,
                    total_comparisons=len(times),
                    min_execution_time=min(times),
                    max_execution_time=max(times),
                    std_execution_time=statistics.stdev(times) if len(times) > 1 else 0.0
                )
                
                report['algorithm_performance'][algorithm] = asdict(metrics)
                
        # Generate recommendations based on analysis
        if report['algorithm_performance']:
            self._generate_recommendations(report)
            
        # Save report if requested
        if output_file:
            self.save_report_to_file(report, output_file)
            
        self.logger.info("Performance report generation complete")
        return report
        
    def _generate_recommendations(self, report: Dict[str, Any]) -> None:
        """Generate performance-based recommendations."""
        algorithms = report['algorithm_performance']
        recommendations = []
        
        # Find fastest algorithm
        fastest_algo = min(algorithms.items(), key=lambda x: x[1]['avg_execution_time'])
        recommendations.append(f"Fastest algorithm: {fastest_algo[0]} ({fastest_algo[1]['avg_execution_time']:.2f}ms avg)")
        
        # Find most reliable algorithm
        most_reliable = max(algorithms.items(), key=lambda x: x[1]['success_rate'])
        recommendations.append(f"Most reliable algorithm: {most_reliable[0]} ({most_reliable[1]['success_rate']:.1%} success rate)")
        
        # Find most optimal algorithm
        most_optimal = max(algorithms.items(), key=lambda x: x[1]['optimality_rate'])
        recommendations.append(f"Most optimal paths: {most_optimal[0]} ({most_optimal[1]['optimality_rate']:.1%} optimal)")
        
        report['recommendations'] = recommendations
        
    def save_report_to_file(self, report: Dict[str, Any], filename: str) -> None:
        """
        Save performance report to a JSON file.
        
        Args:
            report: Performance report dictionary
            filename: Output file path
        """
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Performance report saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save report to {filename}: {str(e)}")
            
    def log_comparison_data(self, comparison_result: ComparisonResult, 
                           log_file: Optional[str] = None) -> None:
        """
        Log structured comparison data for analysis.
        
        Args:
            comparison_result: ComparisonResult to log
            log_file: Optional specific log file path
        """
        log_data = {
            'timestamp': comparison_result.timestamp,
            'scenario_id': comparison_result.scenario_id,
            'grid_size': comparison_result.grid_size,
            'start_pos': comparison_result.start_pos,
            'goal_pos': comparison_result.goal_pos,
            'results': {}
        }
        
        # Add manual algorithm results
        for name, result in comparison_result.manual_results.items():
            log_data['results'][name] = {
                'success': result.success,
                'execution_time': result.execution_time,
                'path_length': result.path_length,
                'is_optimal': result.is_optimal,
                'error': result.error_message
            }
            
        # Add Gemini result
        log_data['results']['Gemini API'] = {
            'success': comparison_result.gemini_result.success,
            'execution_time': comparison_result.gemini_result.execution_time,
            'path_length': comparison_result.gemini_result.path_length,
            'is_optimal': comparison_result.gemini_result.is_optimal,
            'error': comparison_result.gemini_result.error_message
        }
        
        # Log the structured data
        if log_file:
            try:
                with open(log_file, 'a') as f:
                    json.dump(log_data, f)
                    f.write('\n')
            except Exception as e:
                self.logger.error(f"Failed to write to log file {log_file}: {str(e)}")
        else:
            self.logger.info(f"Comparison logged: {json.dumps(log_data, default=str)}")
            
        # Store in history
        self.results_history.append(comparison_result)


# Example usage and testing
if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Create analyzer
    analyzer = PerformanceAnalyzer()
    
    # Mock algorithm functions for testing
    def mock_a_star(grid, start, goal):
        time.sleep(0.01)  # Simulate computation time
        return [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]
        
    def mock_greedy_bfs(grid, start, goal):
        time.sleep(0.005)  # Faster but less optimal
        return [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
        
    def mock_gemini_api(grid, start, goal):
        time.sleep(0.1)  # Simulate API call time
        return [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
        
    print("Testing PerformanceAnalyzer...")
    print("=" * 50)
    
    # Test individual algorithm measurement
    result = analyzer.measure_algorithm_performance(
        mock_a_star, 
        None, (0, 0), (2, 2), 
        algorithm_name="A* Search"
    )
    print(f"A* Result: {result}")
    
    # Test comparison
    manual_results = {
        'A* Search': analyzer.measure_algorithm_performance(mock_a_star, None, (0, 0), (2, 2), algorithm_name="A* Search"),
        'Greedy BFS': analyzer.measure_algorithm_performance(mock_greedy_bfs, None, (0, 0), (2, 2), algorithm_name="Greedy BFS")
    }
    
    gemini_result = analyzer.measure_algorithm_performance(mock_gemini_api, None, (0, 0), (2, 2), algorithm_name="Gemini API")
    
    comparison = analyzer.compare_paths(manual_results, gemini_result)
    print(f"\nComparison Results:")
    print(json.dumps(comparison, indent=2, default=str))