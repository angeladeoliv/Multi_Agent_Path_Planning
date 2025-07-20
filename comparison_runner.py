"""
-------------------------------------------------------
Orchestration System for Running Pathfinding Comparisons
-------------------------------------------------------
Author: Kiro AI Assistant
__updated__ = "2025-07-20"
-------------------------------------------------------
This module provides orchestration capabilities for running comprehensive
comparisons between Gemini API and manual pathfinding algorithms.
-------------------------------------------------------
"""

import logging
import json
import time
import traceback
import os
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

from gemini_api_client import GeminiAPIClient
from performance_analyzer import PerformanceAnalyzer, PathResult, ComparisonResult
from path_validator import PathValidator
from search_algorithms import a_star_search, greedy_bfs_search, weighted_a_star_search, manhattan_distance, euclidean_distance


class ErrorType(Enum):
    """Enumeration of error types for categorization and recovery strategies."""
    API_CONNECTION_ERROR = "api_connection_error"
    API_RATE_LIMIT_ERROR = "api_rate_limit_error"
    API_TIMEOUT_ERROR = "api_timeout_error"
    API_AUTHENTICATION_ERROR = "api_authentication_error"
    ALGORITHM_EXECUTION_ERROR = "algorithm_execution_error"
    PATH_VALIDATION_ERROR = "path_validation_error"
    GRID_VALIDATION_ERROR = "grid_validation_error"
    CONFIGURATION_ERROR = "configuration_error"
    FILE_IO_ERROR = "file_io_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ErrorRecord:
    """Structured error record for comprehensive error tracking."""
    error_type: ErrorType
    error_message: str
    timestamp: str
    scenario_id: Optional[str] = None
    algorithm_name: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[str] = None
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error record to dictionary for logging."""
        return {
            'error_type': self.error_type.value,
            'error_message': self.error_message,
            'timestamp': self.timestamp,
            'scenario_id': self.scenario_id,
            'algorithm_name': self.algorithm_name,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful,
            'recovery_strategy': self.recovery_strategy,
            'stack_trace': self.stack_trace
        }


@dataclass
class ComparisonConfig:
    """Configuration for comparison runs."""
    include_algorithms: List[str] = None
    max_retries: int = 3
    timeout_seconds: int = 30
    save_results: bool = True
    output_directory: str = "comparison_results"
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.include_algorithms is None:
            self.include_algorithms = ["A* Manhattan", "A* Euclidean", "Greedy BFS Manhattan", "Weighted A*"]


class ComparisonRunner:
    """
    Orchestrates comprehensive comparisons between Gemini API and manual pathfinding algorithms.
    """
    
    def __init__(self, config: Optional[ComparisonConfig] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the ComparisonRunner.
        
        Args:
            config: Configuration for comparison runs
            logger: Optional logger instance
        """
        self.config = config or ComparisonConfig()
        self.logger = logger or self._setup_logger()
        
        # Error tracking
        self.error_records: List[ErrorRecord] = []
        self.error_log_file = "comparison_errors.log"
        
        # Initialize components with error handling
        try:
            self.gemini_client = GeminiAPIClient(max_retries=self.config.max_retries, timeout=self.config.timeout_seconds)
            self.performance_analyzer = PerformanceAnalyzer(logger=self.logger)
            self.path_validator = PathValidator(logger=self.logger)
        except Exception as e:
            self._record_error(
                ErrorType.CONFIGURATION_ERROR,
                f"Failed to initialize components: {str(e)}",
                stack_trace=traceback.format_exc()
            )
            raise
        
        # Algorithm registry
        self.algorithms = {
            "A* Manhattan": lambda grid, start, goal: a_star_search(grid, start, goal, manhattan_distance),
            "A* Euclidean": lambda grid, start, goal: a_star_search(grid, start, goal, euclidean_distance),
            "Greedy BFS Manhattan": lambda grid, start, goal: greedy_bfs_search(grid, start, goal, manhattan_distance),
            "Greedy BFS Euclidean": lambda grid, start, goal: greedy_bfs_search(grid, start, goal, euclidean_distance),
            "Weighted A*": lambda grid, start, goal: weighted_a_star_search(grid, start, goal, manhattan_distance, weight=1.5)
        }
        
        # Validate configuration
        self._validate_configuration()
        
        self.logger.info("ComparisonRunner initialized successfully")
        
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for comparison operations."""
        logger = logging.getLogger(f"{__name__}.ComparisonRunner")
        if not logger.handlers:
            # File handler for detailed logs
            file_handler = logging.FileHandler('comparison_runner.log')
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)
            
            # Console handler for important messages
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(getattr(logging, self.config.log_level))
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.DEBUG)
            
        return logger
        
    def _validate_configuration(self) -> None:
        """Validate the configuration settings."""
        try:
            # Validate algorithms
            unknown_algorithms = [alg for alg in self.config.include_algorithms if alg not in self.algorithms]
            if unknown_algorithms:
                error_msg = f"Unknown algorithms in configuration: {unknown_algorithms}"
                self._record_error(ErrorType.CONFIGURATION_ERROR, error_msg)
                raise ValueError(error_msg)
                
            # Validate timeout and retries
            if self.config.timeout_seconds <= 0:
                error_msg = "Timeout must be positive"
                self._record_error(ErrorType.CONFIGURATION_ERROR, error_msg)
                raise ValueError(error_msg)
                
            if self.config.max_retries < 0:
                error_msg = "Max retries cannot be negative"
                self._record_error(ErrorType.CONFIGURATION_ERROR, error_msg)
                raise ValueError(error_msg)
                
            # Validate output directory
            if self.config.save_results:
                try:
                    os.makedirs(self.config.output_directory, exist_ok=True)
                except Exception as e:
                    error_msg = f"Cannot create output directory: {str(e)}"
                    self._record_error(ErrorType.FILE_IO_ERROR, error_msg)
                    raise
                    
            self.logger.info("Configuration validation successful")
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            raise
            
    def _record_error(self, error_type: ErrorType, error_message: str, 
                     scenario_id: Optional[str] = None, algorithm_name: Optional[str] = None,
                     recovery_attempted: bool = False, recovery_successful: bool = False,
                     recovery_strategy: Optional[str] = None, stack_trace: Optional[str] = None) -> None:
        """
        Record an error with comprehensive details for analysis and recovery.
        
        Args:
            error_type: Type of error that occurred
            error_message: Detailed error message
            scenario_id: Optional scenario identifier
            algorithm_name: Optional algorithm name
            recovery_attempted: Whether recovery was attempted
            recovery_successful: Whether recovery was successful
            recovery_strategy: Description of recovery strategy used
            stack_trace: Optional stack trace
        """
        error_record = ErrorRecord(
            error_type=error_type,
            error_message=error_message,
            timestamp=datetime.now().isoformat(),
            scenario_id=scenario_id,
            algorithm_name=algorithm_name,
            recovery_attempted=recovery_attempted,
            recovery_successful=recovery_successful,
            recovery_strategy=recovery_strategy,
            stack_trace=stack_trace
        )
        
        self.error_records.append(error_record)
        
        # Log to structured error log file
        try:
            with open(self.error_log_file, 'a') as f:
                json.dump(error_record.to_dict(), f)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to write to error log file: {str(e)}")
            
        # Log to main logger
        self.logger.error(f"[{error_type.value}] {error_message}")
        if stack_trace:
            self.logger.debug(f"Stack trace: {stack_trace}")
            
    def _classify_error(self, exception: Exception) -> ErrorType:
        """
        Classify an exception into an appropriate error type.
        
        Args:
            exception: The exception to classify
            
        Returns:
            Appropriate ErrorType for the exception
        """
        error_message = str(exception).lower()
        
        # API-related errors
        if any(keyword in error_message for keyword in ['connection', 'network', 'timeout']):
            if 'timeout' in error_message:
                return ErrorType.API_TIMEOUT_ERROR
            else:
                return ErrorType.API_CONNECTION_ERROR
        elif any(keyword in error_message for keyword in ['rate limit', 'quota', 'throttle']):
            return ErrorType.API_RATE_LIMIT_ERROR
        elif any(keyword in error_message for keyword in ['auth', 'key', 'permission', 'unauthorized']):
            return ErrorType.API_AUTHENTICATION_ERROR
            
        # File I/O errors
        elif any(keyword in error_message for keyword in ['file', 'directory', 'permission denied', 'no such file']):
            return ErrorType.FILE_IO_ERROR
            
        # Grid/path validation errors
        elif any(keyword in error_message for keyword in ['bounds', 'obstacle', 'invalid path', 'grid']):
            if 'path' in error_message:
                return ErrorType.PATH_VALIDATION_ERROR
            else:
                return ErrorType.GRID_VALIDATION_ERROR
                
        # Algorithm execution errors
        elif any(keyword in error_message for keyword in ['algorithm', 'search', 'pathfinding']):
            return ErrorType.ALGORITHM_EXECUTION_ERROR
            
        else:
            return ErrorType.UNKNOWN_ERROR
            
    def _attempt_recovery(self, error_type: ErrorType, exception: Exception, 
                         context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Attempt to recover from an error based on its type.
        
        Args:
            error_type: Type of error that occurred
            exception: The original exception
            context: Context information for recovery
            
        Returns:
            Tuple of (recovery_successful, recovery_strategy_description)
        """
        recovery_strategy = None
        recovery_successful = False
        
        try:
            if error_type == ErrorType.API_RATE_LIMIT_ERROR:
                recovery_strategy = "Wait and retry with exponential backoff"
                wait_time = context.get('retry_attempt', 1) * 2
                self.logger.info(f"Rate limit hit, waiting {wait_time} seconds")
                time.sleep(wait_time)
                recovery_successful = True
                
            elif error_type == ErrorType.API_TIMEOUT_ERROR:
                recovery_strategy = "Increase timeout and retry"
                # Increase timeout for this specific call
                context['timeout'] = context.get('timeout', 30) * 1.5
                recovery_successful = True
                
            elif error_type == ErrorType.API_CONNECTION_ERROR:
                recovery_strategy = "Wait and retry connection"
                wait_time = min(context.get('retry_attempt', 1) * 1.5, 10)
                self.logger.info(f"Connection error, waiting {wait_time} seconds")
                time.sleep(wait_time)
                recovery_successful = True
                
            elif error_type == ErrorType.ALGORITHM_EXECUTION_ERROR:
                recovery_strategy = "Continue with remaining algorithms"
                # Don't fail the entire comparison, just mark this algorithm as failed
                recovery_successful = True
                
            elif error_type == ErrorType.PATH_VALIDATION_ERROR:
                recovery_strategy = "Mark path as invalid and continue"
                recovery_successful = True
                
            elif error_type == ErrorType.GRID_VALIDATION_ERROR:
                recovery_strategy = "Skip invalid scenario and continue"
                recovery_successful = True
                
            else:
                recovery_strategy = "No recovery strategy available"
                recovery_successful = False
                
        except Exception as recovery_exception:
            self.logger.error(f"Recovery attempt failed: {str(recovery_exception)}")
            recovery_successful = False
            
        return recovery_successful, recovery_strategy
        
    def _validate_grid_and_positions(self, grid: List[List[str]], start: Tuple[int, int], 
                                   goal: Tuple[int, int]) -> bool:
        """
        Validate grid structure and position coordinates.
        
        Args:
            grid: 2D grid to validate
            start: Starting position
            goal: Goal position
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check grid structure
            if not grid or not grid[0]:
                self._record_error(ErrorType.GRID_VALIDATION_ERROR, "Empty or invalid grid")
                return False
                
            grid_height = len(grid)
            grid_width = len(grid[0])
            
            # Check grid consistency
            for i, row in enumerate(grid):
                if len(row) != grid_width:
                    self._record_error(
                        ErrorType.GRID_VALIDATION_ERROR, 
                        f"Inconsistent row length at row {i}: expected {grid_width}, got {len(row)}"
                    )
                    return False
                    
            # Check position bounds
            start_row, start_col = start
            goal_row, goal_col = goal
            
            if not (0 <= start_row < grid_height and 0 <= start_col < grid_width):
                self._record_error(
                    ErrorType.GRID_VALIDATION_ERROR,
                    f"Start position {start} is out of bounds for grid {grid_height}x{grid_width}"
                )
                return False
                
            if not (0 <= goal_row < grid_height and 0 <= goal_col < grid_width):
                self._record_error(
                    ErrorType.GRID_VALIDATION_ERROR,
                    f"Goal position {goal} is out of bounds for grid {grid_height}x{grid_width}"
                )
                return False
                
            # Check if start and goal are not obstacles
            if grid[start_row][start_col] == '1':
                self._record_error(
                    ErrorType.GRID_VALIDATION_ERROR,
                    f"Start position {start} is an obstacle"
                )
                return False
                
            if grid[goal_row][goal_col] == '1':
                self._record_error(
                    ErrorType.GRID_VALIDATION_ERROR,
                    f"Goal position {goal} is an obstacle"
                )
                return False
                
            return True
            
        except Exception as e:
            self._record_error(
                ErrorType.GRID_VALIDATION_ERROR,
                f"Error during grid validation: {str(e)}",
                stack_trace=traceback.format_exc()
            )
            return False
        
    def run_single_comparison(self, grid: List[List[str]], start: Tuple[int, int], 
                            goal: Tuple[int, int], scenario_id: Optional[str] = None) -> ComparisonResult:
        """
        Run a comprehensive comparison for a single scenario with robust error handling.
        
        Args:
            grid: 2D grid representation of the environment
            start: Starting position as (row, col)
            goal: Goal position as (row, col)
            scenario_id: Optional identifier for this scenario
            
        Returns:
            ComparisonResult containing all algorithm results and analysis
        """
        if scenario_id is None:
            scenario_id = f"scenario_{int(time.time())}"
            
        self.logger.info(f"Starting single comparison: {scenario_id}")
        self.logger.info(f"Grid size: {len(grid)}x{len(grid[0]) if grid else 0}, Start: {start}, Goal: {goal}")
        
        # Validate input parameters
        if not self._validate_grid_and_positions(grid, start, goal):
            self.logger.error(f"Grid validation failed for scenario {scenario_id}")
            # Return a failed comparison result
            return ComparisonResult(
                scenario_id=scenario_id,
                manual_results={},
                gemini_result=PathResult(
                    path=[],
                    execution_time=0.0,
                    algorithm_name="Gemini API",
                    success=False,
                    error_message="Grid validation failed"
                ),
                grid_size=(len(grid), len(grid[0]) if grid else 0),
                start_pos=start,
                goal_pos=goal,
                timestamp=datetime.now().isoformat()
            )
        
        # Run manual algorithms with comprehensive error handling
        manual_results = {}
        for algorithm_name in self.config.include_algorithms:
            if algorithm_name in self.algorithms:
                retry_count = 0
                max_retries = 2  # Allow 2 retries for manual algorithms
                
                while retry_count <= max_retries:
                    try:
                        self.logger.debug(f"Running {algorithm_name} (attempt {retry_count + 1})")
                        result = self.performance_analyzer.measure_algorithm_performance(
                            self.algorithms[algorithm_name],
                            grid, start, goal,
                            algorithm_name=algorithm_name
                        )
                        manual_results[algorithm_name] = result
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        error_type = self._classify_error(e)
                        self.logger.error(f"Error running {algorithm_name} (attempt {retry_count + 1}): {str(e)}")
                        
                        # Record the error
                        self._record_error(
                            error_type,
                            f"Algorithm {algorithm_name} failed: {str(e)}",
                            scenario_id=scenario_id,
                            algorithm_name=algorithm_name,
                            stack_trace=traceback.format_exc()
                        )
                        
                        # Attempt recovery
                        context = {'retry_attempt': retry_count, 'algorithm_name': algorithm_name}
                        recovery_successful, recovery_strategy = self._attempt_recovery(error_type, e, context)
                        
                        # Update error record with recovery information
                        if self.error_records:
                            self.error_records[-1].recovery_attempted = True
                            self.error_records[-1].recovery_successful = recovery_successful
                            self.error_records[-1].recovery_strategy = recovery_strategy
                        
                        if recovery_successful and retry_count < max_retries:
                            retry_count += 1
                            continue
                        else:
                            # Final failure
                            manual_results[algorithm_name] = PathResult(
                                path=[],
                                execution_time=0.0,
                                algorithm_name=algorithm_name,
                                success=False,
                                error_message=str(e)
                            )
                            break
            else:
                self.logger.warning(f"Unknown algorithm: {algorithm_name}")
                self._record_error(
                    ErrorType.CONFIGURATION_ERROR,
                    f"Unknown algorithm specified: {algorithm_name}",
                    scenario_id=scenario_id,
                    algorithm_name=algorithm_name
                )
                
        # Run Gemini API with comprehensive error handling
        self.logger.debug("Running Gemini API")
        gemini_result = self._run_gemini_with_recovery(grid, start, goal, scenario_id)
            
        # Create comparison result
        comparison_result = ComparisonResult(
            scenario_id=scenario_id,
            manual_results=manual_results,
            gemini_result=gemini_result,
            grid_size=(len(grid), len(grid[0]) if grid else 0),
            start_pos=start,
            goal_pos=goal,
            timestamp=datetime.now().isoformat()
        )
        
        # Log the comparison data
        try:
            self.performance_analyzer.log_comparison_data(comparison_result)
        except Exception as e:
            self._record_error(
                ErrorType.FILE_IO_ERROR,
                f"Failed to log comparison data: {str(e)}",
                scenario_id=scenario_id
            )
        
        self.logger.info(f"Single comparison completed: {scenario_id}")
        return comparison_result
        
    def _run_gemini_with_recovery(self, grid: List[List[str]], start: Tuple[int, int], 
                                 goal: Tuple[int, int], scenario_id: str) -> PathResult:
        """
        Run Gemini API with comprehensive error handling and recovery strategies.
        
        Args:
            grid: 2D grid representation of the environment
            start: Starting position
            goal: Goal position
            scenario_id: Scenario identifier for error tracking
            
        Returns:
            PathResult from Gemini API execution
        """
        retry_count = 0
        max_retries = self.config.max_retries
        
        while retry_count <= max_retries:
            try:
                self.logger.debug(f"Running Gemini API (attempt {retry_count + 1})")
                result = self.performance_analyzer.measure_algorithm_performance(
                    self._gemini_wrapper,
                    grid, start, goal,
                    algorithm_name="Gemini API"
                )
                
                # Additional validation for Gemini results
                if result.success and result.path:
                    if not self.path_validator.is_valid_path(result.path, grid, start, goal):
                        self.logger.warning("Gemini API returned invalid path")
                        self._record_error(
                            ErrorType.PATH_VALIDATION_ERROR,
                            "Gemini API returned invalid path",
                            scenario_id=scenario_id,
                            algorithm_name="Gemini API"
                        )
                        result.success = False
                        result.error_message = "Path validation failed"
                
                return result
                
            except Exception as e:
                error_type = self._classify_error(e)
                self.logger.error(f"Error running Gemini API (attempt {retry_count + 1}): {str(e)}")
                
                # Record the error
                self._record_error(
                    error_type,
                    f"Gemini API failed: {str(e)}",
                    scenario_id=scenario_id,
                    algorithm_name="Gemini API",
                    stack_trace=traceback.format_exc()
                )
                
                # Attempt recovery
                context = {
                    'retry_attempt': retry_count, 
                    'algorithm_name': 'Gemini API',
                    'timeout': self.config.timeout_seconds
                }
                recovery_successful, recovery_strategy = self._attempt_recovery(error_type, e, context)
                
                # Update error record with recovery information
                if self.error_records:
                    self.error_records[-1].recovery_attempted = True
                    self.error_records[-1].recovery_successful = recovery_successful
                    self.error_records[-1].recovery_strategy = recovery_strategy
                
                if recovery_successful and retry_count < max_retries:
                    retry_count += 1
                    continue
                else:
                    # Final failure
                    return PathResult(
                        path=[],
                        execution_time=0.0,
                        algorithm_name="Gemini API",
                        success=False,
                        error_message=str(e)
                    )
        
    def _gemini_wrapper(self, grid: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Wrapper function for Gemini API to match algorithm interface."""
        return self.gemini_client.generate_path(grid, start, goal)
        
    def run_multiple_robots(self, robot_starts: List[Tuple[int, int]], goal: Tuple[int, int], 
                          grid: List[List[str]], batch_id: Optional[str] = None) -> List[ComparisonResult]:
        """
        Run comparisons for multiple robot starting positions with the same goal.
        
        Args:
            robot_starts: List of starting positions for different robots
            goal: Common goal position for all robots
            grid: 2D grid representation of the environment
            batch_id: Optional identifier for this batch of comparisons
            
        Returns:
            List of ComparisonResult objects, one for each robot
        """
        if batch_id is None:
            batch_id = f"batch_{int(time.time())}"
            
        self.logger.info(f"Starting multiple robot comparison: {batch_id}")
        self.logger.info(f"Processing {len(robot_starts)} robots with goal at {goal}")
        
        results = []
        successful_comparisons = 0
        
        for i, start_pos in enumerate(robot_starts):
            scenario_id = f"{batch_id}_robot_{i+1}"
            
            try:
                self.logger.info(f"Processing robot {i+1}/{len(robot_starts)} from {start_pos}")
                
                # Add delay between API calls to respect rate limits
                if i > 0:
                    self.logger.debug("Adding delay between robot comparisons for rate limiting")
                    time.sleep(1.0)  # 1 second delay between robots
                    
                result = self.run_single_comparison(grid, start_pos, goal, scenario_id)
                results.append(result)
                
                if result.gemini_result.success or any(r.success for r in result.manual_results.values()):
                    successful_comparisons += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to process robot {i+1} from {start_pos}: {str(e)}")
                # Create a failed result to maintain consistency
                failed_result = ComparisonResult(
                    scenario_id=scenario_id,
                    manual_results={},
                    gemini_result=PathResult(
                        path=[],
                        execution_time=0.0,
                        algorithm_name="Gemini API",
                        success=False,
                        error_message=str(e)
                    ),
                    grid_size=(len(grid), len(grid[0]) if grid else 0),
                    start_pos=start_pos,
                    goal_pos=goal,
                    timestamp=datetime.now().isoformat()
                )
                results.append(failed_result)
                
        self.logger.info(f"Multiple robot comparison completed: {batch_id}")
        self.logger.info(f"Successfully processed {successful_comparisons}/{len(robot_starts)} robots")
        
        return results
        
    def run_batch_processing(self, scenarios: List[Dict[str, Any]], 
                           batch_id: Optional[str] = None) -> List[ComparisonResult]:
        """
        Process multiple scenarios in batch mode with comprehensive error handling.
        
        Args:
            scenarios: List of scenario dictionaries containing 'grid', 'start', 'goal', and optional 'id'
            batch_id: Optional identifier for this batch
            
        Returns:
            List of ComparisonResult objects for all scenarios
        """
        if batch_id is None:
            batch_id = f"batch_{int(time.time())}"
            
        self.logger.info(f"Starting batch processing: {batch_id}")
        self.logger.info(f"Processing {len(scenarios)} scenarios")
        
        results = []
        successful_scenarios = 0
        
        for i, scenario in enumerate(scenarios):
            scenario_id = scenario.get('id', f"{batch_id}_scenario_{i+1}")
            
            try:
                self.logger.info(f"Processing scenario {i+1}/{len(scenarios)}: {scenario_id}")
                
                # Validate scenario structure
                required_keys = ['grid', 'start', 'goal']
                missing_keys = [key for key in required_keys if key not in scenario]
                if missing_keys:
                    error_msg = f"Scenario {scenario_id} missing required keys: {missing_keys}"
                    self._record_error(
                        ErrorType.CONFIGURATION_ERROR,
                        error_msg,
                        scenario_id=scenario_id
                    )
                    self.logger.error(error_msg)
                    continue
                
                # Add delay between scenarios for rate limiting
                if i > 0:
                    time.sleep(0.5)  # 0.5 second delay between scenarios
                    
                result = self.run_single_comparison(
                    grid=scenario['grid'],
                    start=scenario['start'],
                    goal=scenario['goal'],
                    scenario_id=scenario_id
                )
                results.append(result)
                
                if result.gemini_result.success or any(r.success for r in result.manual_results.values()):
                    successful_scenarios += 1
                    
            except Exception as e:
                error_type = self._classify_error(e)
                self._record_error(
                    error_type,
                    f"Failed to process scenario {scenario_id}: {str(e)}",
                    scenario_id=scenario_id,
                    stack_trace=traceback.format_exc()
                )
                self.logger.error(f"Failed to process scenario {i+1}: {str(e)}")
                continue
                
        self.logger.info(f"Batch processing completed: {batch_id}")
        self.logger.info(f"Successfully processed {successful_scenarios}/{len(scenarios)} scenarios")
        
        return results
        
    def save_results(self, results: List[ComparisonResult], filename: Optional[str] = None) -> str:
        """
        Save comparison results to a JSON file.
        
        Args:
            results: List of ComparisonResult objects to save
            filename: Optional filename. If not provided, generates timestamp-based name
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.output_directory}/comparison_results_{timestamp}.json"
            
        try:
            # Ensure output directory exists
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Convert results to serializable format
            serializable_results = []
            for result in results:
                serializable_result = {
                    'scenario_id': result.scenario_id,
                    'timestamp': result.timestamp,
                    'grid_size': result.grid_size,
                    'start_pos': result.start_pos,
                    'goal_pos': result.goal_pos,
                    'optimal_path_length': result.optimal_path_length,
                    'manual_results': {},
                    'gemini_result': {
                        'path': result.gemini_result.path,
                        'execution_time': result.gemini_result.execution_time,
                        'algorithm_name': result.gemini_result.algorithm_name,
                        'success': result.gemini_result.success,
                        'error_message': result.gemini_result.error_message,
                        'path_length': result.gemini_result.path_length,
                        'is_optimal': result.gemini_result.is_optimal
                    }
                }
                
                for name, manual_result in result.manual_results.items():
                    serializable_result['manual_results'][name] = {
                        'path': manual_result.path,
                        'execution_time': manual_result.execution_time,
                        'algorithm_name': manual_result.algorithm_name,
                        'success': manual_result.success,
                        'error_message': manual_result.error_message,
                        'path_length': manual_result.path_length,
                        'is_optimal': manual_result.is_optimal
                    }
                    
                serializable_results.append(serializable_result)
                
            # Save to file
            with open(filename, 'w') as f:
                json.dump({
                    'metadata': {
                        'total_results': len(results),
                        'generated_at': datetime.now().isoformat(),
                        'config': {
                            'include_algorithms': self.config.include_algorithms,
                            'max_retries': self.config.max_retries,
                            'timeout_seconds': self.config.timeout_seconds
                        }
                    },
                    'results': serializable_results
                }, f, indent=2)
                
            self.logger.info(f"Results saved to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to save results to {filename}: {str(e)}")
            raise
            
    def print_summary(self, results: List[ComparisonResult]) -> None:
        """
        Print a comprehensive summary of comparison results.
        
        Args:
            results: List of ComparisonResult objects to summarize
        """
        if not results:
            print("No results to summarize.")
            return
            
        print("\n" + "=" * 60)
        print("PATHFINDING COMPARISON SUMMARY")
        print("=" * 60)
        
        # Overall statistics
        total_scenarios = len(results)
        successful_scenarios = sum(1 for r in results if r.gemini_result.success or any(mr.success for mr in r.manual_results.values()))
        
        print(f"\nOverall Statistics:")
        print(f"  Total scenarios: {total_scenarios}")
        print(f"  Successful scenarios: {successful_scenarios}")
        print(f"  Success rate: {successful_scenarios/total_scenarios:.1%}")
        
        # Algorithm performance summary
        algorithm_stats = {}
        
        for result in results:
            # Process manual algorithms
            for name, manual_result in result.manual_results.items():
                if name not in algorithm_stats:
                    algorithm_stats[name] = {'successes': 0, 'total': 0, 'times': [], 'path_lengths': []}
                    
                algorithm_stats[name]['total'] += 1
                if manual_result.success:
                    algorithm_stats[name]['successes'] += 1
                    algorithm_stats[name]['times'].append(manual_result.execution_time)
                    if manual_result.path_length > 0:
                        algorithm_stats[name]['path_lengths'].append(manual_result.path_length)
                        
            # Process Gemini API
            if 'Gemini API' not in algorithm_stats:
                algorithm_stats['Gemini API'] = {'successes': 0, 'total': 0, 'times': [], 'path_lengths': []}
                
            algorithm_stats['Gemini API']['total'] += 1
            if result.gemini_result.success:
                algorithm_stats['Gemini API']['successes'] += 1
                algorithm_stats['Gemini API']['times'].append(result.gemini_result.execution_time)
                if result.gemini_result.path_length > 0:
                    algorithm_stats['Gemini API']['path_lengths'].append(result.gemini_result.path_length)
                    
        print(f"\nAlgorithm Performance:")
        print(f"{'Algorithm':<20} {'Success Rate':<12} {'Avg Time (ms)':<15} {'Avg Path Length':<15}")
        print("-" * 65)
        
        for name, stats in algorithm_stats.items():
            success_rate = stats['successes'] / stats['total'] if stats['total'] > 0 else 0
            avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0
            avg_path_length = sum(stats['path_lengths']) / len(stats['path_lengths']) if stats['path_lengths'] else 0
            
            print(f"{name:<20} {success_rate:<12.1%} {avg_time:<15.2f} {avg_path_length:<15.1f}")
            
        # Best performing algorithm
        if algorithm_stats:
            best_success_rate = max(algorithm_stats.items(), key=lambda x: x[1]['successes'] / x[1]['total'] if x[1]['total'] > 0 else 0)
            fastest_algorithm = min(
                [(name, sum(stats['times']) / len(stats['times'])) for name, stats in algorithm_stats.items() if stats['times']], 
                key=lambda x: x[1], 
                default=(None, 0)
            )
            
            print(f"\nKey Findings:")
            print(f"  Most reliable: {best_success_rate[0]} ({best_success_rate[1]['successes']}/{best_success_rate[1]['total']} successes)")
            if fastest_algorithm[0]:
                print(f"  Fastest: {fastest_algorithm[0]} ({fastest_algorithm[1]:.2f}ms avg)")
                
        print("=" * 60)
        
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of all errors encountered.
        
        Returns:
            Dictionary containing error analysis and statistics
        """
        if not self.error_records:
            return {
                'total_errors': 0,
                'error_types': {},
                'recovery_stats': {'attempted': 0, 'successful': 0},
                'most_common_errors': [],
                'recommendations': []
            }
            
        # Count errors by type
        error_type_counts = {}
        recovery_attempted = 0
        recovery_successful = 0
        
        for error in self.error_records:
            error_type = error.error_type.value
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
            
            if error.recovery_attempted:
                recovery_attempted += 1
                if error.recovery_successful:
                    recovery_successful += 1
                    
        # Find most common errors
        most_common = sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate recommendations based on error patterns
        recommendations = []
        if error_type_counts.get('api_connection_error', 0) > 0:
            recommendations.append("Consider checking network connectivity and API endpoint availability")
        if error_type_counts.get('api_rate_limit_error', 0) > 0:
            recommendations.append("Consider increasing delays between API calls or reducing batch sizes")
        if error_type_counts.get('api_authentication_error', 0) > 0:
            recommendations.append("Verify API key configuration and permissions")
        if error_type_counts.get('grid_validation_error', 0) > 0:
            recommendations.append("Validate input grids and positions before processing")
            
        return {
            'total_errors': len(self.error_records),
            'error_types': error_type_counts,
            'recovery_stats': {
                'attempted': recovery_attempted,
                'successful': recovery_successful,
                'success_rate': recovery_successful / recovery_attempted if recovery_attempted > 0 else 0
            },
            'most_common_errors': most_common,
            'recommendations': recommendations,
            'error_timeline': [
                {
                    'timestamp': error.timestamp,
                    'type': error.error_type.value,
                    'message': error.error_message,
                    'scenario_id': error.scenario_id,
                    'algorithm_name': error.algorithm_name
                }
                for error in self.error_records[-10:]  # Last 10 errors
            ]
        }
        
    def save_error_report(self, filename: Optional[str] = None) -> str:
        """
        Save a comprehensive error report to file.
        
        Args:
            filename: Optional filename. If not provided, generates timestamp-based name
            
        Returns:
            Path to the saved error report file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.output_directory}/error_report_{timestamp}.json"
            
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            error_summary = self.get_error_summary()
            
            # Add detailed error records
            error_summary['detailed_errors'] = [error.to_dict() for error in self.error_records]
            
            with open(filename, 'w') as f:
                json.dump(error_summary, f, indent=2)
                
            self.logger.info(f"Error report saved to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to save error report to {filename}: {str(e)}")
            raise
            
    def print_error_summary(self) -> None:
        """Print a human-readable error summary to console."""
        summary = self.get_error_summary()
        
        print("\n" + "=" * 60)
        print("ERROR SUMMARY")
        print("=" * 60)
        
        if summary['total_errors'] == 0:
            print("No errors encountered during execution.")
            return
            
        print(f"\nTotal Errors: {summary['total_errors']}")
        
        print(f"\nError Types:")
        for error_type, count in summary['error_types'].items():
            print(f"  {error_type.replace('_', ' ').title()}: {count}")
            
        recovery_stats = summary['recovery_stats']
        print(f"\nRecovery Statistics:")
        print(f"  Recovery Attempted: {recovery_stats['attempted']}")
        print(f"  Recovery Successful: {recovery_stats['successful']}")
        print(f"  Recovery Success Rate: {recovery_stats['success_rate']:.1%}")
        
        if summary['most_common_errors']:
            print(f"\nMost Common Errors:")
            for error_type, count in summary['most_common_errors']:
                print(f"  {error_type.replace('_', ' ').title()}: {count} occurrences")
                
        if summary['recommendations']:
            print(f"\nRecommendations:")
            for i, recommendation in enumerate(summary['recommendations'], 1):
                print(f"  {i}. {recommendation}")
                
        print("=" * 60)


# Example usage and testing
if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Create test configuration
    config = ComparisonConfig(
        include_algorithms=["A* Manhattan", "Greedy BFS Manhattan"],
        max_retries=2,
        save_results=True
    )
    
    # Create comparison runner
    runner = ComparisonRunner(config)
    
    # Test grid
    test_grid = [
        ['R', '0', '0', '0'],
        ['0', '1', '0', '0'],
        ['0', '1', '0', '0'],
        ['0', '0', '0', 'G']
    ]
    
    print("Testing ComparisonRunner...")
    print("=" * 50)
    
    # Test single comparison
    print("\n1. Testing single comparison:")
    result = runner.run_single_comparison(test_grid, (0, 0), (3, 3), "test_scenario")
    print(f"Single comparison completed: {result.scenario_id}")
    
    # Test multiple robots
    print("\n2. Testing multiple robots:")
    robot_starts = [(0, 0), (0, 3), (3, 0)]
    results = runner.run_multiple_robots(robot_starts, (3, 3), test_grid, "test_batch")
    print(f"Multiple robot comparison completed: {len(results)} results")
    
    # Print summary
    print("\n3. Summary:")
    runner.print_summary(results)
    
    # Save results
    print("\n4. Saving results:")
    filename = runner.save_results(results)
    print(f"Results saved to: {filename}")