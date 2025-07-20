"""
-------------------------------------------------------
Comprehensive Pathfinding Algorithm Comparison Script
-------------------------------------------------------
Author: Kiro AI Assistant
__updated__ = "2025-07-20"
-------------------------------------------------------
This script provides comprehensive comparison analysis between
Gemini API and traditional pathfinding algorithms with detailed
reporting and pros/cons analysis.
-------------------------------------------------------
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

from interpret_environment import read_robot_file
from comparison_runner import ComparisonRunner, ComparisonConfig
from performance_analyzer import PerformanceAnalyzer


class ComprehensiveAnalyzer:
    """Provides comprehensive analysis and reporting for pathfinding comparisons."""
    
    def __init__(self, output_dir: str = "comparison_results"):
        """
        Initialize the comprehensive analyzer.
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_pros_cons_analysis(self, performance_report: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
        """
        Generate pros and cons analysis for each algorithm based on performance data.
        
        Args:
            performance_report: Performance report from PerformanceAnalyzer
            
        Returns:
            Dictionary with pros/cons for each algorithm
        """
        analysis = {}
        algorithms = performance_report.get('algorithm_performance', {})
        
        if not algorithms:
            return analysis
            
        # Calculate relative performance metrics
        execution_times = {name: data['avg_execution_time'] for name, data in algorithms.items()}
        success_rates = {name: data['success_rate'] for name, data in algorithms.items()}
        path_lengths = {name: data['avg_path_length'] for name, data in algorithms.items() if data['avg_path_length'] > 0}
        optimality_rates = {name: data['optimality_rate'] for name, data in algorithms.items()}
        
        # Find best performers in each category
        fastest_algo = min(execution_times.items(), key=lambda x: x[1])[0] if execution_times else None
        most_reliable = max(success_rates.items(), key=lambda x: x[1])[0] if success_rates else None
        shortest_paths = min(path_lengths.items(), key=lambda x: x[1])[0] if path_lengths else None
        most_optimal = max(optimality_rates.items(), key=lambda x: x[1])[0] if optimality_rates else None
        
        for algo_name, metrics in algorithms.items():
            pros = []
            cons = []
            
            # Analyze execution time
            if algo_name == fastest_algo:
                pros.append("Fastest execution time")
            elif metrics['avg_execution_time'] > 100:  # > 100ms
                cons.append("Relatively slow execution")
            
            # Analyze reliability
            if metrics['success_rate'] >= 0.9:
                pros.append("High success rate (≥90%)")
            elif metrics['success_rate'] < 0.7:
                cons.append("Lower success rate (<70%)")
                
            if algo_name == most_reliable:
                pros.append("Most reliable algorithm")
            
            # Analyze path quality
            if algo_name == shortest_paths:
                pros.append("Generates shortest paths on average")
            elif algo_name in path_lengths and path_lengths[algo_name] > min(path_lengths.values()) * 1.2:
                cons.append("Generates longer paths than optimal")
            
            # Analyze optimality
            if metrics['optimality_rate'] >= 0.8:
                pros.append("High optimality rate (≥80%)")
            elif metrics['optimality_rate'] < 0.5:
                cons.append("Low optimality rate (<50%)")
                
            if algo_name == most_optimal:
                pros.append("Most optimal solutions")
            
            # Algorithm-specific analysis
            if "Gemini" in algo_name:
                if metrics['avg_execution_time'] > 1000:  # > 1 second
                    cons.append("API latency affects performance")
                if metrics['success_rate'] < 1.0:
                    cons.append("Dependent on API availability and response quality")
                pros.append("Potentially novel pathfinding approaches")
                pros.append("No need for algorithm implementation")
                cons.append("Requires internet connection and API key")
                cons.append("Variable response quality")
                
            elif "A*" in algo_name:
                pros.append("Guaranteed optimal solution (when admissible heuristic)")
                pros.append("Well-established algorithm")
                if "Manhattan" in algo_name:
                    pros.append("Efficient for grid-based pathfinding")
                elif "Euclidean" in algo_name:
                    pros.append("More accurate distance estimation")
                    
            elif "Greedy" in algo_name:
                pros.append("Fast execution")
                cons.append("Not guaranteed to find optimal solution")
                cons.append("Can get stuck in local optima")
                
            elif "Weighted" in algo_name:
                pros.append("Faster than standard A* with acceptable optimality trade-off")
                cons.append("May sacrifice optimality for speed")
            
            analysis[algo_name] = {
                'pros': pros,
                'cons': cons
            }
            
        return analysis
        
    def format_summary_report(self, performance_report: Dict[str, Any], 
                            pros_cons: Dict[str, Dict[str, List[str]]]) -> str:
        """
        Format a comprehensive summary report.
        
        Args:
            performance_report: Performance metrics report
            pros_cons: Pros and cons analysis
            
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE PATHFINDING ALGORITHM COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Comparisons: {performance_report.get('metadata', {}).get('total_comparisons', 'N/A')}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)
        
        algorithms = performance_report.get('algorithm_performance', {})
        if algorithms:
            # Find top performers
            fastest = min(algorithms.items(), key=lambda x: x[1]['avg_execution_time'])
            most_reliable = max(algorithms.items(), key=lambda x: x[1]['success_rate'])
            most_optimal = max(algorithms.items(), key=lambda x: x[1]['optimality_rate'])
            
            report_lines.append(f"Fastest Algorithm: {fastest[0]} ({fastest[1]['avg_execution_time']:.2f}ms avg)")
            report_lines.append(f"Most Reliable: {most_reliable[0]} ({most_reliable[1]['success_rate']:.1%} success)")
            report_lines.append(f"Most Optimal: {most_optimal[0]} ({most_optimal[1]['optimality_rate']:.1%} optimal)")
            report_lines.append("")
        
        # Detailed Algorithm Analysis
        report_lines.append("DETAILED ALGORITHM ANALYSIS")
        report_lines.append("-" * 40)
        
        for algo_name in sorted(algorithms.keys()):
            metrics = algorithms[algo_name]
            analysis = pros_cons.get(algo_name, {'pros': [], 'cons': []})
            
            report_lines.append(f"\n{algo_name.upper()}")
            report_lines.append("─" * len(algo_name))
            
            # Performance metrics
            report_lines.append(f"Average Execution Time: {metrics['avg_execution_time']:.2f}ms")
            report_lines.append(f"Success Rate: {metrics['success_rate']:.1%}")
            report_lines.append(f"Average Path Length: {metrics['avg_path_length']:.1f}")
            report_lines.append(f"Optimality Rate: {metrics['optimality_rate']:.1%}")
            report_lines.append(f"Total Comparisons: {metrics['total_comparisons']}")
            
            # Pros and Cons
            if analysis['pros']:
                report_lines.append("\nPROS:")
                for pro in analysis['pros']:
                    report_lines.append(f"  ✓ {pro}")
                    
            if analysis['cons']:
                report_lines.append("\nCONS:")
                for con in analysis['cons']:
                    report_lines.append(f"  ✗ {con}")
            
            report_lines.append("")
        
        # Recommendations
        recommendations = performance_report.get('recommendations', [])
        if recommendations:
            report_lines.append("RECOMMENDATIONS")
            report_lines.append("-" * 40)
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")
        
        # Use Case Recommendations
        report_lines.append("USE CASE RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        if algorithms:
            # Real-time applications
            fastest_algos = sorted(algorithms.items(), key=lambda x: x[1]['avg_execution_time'])[:2]
            report_lines.append("For Real-time Applications:")
            for algo, metrics in fastest_algos:
                if metrics['success_rate'] >= 0.8:
                    report_lines.append(f"  • {algo} - Fast execution ({metrics['avg_execution_time']:.2f}ms)")
            
            # Optimal path requirements
            optimal_algos = sorted(algorithms.items(), key=lambda x: x[1]['optimality_rate'], reverse=True)[:2]
            report_lines.append("\nFor Optimal Path Requirements:")
            for algo, metrics in optimal_algos:
                if metrics['optimality_rate'] >= 0.7:
                    report_lines.append(f"  • {algo} - High optimality ({metrics['optimality_rate']:.1%})")
            
            # Research and experimentation
            report_lines.append("\nFor Research and Experimentation:")
            for algo, metrics in algorithms.items():
                if "Gemini" in algo:
                    report_lines.append(f"  • {algo} - Novel AI-based approach")
                    break
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
        
    def save_detailed_results(self, performance_report: Dict[str, Any], 
                            pros_cons: Dict[str, Dict[str, List[str]]], 
                            raw_results: List[Any]) -> Dict[str, str]:
        """
        Save detailed results to multiple output formats.
        
        Args:
            performance_report: Performance metrics report
            pros_cons: Pros and cons analysis
            raw_results: Raw comparison results
            
        Returns:
            Dictionary mapping output type to file path
        """
        output_files = {}
        
        # Save JSON report
        json_data = {
            'timestamp': self.timestamp,
            'performance_report': performance_report,
            'pros_cons_analysis': pros_cons,
            'summary': {
                'total_comparisons': len(raw_results),
                'algorithms_tested': list(performance_report.get('algorithm_performance', {}).keys())
            }
        }
        
        json_file = os.path.join(self.output_dir, f"comparison_report_{self.timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        output_files['json'] = json_file
        
        # Save formatted text report
        text_report = self.format_summary_report(performance_report, pros_cons)
        text_file = os.path.join(self.output_dir, f"comparison_summary_{self.timestamp}.txt")
        with open(text_file, 'w') as f:
            f.write(text_report)
        output_files['text'] = text_file
        
        # Save raw results
        raw_file = os.path.join(self.output_dir, f"raw_results_{self.timestamp}.json")
        with open(raw_file, 'w') as f:
            json.dump([result.__dict__ if hasattr(result, '__dict__') else result 
                      for result in raw_results], f, indent=2, default=str)
        output_files['raw'] = raw_file
        
        return output_files


def run_comprehensive_analysis(environment_file: str, config: ComparisonConfig, 
                             single_robot: bool = False) -> None:
    """
    Run comprehensive pathfinding algorithm comparison and analysis.
    
    Args:
        environment_file: Path to robot environment file
        config: Comparison configuration
        single_robot: Whether to use only the first robot
    """
    print("Starting Comprehensive Pathfinding Algorithm Analysis")
    print("=" * 60)
    
    # Load environment
    try:
        grid_size, robot_starts, goal_pos, grid = read_robot_file(environment_file)
        print(f"Environment loaded: {grid_size[0]}x{grid_size[1]} grid")
        print(f"Robots: {len(robot_starts)}, Goal: {goal_pos}")
    except FileNotFoundError:
        print(f"Error: Environment file '{environment_file}' not found.")
        return
    except Exception as e:
        print(f"Error loading environment: {str(e)}")
        return
    
    if not robot_starts:
        print("Error: No robots found in environment.")
        return
    
    # Initialize components
    runner = ComparisonRunner(config)
    analyzer = PerformanceAnalyzer()
    comprehensive_analyzer = ComprehensiveAnalyzer()
    
    print(f"Algorithms to test: {', '.join(config.include_algorithms)}")
    print(f"Mode: {'Single robot' if single_robot else 'Multi-robot'}")
    print("-" * 60)
    
    # Run comparisons
    results = []
    try:
        if single_robot:
            print("Running single robot comparison...")
            result = runner.run_single_comparison(grid, robot_starts[0], goal_pos)
            results = [result]
        else:
            print(f"Running multi-robot comparison with {len(robot_starts)} robots...")
            results = runner.run_multiple_robots(robot_starts, goal_pos, grid)
            
        print(f"Completed {len(results)} comparison(s)")
        
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        return
    
    if not results:
        print("No comparison results generated.")
        return
    
    # Generate performance report
    print("\nGenerating performance analysis...")
    performance_report = analyzer.generate_performance_report(results)
    
    # Generate pros/cons analysis
    print("Analyzing pros and cons...")
    pros_cons = comprehensive_analyzer.generate_pros_cons_analysis(performance_report)
    
    # Save detailed results
    print("Saving results...")
    output_files = comprehensive_analyzer.save_detailed_results(
        performance_report, pros_cons, results
    )
    
    # Display summary
    summary_report = comprehensive_analyzer.format_summary_report(performance_report, pros_cons)
    print("\n" + summary_report)
    
    # Show output file locations
    print("\nOutput Files Generated:")
    for output_type, filepath in output_files.items():
        print(f"  {output_type.upper()}: {filepath}")
    
    print(f"\nAnalysis complete! Check the '{comprehensive_analyzer.output_dir}' directory for detailed results.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive pathfinding algorithm comparison and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_comprehensive_comparison.py
  python run_comprehensive_comparison.py --single-robot
  python run_comprehensive_comparison.py --algorithms "A* Manhattan,Gemini API"
  python run_comprehensive_comparison.py --file custom_environment.txt --output results_dir
        """
    )
    
    parser.add_argument(
        '--file',
        type=str,
        default='robot_room.txt',
        help='Path to robot environment file (default: robot_room.txt)'
    )
    
    parser.add_argument(
        '--single-robot',
        action='store_true',
        help='Use only the first robot for comparison (faster)'
    )
    
    parser.add_argument(
        '--algorithms',
        type=str,
        help='Comma-separated list of algorithms to include'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='comparison_results',
        help='Output directory for results (default: comparison_results)'
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum API retry attempts (default: 3)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='API timeout in seconds (default: 30)'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Configure comparison
    config = ComparisonConfig(
        max_retries=args.max_retries,
        timeout_seconds=args.timeout,
        output_directory=args.output
    )
    
    if args.algorithms:
        config.include_algorithms = [alg.strip() for alg in args.algorithms.split(',')]
    
    # Run comprehensive analysis
    run_comprehensive_analysis(args.file, config, args.single_robot)


if __name__ == "__main__":
    main()