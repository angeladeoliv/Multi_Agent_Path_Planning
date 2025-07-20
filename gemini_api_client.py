"""
-------------------------------------------------------
Gemini API Client
-------------------------------------------------------
Author:  Rahnuma Haque, Suvethan Yogathasan
ID:  169024593, 169039244
Email: haqu4593@mylaurier.ca, yoga9244@mylaurier.ca
__updated__ = "2025-07-20"
-------------------------------------------------------
"""

import os
import time
import logging
import re
from typing import List, Tuple, Optional
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()


class GeminiAPIClient:
    """Client for interacting with Google's Gemini API for pathfinding tasks."""
    
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, timeout: int = 30):
        """
        Initialize the Gemini API client.
        
        Args:
            api_key: Optional API key. If not provided, will use GEMINI_API_KEY from environment
            max_retries: Maximum number of retry attempts for failed requests
            timeout: Timeout in seconds for API requests
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.model_name = 'gemini-1.5-flash'
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.FileHandler('gemini_api.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
        
        # Configure API
        self._configure_api(api_key)
        
    def _configure_api(self, api_key: Optional[str] = None) -> None:
        """Configure the Gemini API with proper authentication."""
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set it in your .env file or pass it directly."
            )
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.logger.info("Gemini API configured successfully")
        
    def create_prompt(self, grid: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int]) -> str:
        """
        Create a formatted prompt for the Gemini API.
        
        Args:
            grid: 2D grid representation of the environment
            start: Starting position as (row, col)
            goal: Goal position as (row, col)
            
        Returns:
            Formatted prompt string
        """
        grid_str = "\n".join([" ".join(row) for row in grid])
        
        prompt = f"""You are a path-planning robot. The following grid represents an environment:

Legend:
- 0 = free space
- 1 = obstacle
- R = robot's starting position
- G = goal position

Grid:
{grid_str}

Start position: {start}
Goal position: {goal}

Rules:
- Only move up, down, left, or right (no diagonals)
- Avoid obstacles ('1')
- Provide the shortest path as a list of coordinates
- Return ONLY the coordinate list in this exact format: [(r1,c1), (r2,c2), ...]
- DO NOT include any explanations, code blocks, or additional text
- DO NOT use backticks or markdown formatting
- Return only the coordinate list on a single line

Example format: [(0,0), (0,1), (1,1), (2,1)]
"""
        return prompt
        
    def _make_api_call(self, prompt: str) -> Optional[str]:
        """
        Make a single API call to Gemini.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            Response text or None if failed
        """
        try:
            self.logger.info("Sending prompt to Gemini API")
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                self.logger.info("Received successful response from Gemini API")
                return response.text.strip()
            else:
                self.logger.warning("Received empty response from Gemini API")
                return None
                
        except Exception as e:
            self.logger.error(f"API call failed: {str(e)}")
            return None
            
    def _exponential_backoff(self, attempt: int) -> None:
        """
        Implement exponential backoff delay.
        
        Args:
            attempt: Current attempt number (0-based)
        """
        delay = min(2 ** attempt, 8)  # Cap at 8 seconds
        self.logger.info(f"Waiting {delay} seconds before retry (attempt {attempt + 1})")
        time.sleep(delay)
        
    def generate_path(self, grid: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Generate a path using the Gemini API with retry logic.
        
        Args:
            grid: 2D grid representation of the environment
            start: Starting position as (row, col)
            goal: Goal position as (row, col)
            
        Returns:
            List of coordinates representing the path, or None if failed
        """
        prompt = self.create_prompt(grid, start, goal)
        
        for attempt in range(self.max_retries):
            if attempt > 0:
                self._exponential_backoff(attempt - 1)
                
            response_text = self._make_api_call(prompt)
            
            if response_text:
                path = self.parse_response(response_text)
                if path and self.validate_path(path, grid, start, goal):
                    self.logger.info(f"Successfully generated valid path on attempt {attempt + 1}")
                    return path
                else:
                    self.logger.warning(f"Invalid path received on attempt {attempt + 1}")
            else:
                self.logger.warning(f"No response received on attempt {attempt + 1}")
                
        self.logger.error(f"Failed to generate valid path after {self.max_retries} attempts")
        return None
        
    def parse_response(self, response_text: str) -> Optional[List[Tuple[int, int]]]:
        """
        Parse the Gemini API response to extract coordinate list.
        
        Args:
            response_text: Raw response text from the API
            
        Returns:
            List of coordinate tuples or None if parsing failed
        """
        try:
            self.logger.debug(f"Parsing response: {response_text}")
            
            # Clean the response text
            cleaned_text = response_text.strip()
            
            # Try multiple parsing strategies
            path = None
            
            # Strategy 1: Look for coordinate list pattern
            coordinate_pattern = r'\[\s*\(\s*\d+\s*,\s*\d+\s*\)(?:\s*,\s*\(\s*\d+\s*,\s*\d+\s*\))*\s*\]'
            match = re.search(coordinate_pattern, cleaned_text)
            
            if match:
                path = self._parse_coordinate_list(match.group())
            else:
                # Strategy 2: Look for individual coordinate pairs
                coord_pairs = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', cleaned_text)
                if coord_pairs:
                    path = [(int(r), int(c)) for r, c in coord_pairs]
                else:
                    # Strategy 3: Look for comma-separated numbers
                    numbers = re.findall(r'\d+', cleaned_text)
                    if len(numbers) >= 4 and len(numbers) % 2 == 0:
                        path = [(int(numbers[i]), int(numbers[i+1])) for i in range(0, len(numbers), 2)]
            
            if path:
                self.logger.info(f"Successfully parsed path with {len(path)} coordinates")
                return path
            else:
                self.logger.warning("Failed to parse any coordinates from response")
                return None
                
        except Exception as e:
            self.logger.error(f"Error parsing response: {str(e)}")
            return None
            
    def _parse_coordinate_list(self, coord_string: str) -> Optional[List[Tuple[int, int]]]:
        """
        Parse a coordinate list string into tuples.
        
        Args:
            coord_string: String containing coordinate list
            
        Returns:
            List of coordinate tuples or None if parsing failed
        """
        try:
            # Remove brackets and split by coordinate pairs
            cleaned = coord_string.strip('[]')
            coord_pairs = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', cleaned)
            return [(int(r), int(c)) for r, c in coord_pairs]
        except Exception as e:
            self.logger.error(f"Error parsing coordinate list: {str(e)}")
            return None
            
    def validate_path(self, path: List[Tuple[int, int]], grid: List[List[str]], 
                     start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """
        Validate that a path is safe and correct.
        
        Args:
            path: List of coordinate tuples representing the path
            grid: 2D grid representation of the environment
            start: Starting position
            goal: Goal position
            
        Returns:
            True if path is valid, False otherwise
        """
        try:
            if not path:
                self.logger.warning("Path is empty")
                return False
                
            # Check if path starts at start position and ends at goal
            if path[0] != start:
                self.logger.warning(f"Path doesn't start at start position. Expected {start}, got {path[0]}")
                return False
                
            if path[-1] != goal:
                self.logger.warning(f"Path doesn't end at goal position. Expected {goal}, got {path[-1]}")
                return False
                
            # Check bounds and obstacles
            grid_height = len(grid)
            grid_width = len(grid[0]) if grid else 0
            
            for i, (row, col) in enumerate(path):
                # Check bounds
                if not (0 <= row < grid_height and 0 <= col < grid_width):
                    self.logger.warning(f"Coordinate {(row, col)} at index {i} is out of bounds")
                    return False
                    
                # Check for obstacles (but allow start 'R' and goal 'G')
                cell_value = grid[row][col]
                if cell_value == '1':
                    self.logger.warning(f"Path goes through obstacle at {(row, col)}")
                    return False
                    
            # Check path connectivity (adjacent cells only)
            for i in range(1, len(path)):
                prev_row, prev_col = path[i-1]
                curr_row, curr_col = path[i]
                
                row_diff = abs(curr_row - prev_row)
                col_diff = abs(curr_col - prev_col)
                
                # Only allow moves to adjacent cells (no diagonals)
                if not ((row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)):
                    self.logger.warning(f"Invalid move from {path[i-1]} to {path[i]} - not adjacent")
                    return False
                    
            self.logger.info("Path validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating path: {str(e)}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Test the client
    client = GeminiAPIClient()
    
    # Simple test grid
    test_grid = [
        ['R', '0', '0'],
        ['0', '1', '0'],
        ['0', '0', 'G']
    ]
    
    start = (0, 0)
    goal = (2, 2)
    
    print("Testing Gemini API Client...")
    path = client.generate_path(test_grid, start, goal)
    
    if path:
        print(f"Generated path: {path}")
    else:
        print("Failed to generate path")