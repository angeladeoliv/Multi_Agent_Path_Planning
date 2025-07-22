"""
-------------------------------------------------------
Gemini API Client
-------------------------------------------------------
Author:  Rahnuma Haque, Suvethan Yogathasan, Bilal Rashid
ID:  169024593, 169039244, 210648510
Email: haqu4593@mylaurier.ca, yoga9244@mylaurier.ca, rash8510@mylaurier.ca
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

load_dotenv()

class GeminiAPIClient:
    
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, timeout: int = 30):

        self.max_retries = max_retries
        self.timeout = timeout
        self.model_name = 'gemini-1.5-flash'
        
        # set up logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.FileHandler('gemini_api.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
        
        self._configure_api(api_key)
        
    def _configure_api(self, api_key: Optional[str] = None) -> None:
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

        delay = min(2 ** attempt, 8)  # cap at 8 seconds
        self.logger.info(f"Waiting {delay} seconds before retry (attempt {attempt + 1})")
        time.sleep(delay)
        
    def generate_path(self, grid: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:

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

        try:
            self.logger.debug(f"Parsing response: {response_text}")
            
            # clean  response text
            cleaned_text = response_text.strip()
            
            # try multiple parsing strategies
            path = None
            
            coordinate_pattern = r'\[\s*\(\s*\d+\s*,\s*\d+\s*\)(?:\s*,\s*\(\s*\d+\s*,\s*\d+\s*\))*\s*\]'
            match = re.search(coordinate_pattern, cleaned_text)
            
            if match:
                path = self._parse_coordinate_list(match.group())
            else:
                coord_pairs = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', cleaned_text)
                if coord_pairs:
                    path = [(int(r), int(c)) for r, c in coord_pairs]
                else:
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

        try:
            cleaned = coord_string.strip('[]')
            coord_pairs = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', cleaned)
            return [(int(r), int(c)) for r, c in coord_pairs]
        except Exception as e:
            self.logger.error(f"Error parsing coordinate list: {str(e)}")
            return None
            
    def validate_path(self, path: List[Tuple[int, int]], grid: List[List[str]], 
                     start: Tuple[int, int], goal: Tuple[int, int]) -> bool:

        try:
            if not path:
                self.logger.warning("Path is empty")
                return False
                
            # check if path starts at start position and ends at goal
            if path[0] != start:
                self.logger.warning(f"Path doesn't start at start position. Expected {start}, got {path[0]}")
                return False
                
            if path[-1] != goal:
                self.logger.warning(f"Path doesn't end at goal position. Expected {goal}, got {path[-1]}")
                return False
                
            # check bounds and obstacles
            grid_height = len(grid)
            grid_width = len(grid[0]) if grid else 0
            
            for i, (row, col) in enumerate(path):
                # check bounds
                if not (0 <= row < grid_height and 0 <= col < grid_width):
                    self.logger.warning(f"Coordinate {(row, col)} at index {i} is out of bounds")
                    return False
                    
                # check for obstacles (allow start 'R' and goal 'G')
                cell_value = grid[row][col]
                if cell_value == '1':
                    self.logger.warning(f"Path goes through obstacle at {(row, col)}")
                    return False
                    
            # check path connectivity (adjacent cells only)
            for i in range(1, len(path)):
                prev_row, prev_col = path[i-1]
                curr_row, curr_col = path[i]
                
                row_diff = abs(curr_row - prev_row)
                col_diff = abs(curr_col - prev_col)
                
                if not ((row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)):
                    self.logger.warning(f"Invalid move from {path[i-1]} to {path[i]} - not adjacent")
                    return False
                    
            self.logger.info("Path validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating path: {str(e)}")
            return False


if __name__ == "__main__":
    client = GeminiAPIClient()
    
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
