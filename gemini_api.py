"""
-------------------------------------------------------
Gemini API Integration
-------------------------------------------------------
Author:  Rahnuma Haque
ID:  169024593
Email: haqu4593@mylaurier.ca
__updated__ = "2025-07-19"
-------------------------------------------------------
"""

import logging
from google import genai
from interpret_environment import read_robot_file

# set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='gemini_api.log'
)

API_KEY = "AIzaSyDyqtSKRLCB99HbP7v7lxr8r-Y0co6AwtQ"
client = genai.Client(api_key=API_KEY)

def query_gemini(prompt):
    try:
        logging.info("sending prompt to Gemini...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        logging.info("received response from Gemini.")
        return response.text
    except Exception as e:
        logging.error(f"Gemini API request failed: {e}")
        return None

def create_prompt(grid, start, goal):
    grid_str = "\n".join([" ".join(row) for row in grid])
    prompt = f"""
You are a path-planning robot. The following grid represents an environment:

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
- Provide the shortest path as a list of coordinates [(r1,c1), (r2,c2), ...]
- DO NOT explain your answer.
- DO NOT include any formatting like code blocks or backticks.
- DO NOT say anything before or after the list.
- ONLY return the list of coordinates, as plain text, on a single line.

"""
    return prompt

if __name__ == "__main__":
    # Load environment from robot_room.txt
    filepath = "robot_room.txt"
    grid_size, robot_starts, goal, grid = read_robot_file(filepath)

    # Use only the first robot for now
    start = robot_starts[0]

    # Create the prompt
    prompt = create_prompt(grid, start, goal)

    print("\n--- Prompt Sent to Gemini ---")
    print(prompt)

    # Call Gemini
    response = query_gemini(prompt)

    # write response to file
    with open("gemini_response.txt", "w") as file:
        file.write(response if response else "No response received")
    logging.info("Gemini's response saved to gemini_response.txt")

    print("\n--- Gemini's Response ---")
    print(response)
