"""
-------------------------------------------------------
[updated robot class]
-------------------------------------------------------
Author:  Rameen Amin
ID:  169068460
Email: amin8460@mylaurier.ca
__updated__ = "2025-07-11"
-------------------------------------------------------
"""

from search_algorithms import a_star_search, manhattan_distance


class Robot:
    robot_positions = {}  # Class-level dict: {robot_id: (x, y)}

    def __init__(self, robot_id, start_pos, grid_dimensions, goal_pos, full_map, sensor_radius=2):

        self.id = robot_id
        self.start_row, self.start_col = start_pos
        self.goal_row, self.goal_col = goal_pos
        self.grid_rows, self.grid_cols = grid_dimensions

        self.full_map = full_map  # actual map (used only for sensing)
        self.sensor_radius = sensor_radius

        self.position = start_pos
        self.pos_x, self.pos_y = start_pos

        self.record = []  # stores (symbol, x, y)
        self.local_map = [
            ['?' for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]

        self.path = []
        self.sense_environment()

    def get_symbol_at(self, grid, row, col):
        if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            return grid[row][col]
        return '?'  # out of bounds is unknown

    def sense_environment(self):
        """Scan local area and update local map."""
        cx, cy = self.pos_x, self.pos_y
        for dx in range(-self.sensor_radius, self.sensor_radius + 1):
            for dy in range(-self.sensor_radius, self.sensor_radius + 1):
                nx, ny = cx + dx, cy + dy
                symbol = self.get_symbol_at(self.full_map, nx, ny)
                if 0 <= nx < self.grid_rows and 0 <= ny < self.grid_cols:
                    self.local_map[nx][ny] = symbol
                    record = (symbol, nx, ny)
                    if record not in self.record:
                        self.record.append(record)

    def plan_path(self):
        """Plans a path from current position to goal using A* and the local map."""
        start = self.position
        goal = (self.goal_row, self.goal_col)
        print(f"[Robot {self.id}] Planning path from {start} to {goal}...")
        path = a_star_search(self.local_map, start, goal,
                             heuristic_func=manhattan_distance)
        if path:
            self.path = path[1:]
            print(f"[Robot {self.id}] Path found: {self.path}")
        else:
            self.path = []
            print(f"[Robot {self.id}] No path found.")

    def move(self):
        """Move one step along the path."""
        if not self.path:
            print(f"[Robot {self.id}] No path to follow.")
            return

        # Remove previous 'R' from map
        prev_x, prev_y = self.pos_x, self.pos_y
        self.local_map[prev_x][prev_y] = '0'

        # Move
        next_pos = self.path.pop(0)
        self.position = next_pos
        self.pos_x, self.pos_y = next_pos
        print(f"[Robot {self.id}] Moved to {self.position}")

        # Update class-level robot position tracking
        Robot.robot_positions[self.id] = (self.pos_x, self.pos_y)

        self.sense_environment()

    def share_knowledge(self):
        """Return known info as a dictionary."""
        return {(x, y): symbol for symbol, x, y in self.record}

    def receive_knowledge(self, shared_data):
        """Update local map with another robot’s knowledge."""
        updated = False
        for (x, y), symbol in shared_data.items():
            if self.local_map[x][y] == '?':
                self.local_map[x][y] = symbol
                self.record.append((symbol, x, y))
                updated = True
        if updated:
            print(f"[Robot {self.id}] Received new info. Replanning path.")
            self.plan_path()

    def print_local_map(self):
        print(f"\n[Robot {self.id}] Local Map View:")
        for row in self.local_map:
            print(" ".join(str(cell) for cell in row))

    # def print_local_map(self):
    #     print(f"\n[Robot {self.id}] Local Map View:")
    #     for i in range(self.grid_rows):
    #         row = []
    #         for j in range(self.grid_cols):
    #             if (i, j) == (self.pos_x, self.pos_y):
    #                 row.append('R')
    #             else:
    #                 row.append(str(self.local_map[i][j]))
    #         print(" ".join(row))

    def __repr__(self):
        return (f"Robot(id={self.id}, start=({self.start_row},{self.start_col}), "
                f"goal=({self.goal_row},{self.goal_col}), current=({self.pos_x},{self.pos_y}))")
