"""
-------------------------------------------------------
[program description]
-------------------------------------------------------
Author:  Max Mortensen
ID:  169065545
Email: mort5545@mylaurier.ca
__updated__ = "2025-07-03"
-------------------------------------------------------
"""


class Robot:
    def __init__(self, start_pos, grid_dimensions, goal_pos, grid):
        self.start_row, self.start_col = start_pos
        self.goal_row, self.goal_col = goal_pos
        self.grid_rows, self.grid_cols = grid_dimensions

        self.grid = grid

        self.position = start_pos
        self.pos_x, self.pos_y = start_pos
        self.record = []

    def state(self):
        pos_x = self.pos_x - 1
        pos_y = self.pos_y - 1

        while pos_x != self.pos_x+2:
            pos_y = self.pos_y-1
            for i in range(3):
                symbol = self.get_symbol_at(self.grid, pos_x, pos_y)
                record = [symbol, pos_x, pos_y]
                if record not in self.record:
                    self.record.append(record)

                pos_y += i

            pos_x += 1

    def get_symbol_at(self, grid, row, col):
        if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
            return grid[row][col]
        else:
            raise IndexError(f"Coordinate ({row}, {col}) is out of bounds.")

    # def record(self):

    # def move(self):

    def __repr__(self):
        return (f"Robot(start=({self.start_row}, {self.start_col}), "
                f"goal=({self.goal_row}, {self.goal_col}), "
                f"grid=({self.grid_rows}x{self.grid_cols}),"
                f"Current X & Y=({self.pos_x -1}, {self.pos_y})")
