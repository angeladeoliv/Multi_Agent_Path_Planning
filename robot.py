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
    def __init__(self, start_pos, grid_dimensions, goal_pos, gridx):
        self.start_row, self.start_col = start_pos
        self.goal_row, self.goal_col = goal_pos
        self.grid_rows, self.grid_cols = grid_dimensions

        self.gridx = gridx  # robot's discovered grid (starts full of Xs)

        self.position = start_pos
        self.pos_x, self.pos_y = start_pos  # current position in (row, col)
        self.record = []  # stores [symbol, row, col] of discovered spaces

        self.gridx[self.goal_row][self.goal_col] = 'G'

    def state(self, true_grid):
        directions = [  # 8 directions + center
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),  (0, 0),  (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        for dx, dy in directions:
            new_x = self.pos_x + dx
            new_y = self.pos_y + dy

            if 0 <= new_x < self.grid_rows and 0 <= new_y < self.grid_cols:
                symbol = self.get_symbol_at(true_grid, new_x, new_y)

                if [symbol, new_x, new_y] not in self.record:
                    self.record.append([symbol, new_x, new_y])

                # Update the robot's known grid
                self.gridx[new_x][new_y] = symbol

    def get_symbol_at(self, grid, row, col):
        if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
            return grid[row][col]
        else:
            raise IndexError(f"Coordinate ({row}, {col}) is out of bounds.")

    def __repr__(self):
        emoji_map = {
            '0': '\u25FB\uFE0F',     # ◻️ White square = free space
            '1': '\U0001FAA8',       # 🪨 Rock = obstacle
            'R': '\U0001F916',       # 🤖 Robot
            'G': '\U0001F6A9',       # 🚩 Red flag (goal)
            'X': '\u2B1B',           # ⬛ Black square (unknown)
        }

        # Convert gridx to emoji representation
        grid_str = '\n'.join(
            ' '.join(emoji_map.get(cell, cell) for cell in row)
            for row in self.gridx
        )

        return f"Robot POV Grid:\n{grid_str}"
