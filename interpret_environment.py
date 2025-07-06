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

from robot import Robot


def read_robot_file(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # --- Parse top section ---
    grid_rows, grid_cols = map(int, lines[0].split())
    num_robots = int(lines[1])

    robot_positions = []
    for i in range(2, 2 + num_robots):
        r, c = map(int, lines[i].split())
        robot_positions.append((r, c))

    goal_row, goal_col = map(int, lines[2 + num_robots].split())
    goal_position = (goal_row, goal_col)

    # --- Parse grid ---
    grid_lines = lines[3 + num_robots:]
    grid = [line.split() for line in grid_lines]

    return (grid_rows, grid_cols), robot_positions, goal_position, grid


def main():
    filepath = "robot_room.txt"  # or provide full path
    grid_size, robot_starts, goal_pos, grid = read_robot_file(filepath)

    print(f"Grid size: {grid_size}")
    print(f"Goal position: {goal_pos}")
    print("\nGrid:")
    for row in grid:
        print(' '.join(row))

    print("\nRobots:")
    robots = []
    for pos in robot_starts:
        robot = Robot(start_pos=pos, grid_dimensions=grid_size,
                      goal_pos=goal_pos, grid=grid)
        robots.append(robot)
        print(robot)

    robots[0].state()
    print()
    print(robots[0].record)
    robots[0].state()
    print()
    print(robots[0].record)


if __name__ == "__main__":
    main()
