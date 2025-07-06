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
import random


def get_input():
    while True:
        try:
            rows = int(input("Enter the number of rows: "))
            cols = int(input("Enter the number of columns: "))
            num_robots = int(input("Enter the number of robots: "))
            if rows <= 0 or cols <= 0 or num_robots <= 0:
                raise ValueError
            if num_robots > rows * cols // 2:
                print("Too many robots for the space. Try fewer.")
                continue
            return rows, cols, num_robots
        except ValueError:
            print("Please enter positive integers.")


def generate_grid(rows, cols):
    total_cells = rows * cols
    max_obstacles = total_cells // 2
    num_obstacles = random.randint(0, max_obstacles)

    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    obstacle_positions = random.sample(range(total_cells), num_obstacles)

    for pos in obstacle_positions:
        r, c = divmod(pos, cols)
        grid[r][c] = 1

    return grid


def find_free_positions(grid, count, exclude=None):
    rows, cols = len(grid), len(grid[0])
    free_positions = [
        (r, c) for r in range(rows) for c in range(cols)
        if grid[r][c] == 0 and (exclude is None or (r, c) not in exclude)
    ]
    if count > len(free_positions):
        raise ValueError("Not enough free space.")
    return random.sample(free_positions, count)


def overlay_grid_with_labels(grid, robot_positions, goal_position):
    labeled_grid = [[str(cell) for cell in row] for row in grid]
    for r, c in robot_positions:
        labeled_grid[r][c] = 'R'
    gr, gc = goal_position
    labeled_grid[gr][gc] = 'G'
    return labeled_grid


def write_to_file(filename, rows, cols, robot_positions, goal_position, labeled_grid):
    with open(filename, 'w') as f:
        f.write(f"{rows} {cols}\n")
        f.write(f"{len(robot_positions)}\n")
        for pos in robot_positions:
            f.write(f"{pos[0]} {pos[1]}\n")
        f.write(f"{goal_position[0]} {goal_position[1]}\n")
        for row in labeled_grid:
            f.write(' '.join(row) + '\n')


def print_grid(labeled_grid):
    print("\nGrid Legend: 1=obstacle, 0=free, R=robot, G=goal")
    for row in labeled_grid:
        print(' '.join(row))


def main():
    rows, cols, num_robots = get_input()
    grid = generate_grid(rows, cols)

    robot_positions = find_free_positions(grid, num_robots)
    reserved = set(robot_positions)
    goal_position = find_free_positions(grid, 1, exclude=reserved)[0]

    labeled_grid = overlay_grid_with_labels(
        grid, robot_positions, goal_position)

    filename = "robot_room.txt"
    write_to_file(filename, rows, cols, robot_positions,
                  goal_position, labeled_grid)

    print_grid(labeled_grid)
    print(
        f"\nRobot data and labeled grid written to '{filename}'")


if __name__ == "__main__":
    main()
