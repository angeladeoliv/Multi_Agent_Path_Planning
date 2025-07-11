from interpret_environment import read_robot_file
from robot_partial_knowledge import Robot
from grid_renderer import print_grid_initial  # optional
import time


def update_all_robot_positions(robots):
    for robot in robots:
        # Clear all old 'R's (in case shared)
        for i in range(robot.grid_rows):
            for j in range(robot.grid_cols):
                if robot.local_map[i][j] == 'R':
                    robot.local_map[i][j] = '0'

        # Re-mark the current robot positions
        for rid, (x, y) in Robot.robot_positions.items():
            robot.local_map[x][y] = 'R'


def run_simulation():
    filepath = "robot_room.txt"
    try:
        grid_size, robot_starts, goal_pos, full_map = read_robot_file(filepath)
    except FileNotFoundError:
        print(
            f"Error: '{filepath}' not found. Run 'create_environment.py' first.")
        return

    print("--- Environment Loaded ---")
    print(f"Grid Size: {grid_size}")
    print(f"Robot Starts: {robot_starts}")
    print(f"Goal: {goal_pos}")
    print_grid_initial(full_map)

    # === Initialize robots ===
    robot1 = Robot(
        robot_id=1, start_pos=robot_starts[0], goal_pos=goal_pos, grid_dimensions=grid_size, full_map=full_map)

    if len(robot_starts) > 1:
        robot2 = Robot(
            robot_id=2, start_pos=robot_starts[1], goal_pos=goal_pos, grid_dimensions=grid_size, full_map=full_map)
        robots = [robot1, robot2]
    else:
        robots = [robot1]

    # === Plan initial paths ===
    for robot in robots:
        robot.plan_path()

    # === Run simulation (5 steps or until all reach goal) ===
    for t in range(10):
        print(f"\n--- Time Step {t} ---")

        # Move each robot
        for robot in robots:
            if robot.position != (robot.goal_row, robot.goal_col):
                robot.move()

        # Share and receive knowledge
        for sender in robots:
            shared_data = sender.share_knowledge()
            for receiver in robots:
                if receiver != sender:
                    receiver.receive_knowledge(shared_data)

        update_all_robot_positions(robots)

        # Print local maps
        for robot in robots:
            robot.print_local_map()

        # Stop early if all robots reached goal
        if all(robot.position == (robot.goal_row, robot.goal_col) for robot in robots):
            print("\n✅ All robots reached the goal!")
            break

        time.sleep(0.5)  # optional pause to simulate time


if __name__ == "__main__":
    run_simulation()
