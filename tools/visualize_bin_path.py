import struct
from .visualize_path import visualize_path


def parse_bin_path(bin_file):
    # Store all positions for path tracking
    all_paths = []
    with open(bin_file, "rb") as f:
        steps = struct.unpack("H", f.read(2))[0]
        agent_num = struct.unpack("H", f.read(2))[0]
        for _ in range(steps):
            agent_locations = []
            for _ in range(agent_num):
                cur_x, cur_y = struct.unpack("BB", f.read(2))
                agent_locations.append((cur_x, cur_y))
            for _ in range(agent_num):
                action = struct.unpack("B", f.read(1))[0]
            all_paths.append(agent_locations)
    return all_paths


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m tools.visualize_bin_path <path_to_bin_file>")
        sys.exit(1)
    file_name = sys.argv[1]
    all_paths = parse_bin_path(file_name)
    goal_locations = [all_paths[-1] for _ in range(len(all_paths))]
    visualize_path(all_paths, goal_locations, file_name, show=True)