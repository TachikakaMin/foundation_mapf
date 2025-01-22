import struct
import glob
from .visualize_path import visualize_path
from .utils import parse_file_name


def parse_bin_path(bin_dir, agent_num):
    # Get all bin files sorted by timestep
    bin_files = sorted(
        glob.glob(str(bin_dir / "*.bin")),
        key=lambda x: int(x.split("-")[-1].split(".")[0]),
    )
    if not bin_files:
        raise ValueError(f"No bin files found in {bin_dir}")

    # Store all positions for path tracking
    all_paths = []
    for bin_file in bin_files:

        agent_locations = []
        with open(bin_file, "rb") as f:
            for _ in range(agent_num):
                cur_x, cur_y = struct.unpack("BB", f.read(2))
                agent_locations.append((cur_x, cur_y))
        all_paths.append(agent_locations)
    return all_paths


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python visualize_bin_path.py <path_to_bin_directory>")
        sys.exit(1)
    folder_path = sys.argv[1]
    map_name, agent_num, path_name = parse_file_name(folder_path)
    all_paths = parse_bin_path(folder_path, agent_num)
    goal_locations = all_paths[-1]
    visualize_path(all_paths, goal_locations, path_name, show=True)
