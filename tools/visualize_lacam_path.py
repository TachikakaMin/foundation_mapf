from .visualize_path import visualize_path
from .utils import parse_coordinates


def parse_lacam_file(file_name):
    # 读取path文件
    with open(file_name, "r") as f:
        lines = f.readlines()

    # 解析路径数据
    all_paths = []
    solution_line = -1
    for i, line in enumerate(lines):
        if line.startswith("solution="):
            solution_line = i
            break
    if solution_line == -1:
        print(f"Error: solution line not found in file: {file_name}")
        return
    for line in lines[solution_line + 1 :]:
        # 解析每个时间步的坐标
        coords = parse_coordinates(line.split(":")[1])
        all_paths.append(coords)
    return all_paths


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python visualize_lacam_path.py <path_to_lacam_result_file>")
        sys.exit(1)
    file_name = sys.argv[1]
    all_paths = parse_lacam_file(file_name)
    goal_locations = all_paths[-1]
    visualize_path(all_paths, goal_locations, file_name, show=True)
