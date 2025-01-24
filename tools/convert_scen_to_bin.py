import struct
import os
from concurrent.futures import ThreadPoolExecutor
import glob
from tqdm import tqdm
import pickle
from .utils import create_distance_map, read_map, parse_file_name
from threading import Lock

distance_map_lock = Lock()

def parse_scen_line(line):
    """Parse a single line from scen file."""
    parts = line.strip().split('\t')
    if len(parts) != 9:
        return None
    
    try:
        start_x = int(parts[5])
        start_y = int(parts[4])
        goal_x = int(parts[7])
        goal_y = int(parts[6])
        
        return {
            'start': (start_x, start_y),
            'goal': (goal_x, goal_y),
        }
    except (ValueError, IndexError):
        return None

def convert_scen_to_bin(file_name):
    """Convert a scen file to binary format."""
    # Create output directory   
    output_file = file_name.replace("scens", "input_data").replace(".scen", ".bin")
    if os.path.exists(output_file):
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Read scen file
    with open(file_name, "r") as f:
        lines = f.readlines()
    
    # Skip header line
    if not lines:
        return
    lines = lines[1:]  # Skip "version 1" line
    agent_num = len(lines)
    start_locations = []
    goal_locations = []
    # Process each scenario
    for i, line in enumerate(lines):
        scenario = parse_scen_line(line)
        if not scenario:
            continue
        start_locations.append(scenario['start'])
        goal_locations.append(scenario['goal'])
    with open(output_file, "wb") as f:
        f.write(struct.pack("H", agent_num))
        for i in range(agent_num):
            f.write(struct.pack("HH", start_locations[i][0], start_locations[i][1]))
        for i in range(agent_num):
            f.write(struct.pack("HH", goal_locations[i][0], goal_locations[i][1]))
        for i in range(agent_num):
            f.write(struct.pack("H", 0))
    map_name, _ = parse_file_name(file_name)
    dir_prefix = map_name.split("/")[0]
    distance_map_path = os.path.join(
        dir_prefix,
        "distance_maps",
        f"{os.path.basename(map_name).split('.')[0]}.pkl",
    )
    if not os.path.exists(distance_map_path):
        # Add lock around distance map creation and saving
        with distance_map_lock:
            # Check again in case another thread created it while we were waiting
            if not os.path.exists(distance_map_path):
                map_data = read_map(map_name)
                distance_map = create_distance_map(map_data)
                os.makedirs(os.path.dirname(distance_map_path), exist_ok=True)
                pickle.dump(distance_map, open(distance_map_path, "wb"))

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python convert_scen_to_bin.py <path_to_scen_file_dir>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    # Recursively find all .scen files
    scen_files = glob.glob(os.path.join(input_dir, "**/*.scen"), recursive=True)
    # Define scenarios to filter out
    filtered_scenarios = [
        "den520d", "Boston_0_256", "Berlin_1_256", "brc202d",
        "Paris_1_256", "ht_chantry", "ht_mansion_n", "lak303d",
        "lt_galowstemplar_n", "orz900d", "ost003d", "w_woundedcoast"
    ]
    # Filter out specified scenarios
    scen_files = [
        f for f in scen_files 
        if not any(os.path.basename(f).startswith(scenario) for scenario in filtered_scenarios)
    ]
    if not scen_files:
        print(f"No .scen files found in directory: {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(scen_files)} .scen files to process")
    
    with ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(convert_scen_to_bin, scen_files),
                total=len(scen_files),
                desc="Converting files"
            )
        )
