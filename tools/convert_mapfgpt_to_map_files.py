import yaml
import os

def convert_map_string(map_str):
    """Convert a map string from YAML format to .map format."""
    # Replace '#' with '@' for obstacles
    return map_str.replace('#', '@').strip()

def create_map_file(map_name, map_str, output_dir):
    """Create a .map file with the converted map string."""
    # Calculate dimensions
    lines = map_str.strip().split('\n')
    height = len(lines)
    width = len(lines[0]) if lines else 0
    
    # Create map file content
    map_content = f"type octile\nheight {height}\nwidth {width}\nmap\n{map_str}"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Write to file
    output_path = os.path.join(output_dir, f"{map_name}.map")
    with open(output_path, 'w') as f:
        f.write(map_content)

def main():
    # Define input YAML files and their corresponding output directories
    yaml_files = [
        ("MAPF_GPT_eval_configs/01-random/maps.yaml", "data_mapf_gpt/map_files/01-random"),
        ("MAPF_GPT_eval_configs/02-mazes/maps.yaml", "data_mapf_gpt/map_files/02-mazes"),
        ("MAPF_GPT_eval_configs/03-warehouse/maps.yaml", "data_mapf_gpt/map_files/03-warehouse"),
        ("MAPF_GPT_eval_configs/04-movingai/maps.yaml", "data_mapf_gpt/map_files/04-movingai"),
        ("MAPF_GPT_eval_configs/05-puzzles/maps.yaml", "data_mapf_gpt/map_files/05-puzzles"),
        ("MAPF_GPT_eval_configs/06-pathfinding/maps.yaml", "data_mapf_gpt/map_files/06-pathfinding"),
    ]
    
    # Process each YAML file
    for yaml_path, output_dir in yaml_files:
        # Skip if YAML file doesn't exist
        if not os.path.exists(yaml_path):
            print(f"Warning: {yaml_path} not found, skipping...")
            continue
            
        # Read YAML file
        with open(yaml_path, 'r') as f:
            maps_data = yaml.safe_load(f)
        
        # Convert each map
        for map_name, map_str in maps_data.items():
            converted_map = convert_map_string(map_str)
            create_map_file(map_name, converted_map, output_dir)
            
        print(f"Processed {len(maps_data)} maps from {yaml_path}")

if __name__ == "__main__":
    main()
