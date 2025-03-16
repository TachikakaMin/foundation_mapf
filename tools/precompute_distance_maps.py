import os
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pickle
from .utils import create_distance_map, read_map
import sys

def process_single_map(map_file):
    """Process a single map file to create its distance map."""
    try:
        # Simply replace map_files with distance_maps and .map with .pkl
        distance_map_path = map_file.replace("map_files", "distance_maps").replace(".map", ".pkl")
        
        # Skip if already exists
        if os.path.exists(distance_map_path):
            return None
            
        # Create distance map
        map_data = read_map(map_file)
        distance_map = create_distance_map(map_data)
        
        # Save distance map
        os.makedirs(os.path.dirname(distance_map_path), exist_ok=True)
        with open(distance_map_path, "wb") as f:
            pickle.dump(distance_map, f)
            
        return map_file
    except Exception as e:
        print(f"Error processing {map_file}: {str(e)}")
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python -m tools.precompute_distance_maps <map_files_dir>")
        sys.exit(1)
        
    map_dir = sys.argv[1]
    if not os.path.exists(map_dir):
        print(f"Directory not found: {map_dir}")
        sys.exit(1)
        
    # Find all map files in the input directory
    map_files = glob.glob(os.path.join(map_dir, "**/*.map"), recursive=True)
    
    if not map_files:
        print(f"No map files found in {map_dir}!")
        return
        
    print(f"Found {len(map_files)} map files")
    
    # Process maps in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        processed_files = list(tqdm(
            executor.map(process_single_map, map_files),
            total=len(map_files),
            desc="Computing distance maps"
        ))
    
    # Count successfully processed files
    successful = [f for f in processed_files if f is not None]
    print(f"\nSuccessfully processed {len(successful)} maps")
    
if __name__ == "__main__":
    main() 