from pogema import GridConfig, Grid

import yaml



if __name__ == '__main__':

    agent_numbers = [ 8, 16, 24, 32, 48, 64 ]
    folder_names = [
        'random_maps',
    ]

    for folder in folder_names:
        maps_path = folder + "/maps.yaml"
        print(f"Loading maps from {maps_path}")
        with open(maps_path, 'r') as f:
            maps = yaml.safe_load(f)
        print(maps)