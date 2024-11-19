#!/bin/bash

# Invoke random_generator.py
python3 map_generation/random_generator.py --number_of_maps 5

# Invoke maze_generator.py
python3 map_generation/maze_generator.py --number_of_maps 5

# Invoke get_path.py
python3 map_generation/get_path.py --seed_range 5 --agent_numbers 8 16 32 64