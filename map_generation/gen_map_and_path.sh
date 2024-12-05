#!/bin/bash

# Invoke random_generator.py
python3 map_generation/random_generator.py --number_of_maps 5

# Invoke maze_generator.py
python3 map_generation/maze_generator.py --number_of_maps 5

# Invoke get_path.py
python3 map_generation/get_path.py --seed_range 1000 --agent_numbers 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64