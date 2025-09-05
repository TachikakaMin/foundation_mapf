#!/bin/bash

height=32
width=32

# 创建输出目录
mkdir -p data/map_files/

# generate map files
for density in $(seq 0.1 0.1 0.6); do 
    for component in $(seq 1 10); do
        for go_straight in $(seq 0.75 0.05 0.85); do
            num_maps=$(printf "%.0f" "$(echo "12 + (${density} * 30) - (${component} * 2)" | bc)")
            echo "Generating maps with density=$density, component=$component, go_straight=$go_straight, num_maps=$num_maps"
            python data_generation_LACAM/maze_generator.py --num_maps $num_maps --width $((width-2)) --height $((height-2)) --obstacle_density $density --wall_components $component --go_straight $go_straight
        done
    done
done