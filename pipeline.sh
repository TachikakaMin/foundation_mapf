#!/usr/bin/env bash

height=32
width=32
# generate map files
for density in $(seq 0.1 0.1 0.6); do 
    for component in {1..10}; do
        for go_straight in $(seq 0.75 0.05 0.85); do
            num_maps=$(printf "%.0f" "$(echo "12 + (${density} * 30) - (${component} * 2)" | bc)")
            python data_generation_LACAM/maze_generator.py --num_maps $num_maps --width $((width-2)) --height $((height-2)) --obstacle_density $density --wall_components $component --go_straight $go_straight
        done
    done
done
# python data_generation_LACAM/random_generator.py --num_maps 100

# generate path files
apt update -y
apt install parallel -y
brew install parallel
git submodule update --init --recursive
cd data_generation_LACAM/lacam
cmake -B build && make -C build
cd ../../

# generate path files
for map_file in data/map_files/maze-*/*.map; do
    # Extract the map pattern from the full path (e.g., maze-64-64-10-1-0.1)
    map_pattern=$(basename $(dirname "$map_file"))
    map_name=$(basename "$map_file" .map)
    density=$(echo "$map_name" | awk -F'-' '{print $4}')

    # Define the N values and corresponding path calculations
    for N in 128 96 64 32 16; do
        case $N in
            128) num_paths=$(echo "60 + $density * 2" | bc) ;;
            96) num_paths=$(echo "40 + $density * 1" | bc) ;;
            64) num_paths=$(echo "20 + $density * 0.8" | bc) ;;
            32) num_paths=$(echo "5 + $density * 0.1" | bc) ;;
            16) num_paths=$(echo "2 + $density * 0.1" | bc) ;;
        esac

        # Convert num_paths to an integer
        num_paths=$(printf "%.0f" "$num_paths")

        mkdir -p "data/path_files/${map_pattern}/${map_name}-${N}"
        echo "Processing map: $map_pattern, $map_name, N: $N, num_paths: $num_paths"

        for seed in $(seq 0 $((num_paths - 1))); do
            output_file="data/path_files/${map_pattern}/${map_name}-${N}/${map_name}-${N}-${seed}.path"
            if [ ! -f "$output_file" ]; then
                echo "$map_pattern $map_name $N $seed"
            fi
        done
    done

done | parallel --progress --bar --eta --timeout 10 --colsep ' ' \
    'data_generation_LACAM/lacam/build/main -m data/map_files/{1}/{2}.map -N {3} -s {4} -v 1 -o data/path_files/{1}/{2}-{3}/{2}-{3}-{4}.path'

# # convert path files to bin files
python -m tools.convert_lacam_path_to_bin data/path_files

# train
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train.py --distributed -bs 8
python train.py -bs 2048 --bilinear -flc 16 -nw 8

# eval
python eval_test.py --model_path model_checkpoint_epoch_4.pth --dataset_paths data_mapf_gpt/input_data/01-random/validation-random-seed-000/validation-random-seed-000-8/validation-random-seed-000-8-1/validation-random-seed-000-8-1-0.bin --show
