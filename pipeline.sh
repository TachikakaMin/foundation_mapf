height=64
width=64
# generate map files
for density in $(seq 0.1 0.1 0.4); do 
    for component in {1..10}; do
        for go_straight in $(seq 0 0.1 0.9); do
            python data_generation_LACAM/maze_generator.py --num_maps 5 --width $((width-2)) --height $((height-2)) --obstacle_density $density --wall_components $component --go_straight $go_straight
        done
    done
done
# python data_generation_LACAM/random_generator.py --num_maps 100

# generate path files
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

    # For N=256, generate 35 paths
    N=256
    mkdir -p "data/path_files/${map_pattern}/${map_name}-${N}"
    for seed in {0..34}; do
        output_file="data/path_files/${map_pattern}/${map_name}-${N}/${map_name}-${N}-${seed}.path"
        if [ ! -f "$output_file" ]; then
            echo "$map_pattern $map_name $N $seed"
        fi
    done

    # For N=64,128,192, generate 5 paths each
    for N in 64 128 192; do
        mkdir -p "data/path_files/${map_pattern}/${map_name}-${N}"
        for seed in {0..4}; do
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
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train.py --distributed -bs 8
# or python train.py -bs 64

# eval
python eval_test.py --model_path model_checkpoint_epoch_5.pth \
--dataset_paths data/input_data/maze-64-64-10-10-0/maze-64-64-10-10-0-0-64/maze-64-64-10-10-0-0-64-0/maze-64-64-10-10-0-0-64-0-0.bin