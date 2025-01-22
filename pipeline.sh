# generate map files
python data_generation_LACAM/maze_generator.py --num_maps 100
python data_generation_LACAM/random_generator.py --num_maps 100

# generate path files
apt install parallel -y
brew install parallel
git submodule update --init --recursive
cd data_generation_LACAM/lacam
cmake -B build && make -C build
cd ../../

for map_type in random maze; do
    for map_num in {0..99}; do
        # For N=64, generate 85 paths
        N=64
        mkdir -p data/path_files/${map_type}-32-32-20-${map_num}/${map_type}-32-32-20-${map_num}-${N}
        for seed in {0..84}; do
            output_file="data/path_files/${map_type}-32-32-20-${map_num}/${map_type}-32-32-20-${map_num}-${N}/${map_type}-32-32-20-${map_num}-${N}-${seed}.path"
            if [ ! -f "$output_file" ]; then
                echo "$map_type $map_num $N $seed"
            fi
        done

        # For N=16,32,48, generate 5 paths each
        for N in 16 32 48; do
            mkdir -p data/path_files/${map_type}-32-32-20-${map_num}/${map_type}-32-32-20-${map_num}-${N}
            for seed in {0..4}; do
                output_file="data/path_files/${map_type}-32-32-20-${map_num}/${map_type}-32-32-20-${map_num}-${N}/${map_type}-32-32-20-${map_num}-${N}-${seed}.path"
                if [ ! -f "$output_file" ]; then
                    echo "$map_type $map_num $N $seed"
                fi
            done
        done
    done
done | parallel --progress --bar --eta --timeout 10 --colsep ' ' \
    'data_generation_LACAM/lacam/build/main -m data/map_files/{1}-32-32-20/{1}-32-32-20-{2}.map -N {3} -s {4} -v 1 -o data/path_files/{1}-32-32-20-{2}/{1}-32-32-20-{2}-{3}/{1}-32-32-20-{2}-{3}-{4}.path'

# convert path files to bin files
python -m tools.convert_lacam_path_to_bin data/path_files

# train
torchrun --nproc_per_node=8 train.py --distributed -bs 8
# or python train.py -bs 64

# eval
python eval_test.py --model_path model_checkpoint_epoch_5.pth \
--dataset_paths data/input_data/maze-32-32-20-5/maze-32-32-20-5-64/maze-32-32-20-5-64-0/maze-32-32-20-5-64-0-0.bin \
--steps 300 