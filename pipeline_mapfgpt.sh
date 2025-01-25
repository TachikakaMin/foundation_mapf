python tools/convert_mapfgpt_to_map_files.py

# Generate paths for each map category and map
for category in 01-random 02-mazes; do
    # Find all .map files in the category directory
    find "data_mapf_gpt/map_files/${category}" -name "*.map" | while read map_file; do
        map_name=$(basename "$map_file" .map)
        

        # generate 10 paths each
        for N in 8 16 24 32 48 64; do
            mkdir -p "data_mapf_gpt/path_files/${category}/${map_name}/${map_name}-${N}"
            for seed in {0..0}; do
                output_file="data_mapf_gpt/path_files/${category}/${map_name}/${map_name}-${N}/${map_name}-${N}-${seed}.path"
                if [ ! -f "$output_file" ]; then
                    echo "$map_file $N $seed $output_file"
                fi
            done
        done
    done
done | parallel --progress --bar --eta --timeout 10 --colsep ' ' \
    'data_generation_LACAM/lacam/build/main -m {1} -N {2} -s {3} -v 1 -o {4}'

# # Generate paths for each map category and map
# for category in 03-warehouse; do
#     # Find all .map files in the category directory
#     find "data_mapf_gpt/map_files/${category}" -name "*.map" | while read map_file; do
#         map_name=$(basename "$map_file" .map)
        

#         # generate 10 paths each
#         for N in 32 64 96 128 160 192; do
#             mkdir -p "data_mapf_gpt/path_files/${category}/${map_name}/${map_name}-${N}"
#             for seed in {0..9}; do
#                 output_file="data_mapf_gpt/path_files/${category}/${map_name}/${map_name}-${N}/${map_name}-${N}-${seed}.path"
#                 if [ ! -f "$output_file" ]; then
#                     echo "$map_file $N $seed $output_file"
#                 fi
#             done
#         done
#     done
# done | parallel --progress --bar --eta --timeout 10 --colsep ' ' \
#     'data_generation_LACAM/lacam/build/main -m {1} -N {2} -s {3} -v 1 -o {4}'

# Generate paths for each map category and map
for category in 04-movingai; do
    # Find all .map files in the category directory
    find "data_mapf_gpt/map_files/${category}" -name "*.map" | while read map_file; do
        map_name=$(basename "$map_file" .map)
        

        # generate 10 paths each
        for N in 64 128 192 256; do
            mkdir -p "data_mapf_gpt/path_files/${category}/${map_name}/${map_name}-${N}"
            for seed in {0..0}; do
                output_file="data_mapf_gpt/path_files/${category}/${map_name}/${map_name}-${N}/${map_name}-${N}-${seed}.path"
                if [ ! -f "$output_file" ]; then
                    echo "$map_file $N $seed $output_file"
                fi
            done
        done
    done
done | parallel --progress --bar --eta --timeout 10 --colsep ' ' \
    'data_generation_LACAM/lacam/build/main -m {1} -N {2} -s {3} -v 1 -o {4}'

# Generate paths for each map category and map
for category in 05-puzzles; do
    # Find all .map files in the category directory
    find "data_mapf_gpt/map_files/${category}" -name "*.map" | while read map_file; do
        map_name=$(basename "$map_file" .map)
        

        # generate 1 paths each
        for N in 2 3 4; do
            mkdir -p "data_mapf_gpt/path_files/${category}/${map_name}/${map_name}-${N}"
            for seed in {0..9}; do
                output_file="data_mapf_gpt/path_files/${category}/${map_name}/${map_name}-${N}/${map_name}-${N}-${seed}.path"
                if [ ! -f "$output_file" ]; then
                    echo "$map_file $N $seed $output_file"
                fi
            done
        done
    done
done | parallel --progress --bar --eta --timeout 10 --colsep ' ' \
    'data_generation_LACAM/lacam/build/main -m {1} -N {2} -s {3} -v 1 -o {4}'

# Convert path files to bin files
python -m tools.convert_lacam_path_to_bin data_mapf_gpt/path_files

