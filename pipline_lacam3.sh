# generate path files
apt update -y
apt install parallel -y

# generate path files
for map_file in data/map_files/maze-*/*.map; do
    # Extract the map pattern from the full path (e.g., maze-64-64-10-1-0.1)
    map_pattern=$(basename $(dirname "$map_file"))
    map_name=$(basename "$map_file" .map)
    density=$(echo "$map_name" | awk -F'-' '{print $4}')

    # Define the N values and corresponding path calculations
    for N in 128 96 64 32 16; do
        case $N in
            128) num_paths=$(echo "50 + $density * 2" | bc) ;;
            96) num_paths=$(echo "30 + $density * 1" | bc) ;;
            64) num_paths=$(echo "10 + $density * 0.8" | bc) ;;
            32) num_paths=$(echo "3 + $density * 0.1" | bc) ;;
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
    'data_generation_LACAM/lacam3/build/main -m data/map_files/{1}/{2}.map -N {3} -s {4} -v 1 -o ~/autodl-tmp/data/path_files/{1}/{2}-{3}/{2}-{3}-{4}.path'