# RAILGUN: A Unified Convolutional Policy for Multi-Agent Path Finding Across Different Environments and Tasks

[Yimin Tang*](https://sites.google.com/view/yimintang), [Xiao Xiong*](https://openreview.net/profile?id=~Xiao_Xiong2), [Jingyi Xi](https://openreview.net/profile?id=~Jingyi_Xi1), [Jiaoyang Li](https://jiaoyangli.me/), [Erdem Bıyık](https://ebiyik.github.io/), [Sven Koenig](https://idm-lab.org/)

### TL;DR: We present the first centralized learning-based method for MAPF, called RAILGUN, which generates actions based on maps rather than individual agents.

Our insight is that in a valid MAPF solution, there will be no collision, which means there can be at most one agent in each map grid cell in each timestep. At any timestep, each agent chooses one of the five edges of its grid cell as its action. Therefore, if we remove all edges that the agents do not use at each timestep, we find that a valid MAPF solution can be viewed as a series of specialized graphs.



<p align="center">
<img width="1021" height="447" alt="image" src="https://github.com/user-attachments/assets/572f1f5e-9bc4-49af-9cf7-a761a4b11c87" />
</p>

<p align="center">
  <img width="504" height="501" alt="image" src="https://github.com/user-attachments/assets/d55687b9-3cdb-4517-ab2d-9aa6712a9fc5" />
</p>

## Abstract

Multi-Agent Path Finding (MAPF), which focuses on finding collision-free paths for multiple robots, is crucial for applications ranging from aerial swarms to warehouse automation. Solving MAPF is NP-hard so learning-based approaches for MAPF have gained attention, particularly those leveraging deep neural networks. Nonetheless, despite the community's continued efforts, all learning-based MAPF planners still rely on decentralized planning due to variability in the number of agents and map sizes. We have developed the first centralized learning-based policy for MAPF problem called RAILGUN. RAILGUN is not an agent-based policy but a map-based policy. By leveraging a CNN-based architecture, RAILGUN can generalize across different maps and handle any number of agents. We collect trajectories from rule-based methods to train our model in a supervised way. In experiments, RAILGUN outperforms most baseline methods and demonstrates great zero-shot generalization capabilities on various tasks, maps and agent numbers that were not seen in the training dataset.


## Installation
```bash
pip install torch numpy tqdm tensorboard psutil matplotlib
apt update -y
apt install parallel -y
brew install parallel
cd data_generation_LACAM
git clone --recursive https://github.com/Kei18/lacam3.git && cd lacam3
cmake -B build && make -C build
cd ../../
```

## Data Generation

### Map Generation

Generate maps with different density, component and go_straight.

```bash
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
```

### Path Generation

Generate paths with different number of agents and seeds.

```bash
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
        for seed in $(seq 1 ${num_paths}); do
            output_file="data/path_files/${map_pattern}/${map_name}-${N}/${map_name}-${N}-${seed}.path"
            if [ ! -f "$output_file" ]; then
                echo "$map_pattern $map_name $N $seed"
            fi
        done
    done

done | parallel --progress --bar --eta --timeout 10 --colsep ' ' \
    'data_generation_LACAM/lacam3/build/main -m data/map_files/{1}/{2}.map -N {3} -s {4} -v 1 -o data/path_files/{1}/{2}-{3}/{2}-{3}-{4}.path'
```

### Convert Path to Input Data And Precompute Distance Maps

```bash
python -m tools.convert_lacam_path_to_bin data/path_files
python -m tools.precompute_distance_maps data/map_files
```

## Train

```bash
# single GPU
python train.py --batch_size 64
# multi-GPUS
torchrun --nproc_per_node=8 train.py --batch_size 8 --distributed 
```

## Evaluation Test
```bash
python eval_test.py --model_path model_checkpoint_epoch_4.pth --dataset_paths data/input_data/maze-32-32-60-1-75/maze-32-32-60-1-75-0-16/maze-32-32-60-1-75-0-16-1.bin --show
```
